import pandas as pd
import numpy as np
import streamlit as st
import os

from filehub_app import upload_dataframe, download_dataframe

# Sanity check for required AWS credentials
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", os.getenv("S3_BUCKET_NAME"))
S3_REGION = st.secrets.get("S3_REGION", os.getenv("S3_REGION"))

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_REGION]):
    st.error("Missing AWS credentials or bucket configuration. Please check your `.streamlit/secrets.toml`.")

# --- Ensure session state for selected_features and feature_selection ---
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = []

if "feature_selection" not in st.session_state:
    st.session_state["feature_selection"] = []

# ----------------- Session State Initialization -----------------
# Ensure all keys used later are initialized before access
for key in ["df", "filename", "selected_features", "analysis_type", "analysis_results", "feature_selector", "feature_selector_options", "analysis_type_selectbox", "dep_var_select", "x_axis_select", "cluster_dropdown", "boxplot_grouping", "boxplot_outliers"]:
    if key not in st.session_state:
        if key in ["df"]:
            st.session_state[key] = None
        elif key in ["selected_features", "feature_selector", "feature_selector_options"]:
            st.session_state[key] = []
        elif key in ["boxplot_outliers"]:
            st.session_state[key] = True
        else:
            st.session_state[key] = None


# ----------------- Centralized DataFrame Handler -----------------
def process_incoming_dataframe(df, filename):
    """
    Handles DataFrame assignment, datetime detection, t-value generation, sampling info, quality checks, and session state updates.
    """
    st.session_state["df"] = df
    st.session_state["filename"] = filename

    # Explicitly reset session state fields to guarantee overwrite
    st.session_state.pop("sampled_data_info", None)
    st.session_state.pop("file_info", None)
    st.session_state.pop("data_cleanliness_msgs", None)

    # --- Sampled Data Detection ---
    sample_cols = [col for col in df.columns if col.startswith("DS_SAMPLE")]
    parsed_samples = []
    for col in sample_cols:
        sample_info_str = df[col].iloc[0]
        try:
            if isinstance(sample_info_str, str) and sample_info_str.startswith("Sampled[") and "]" in sample_info_str:
                parts = sample_info_str.split("[")
                sample_type = parts[1][:-1]
                counts = parts[2][:-1].split("/")
                sample_n = int(counts[0])
                total_n = int(counts[1])
                sample_pct = round((sample_n / total_n) * 100, 2)
                parsed_samples.append({
                    "column": col,
                    "sample_type": sample_type,
                    "sample_n": sample_n,
                    "total_n": total_n,
                    "sample_pct": sample_pct
                })
        except Exception:
            continue
    st.session_state.sampled_data_info = parsed_samples if parsed_samples else None

    # Datetime detection
    def detect_datetime_columns(df):
        datetime_cols = []
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                try:
                    parsed = pd.to_datetime(df[col], errors='raise', infer_datetime_format=True)
                    if pd.api.types.is_datetime64_any_dtype(parsed):
                        datetime_cols.append(col)
                except:
                    continue
        return datetime_cols

    inferred_types = {}
    datetime_fields = detect_datetime_columns(df)
    for col in df.columns:
        if col in datetime_fields:
            # Only convert object/string columns to datetime
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            inferred_types[col] = "datetime"
        else:
            dtype = pd.api.types.infer_dtype(df[col], skipna=True)
            if dtype in ["string", "categorical", "object"]:
                inferred_types[col] = "categorical"
            elif dtype in ["integer", "floating", "mixed-integer-float"]:
                inferred_types[col] = "numeric"
            elif dtype.startswith("datetime"):
                inferred_types[col] = "datetime"
            else:
                inferred_types[col] = "other"

    # Generate 't - fieldname' columns for datetime fields
    datetime_cols = [col for col, t in inferred_types.items() if t == "datetime"]
    for dt_col in datetime_cols:
        try:
            temp_df = df[[dt_col]].copy()
            temp_df["original_index"] = temp_df.index
            temp_df = temp_df.dropna().sort_values(by=dt_col, ascending=True).reset_index()
            temp_df["t_index"] = temp_df.index + 1  # Start at 1
            df[f"t - {dt_col}"] = np.nan
            df.loc[temp_df["original_index"], f"t - {dt_col}"] = temp_df["t_index"].values
            inferred_types[f"t - {dt_col}"] = "numeric"
        except Exception as e:
            st.warning(f"Could not generate t index for {dt_col}: {e}")

    # Always regenerate cleanliness messages and metadata on new file
    st.session_state["data_cleanliness_msgs"] = evaluate_data_cleanliness(df, inferred_types)

    # File info
    file_info = {
        "name": filename,
        "size": round(df.memory_usage(deep=True).sum()/1024/1024, 3) if hasattr(df, "memory_usage") else "N/A",
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum())
    }
    st.session_state["file_info"] = file_info

    # Reset selected features in session state when a new file is uploaded
    st.session_state["selected_features"] = []
    st.session_state["feature_selector"] = []
    st.session_state["feature_selector_options"] = []
    # Set a flag indicating data is loaded
    st.session_state["data_loaded"] = True
def evaluate_data_cleanliness(df, inferred_types):
    messages = []
    missing_ratio = df.isna().mean().mean()
    missing_per_col = df.isna().mean()
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
    duplicate_count = df.duplicated().sum()
    datetime_cols = [col for col, t in inferred_types.items() if t == "datetime"]
    mixed_type_cols = []
    for col in df.select_dtypes(include=["object"]):
        sample = df[col].dropna().astype(str).sample(min(20, len(df[col].dropna())))
        types = sample.apply(lambda x: type(eval(x)) if x.isdigit() or (x.replace('.', '', 1).isdigit() and '.' in x) else type(x)).nunique()
        if types > 1:
            mixed_type_cols.append(col)

    # Cleanliness messages
    if missing_ratio < 0.01:
        messages.append("Dataset is extremely clean. Few or no missing values, well-defined structure.")
    elif missing_ratio < 0.05:
        messages.append("Dataset is clean with minor missing data. All columns appear well-defined.")
    elif missing_ratio < 0.2 or duplicate_count > 0:
        messages.append("Dataset is generally usable but includes some missing values and/or duplicate rows.")
    elif missing_ratio < 0.5:
        messages.append("Dataset contains significant missing values across multiple columns.")
    else:
        messages.append("Dataset has severe missing data issues. Many rows or columns are incomplete.")

    # Structure messages
    if unnamed_cols:
        messages.append("Some columns may be improperly labeled (e.g., 'Unnamed'). Consider checking headers.")
    elif df.shape[0] == 1 or df.shape[1] == 1:
        messages.append("Dataset may lack structure—contains only one row or one column.")
    elif mixed_type_cols:
        messages.append("Some columns contain inconsistent data types. Check for formatting issues.")
    elif datetime_cols:
        messages.append("Likely time-series data detected. Datetime structure parsed successfully.")
    elif missing_ratio > 0.5 and len(datetime_cols) == 0:
        messages.append("Unable to determine clear structure. File may need reformatting before use.")
    return messages[:2]
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.dates as mdates

st.set_page_config(layout="wide")




# --------- Transfer Token Section (FileHub) ---------
transfer_token = st.sidebar.text_input("Enter transfer token to load file from FileHub", key="transfer_token_input")
if st.sidebar.button("Submit Token"):
    if transfer_token:
        with st.spinner("Retrieving file from FileHub..."):
            try:
                from filehub_app import download_dataframe
                df_from_filehub, original_filename = download_dataframe(transfer_token.strip())
                # Unify logic: use same processing as file upload
                process_incoming_dataframe(df_from_filehub, original_filename)
                st.session_state["file_from_filehub"] = True
                st.success(f"Successfully imported file: {original_filename}")
            except Exception as e:
                st.sidebar.error(f"Error retrieving file: {e}")
    else:
        st.sidebar.warning("Please enter a token before submitting.")

# ---------- Main Layout ----------
st.title("DataWizard")
st.subheader("Upload your data to begin analysis")
st.write(
    "This tool lets you upload a CSV or Excel file and quickly view summary statistics, charts, distributions, run regressions, cluster your data, and more. "
    "You can upload your file using the button below or import data by entering a FileHub token."
)

main_file = st.file_uploader("Upload your file here", type=["csv", "xls", "xlsx"], key="main")
st.info(
    "**Need more power tools?**\n\n"
    "[**DataBlender**](https://datablendertool.streamlit.app/): Merge, pivot, and reshape your data.\n\n"
    "[**DataSampler**](https://datasamplertool.streamlit.app/): Create smaller samples from large datasets."
)

 # ---------- File Selection Logic (uploaded file or FileHub) ----------
# Uploaded file always takes precedence over file loaded via transfer token
uploaded_file = main_file

if uploaded_file is not None:
    # Prevent reprocessing if already processed (mimic FileHub behavior)
    if not st.session_state.get("upload_processed", False):
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            if df.shape[0] > 75000:
                st.error("File exceeds 75,000 row limit. Please upload a smaller file.")
            else:
                # Unify logic: use same processing as FileHub
                process_incoming_dataframe(df, uploaded_file.name)
                st.session_state["file_from_filehub"] = False
                st.success(f"File uploaded successfully: {uploaded_file.name}")
                st.session_state["upload_processed"] = True
        except Exception as e:
            st.error(f"Error processing file: {e}")
elif st.session_state.get("df") is not None and st.session_state.get("filename"):
    uploaded_file = type("FakeUpload", (), {"name": st.session_state["filename"], "size": st.session_state["df"].memory_usage(deep=True).sum() if hasattr(st.session_state["df"], "memory_usage") else "N/A"})()
else:
    # On reset or file removal, ensure sampled_data_info and cleanliness messages disappear
    st.session_state.pop("sampled_data_info", None)
    st.session_state.pop("file_info", None)
    st.session_state.pop("data_cleanliness_msgs", None)
    st.session_state["df"] = None
    st.session_state.pop("filename", None)
    st.session_state["file_from_filehub"] = False
    st.session_state["upload_processed"] = False

# ---------- File Loaded State (controls UI) ----------
df = st.session_state.get("df", None)
if df is not None and not df.empty:
    file_loaded = True
else:
    file_loaded = False
st.session_state["file_loaded"] = file_loaded

run_analysis = False

# ---------- UI Rendering (only if data loaded) ----------
if st.session_state.get("file_loaded", False):
    df = st.session_state["df"]
    inferred_types = {}
    # Recalculate inferred_types for UI (since session_state doesn't persist it)
    for col in df.columns:
        dtype = pd.api.types.infer_dtype(df[col], skipna=True)
        if dtype in ["string", "categorical", "object"]:
            inferred_types[col] = "categorical"
        elif dtype in ["integer", "floating", "mixed-integer-float"]:
            inferred_types[col] = "numeric"
        elif dtype.startswith("datetime"):
            inferred_types[col] = "datetime"
        else:
            inferred_types[col] = "other"
        if col.startswith("t - "):
            inferred_types[col] = "numeric"

    # ---------- Reordered Sidebar Layout ----------
    with st.sidebar:
        st.markdown("### 📁 File Summary")
        sampled_info = st.session_state.get("sampled_data_info", [])
        if sampled_info:
            if len(sampled_info) == 1:
                s = sampled_info[0]
                st.markdown(f"""
<div style='background-color:#fff3cd; padding:10px; border-radius:5px; border-left:5px solid #ffeeba'>
<b>Sampled Data Detected</b><br><ul>
<li><b>Sample type:</b> {s['sample_type']}</li>
<li><b>Original size:</b> {s['total_n']}</li>
<li><b>Sample size:</b> {s['sample_n']}</li>
<li><b>Coverage:</b> {s['sample_pct']}%</li>
</ul></div>
""", unsafe_allow_html=True)
            elif len(sampled_info) > 1:
                st.error("**Repeated sampling detected. Validate data accuracy.**")
                with st.expander("View all sample details"):
                    for s in sampled_info:
                        st.markdown(f"""
<b>{s['column']}</b><br>
- Sample type: {s['sample_type']}<br>
- Original size: {s['total_n']}<br>
- Sample size: {s['sample_n']}<br>
- Coverage: {s['sample_pct']}%
""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        dq_messages = st.session_state.get("data_cleanliness_msgs", [])
        if dq_messages:
            for msg in dq_messages:
                msg_lower = msg.lower()
                if any(x in msg_lower for x in [
                    "no missing",
                    "extremely clean",
                    "well-defined",
                    "looks great",
                    "parsed successfully"
                ]):
                    st.success(msg)
                elif any(x in msg_lower for x in ["too many", "poorly structured", "inconsistent", "invalid", "corrupt"]):
                    st.error(msg)
                else:
                    st.warning(msg)
        file_info = st.session_state.get("file_info", None)
        if file_info:
            st.markdown(f"**File Name:** {file_info['name']}")
            st.markdown(f"**File Size:** {file_info['size']} MB")
            st.markdown(f"**Rows:** {file_info['rows']}")
            st.markdown(f"**Columns:** {file_info['cols']}")
            st.markdown(f"**Missing Fields:** {file_info['missing']}")

    left_col, right_col = st.columns([1, 1])
    with left_col:
        st.markdown("### Feature Selection")
        allowed_feature_types = {"categorical", "datetime", "numeric"}
        allowed_features = [col for col in df.columns if inferred_types.get(col) in allowed_feature_types or col.startswith("t - ")]
        # --- FEATURE SELECTION BUTTONS AND MULTISELECT (Safe session state handling) ---
        if st.session_state.get("df") is not None:
            all_features = st.session_state["df"].columns.tolist()
            # Initialize trigger flag if not present
            if "trigger_feature_update" not in st.session_state:
                st.session_state["trigger_feature_update"] = False
            col1, col2 = st.columns([1, 1])
            with col1:
                select_all = st.button("Select All")
            with col2:
                clear_all = st.button("Clear All")

            if select_all:
                st.session_state["selected_features"] = all_features
                st.session_state["trigger_feature_update"] = True

            if clear_all:
                st.session_state["selected_features"] = []
                st.session_state["trigger_feature_update"] = True

            # Multiselect for feature selection, update trigger flag after rendering
            selected_features = st.multiselect(
                "Select features",
                options=st.session_state["df"].columns.tolist(),
                default=st.session_state.get("selected_features", []),
                key="selected_features"
            )
            st.session_state["trigger_feature_update"] = False
        # Analysis type selection
        st.markdown("### Analysis Type")
        analysis_type = st.session_state.get(
            "analysis_type_selectbox",
            "Summary statistics"
        )
        analysis_type = st.selectbox(
            "Select analysis type",
            [
                "Summary statistics",
                "Frequency Table",
                "Histogram",
                "Box Plot",
                "Line Chart",
                "Correlation matrix",
                "Linear Regression",
                "Clustering"
            ],
            index=0,
            key="analysis_type_selectbox"
        )
        # "Run Analysis" button now in main panel, after analysis type selection
        run_analysis = st.button("Run Analysis")
        # X field for Line Chart
        x_field = None
        valid_x_fields = []
        if analysis_type == "Line Chart":
            valid_x_fields = [col for col in df.columns if inferred_types.get(col) == "datetime" or col.startswith("t - ")]
            if not valid_x_fields:
                st.warning("Line charts require a time-based x-axis. No datetime or t-index fields were found.")
            else:
                x_field = st.session_state.get("x_axis_select", valid_x_fields[0] if valid_x_fields else None)
                x_field = st.selectbox("Select a time-based field for the X-axis", options=valid_x_fields, key="x_axis_select", index=0)
        # Clustering k
        cluster_k = None
        if analysis_type == "Clustering":
            cluster_k = st.session_state.get("cluster_dropdown", 3)
            cluster_k = st.selectbox(
                "Select number of clusters (k)",
                options=list(range(2, 11)),
                index=1,
                key="cluster_dropdown"
            )
        # Box plot group col and outliers
        group_col = None
        show_outliers = None
        if analysis_type == "Box Plot":
            group_col = st.session_state.get("boxplot_grouping", "None")
            group_col = st.selectbox(
                "Optional: Select categorical column to group by",
                options=["None"] + [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])],
                key="boxplot_grouping"
            )
            group_col = None if group_col == "None" else group_col
            show_outliers = st.session_state.get("boxplot_outliers", True)
            show_outliers = st.checkbox("Show Outliers", value=True, key="boxplot_outliers")
        else:
            group_col = None
        # Session state for last analysis type
        if "last_analysis_type" not in st.session_state:
            st.session_state.last_analysis_type = None
        if st.session_state.last_analysis_type != analysis_type:
            st.session_state.run_clustering = False
            st.session_state.last_analysis_type = analysis_type
        # Dependent variable for regression
        dependent_var = None
        if analysis_type == "Linear Regression" and selected_features:
            dependent_var = st.session_state.get("dep_var_select", selected_features[0] if selected_features else None)
            dependent_var = st.selectbox(
                "Select dependent variable",
                options=selected_features,
                key="dep_var_select"
            )
    with right_col:
        st.markdown("### Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        # --- Send to FileHub Functionality ---
        st.markdown("")
        try:
            from filehub_app import upload_dataframe
            if st.button("📤 Send to FileHub"):
                st.info("📡 Uploading to FileHub...")
                token = upload_dataframe(
                    df,
                    source_app="DataWizard",
                    original_filename=st.session_state.get("filename", "datawizard_output.csv")
                )
                st.success(f"✅ File sent to FileHub! Transfer Token: `{token}`")
                st.info("You can now use this token in another app (e.g., DataBlender, DataSampler) to retrieve this file.")
        except Exception as e:
            st.error("❌ Upload to FileHub failed.")
            st.exception(e)
    # ---- All the rest of main UI logic, unchanged ----
    # (Paste the rest of the analysis logic here unchanged)

if run_analysis and st.session_state.get("file_loaded", False) and st.session_state["df"] is not None:
    if analysis_type == "Summary statistics":
        st.markdown("## Summary Statistics")

        results = []
        for col in selected_features:
            col_data = df[col].dropna()
            col_type = inferred_types[col]
            missing = df[col].isna().sum()

            if col_type == "numeric":
                if len(col_data) >= 2:
                    x = np.arange(len(col_data))
                    y = col_data.values
                    slope = np.polyfit(x, y, 1)[0]
                    trend = "Positive" if slope > 0 else "Negative" if slope < 0 else "None"
                    seasonality = "Yes" if abs(col_data.autocorr(lag=12)) > 0.3 else "No"
                else:
                    trend = "Insufficient data"
                    seasonality = "Insufficient data"

                results.append({
                    "Field": col,
                    "Type": "Numeric",
                    "Mean": round(col_data.mean(), 3),
                    "Median": round(col_data.median(), 3),
                    "Min": round(col_data.min(), 3),
                    "Max": round(col_data.max(), 3),
                    "Std Dev": round(col_data.std(), 3),
                    "Trend": trend,
                    "Seasonality?": seasonality,
                    "Missing": missing
                })

            elif col_type == "categorical":
                most_common = col_data.mode().iloc[0] if not col_data.mode().empty else "None"
                value_counts = col_data.value_counts()
                least_common = value_counts.idxmin() if not value_counts.empty else "None"

                results.append({
                    "Field": col,
                    "Type": "Categorical",
                    "Mean": "-",
                    "Median": "-",
                    "Min": "-",
                    "Max": "-",
                    "Std Dev": "-",
                    "Trend": "-",
                    "Seasonality?": "-",
                    "Most Common": most_common,
                    "Least Common": least_common,
                    "Missing": missing
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df.set_index("Field"), use_container_width=True)

    elif analysis_type == "Histogram":
        st.markdown("## Histograms")

        numeric_features = [f for f in selected_features if inferred_types[f] == "numeric"]
        non_numeric = [f for f in selected_features if f not in numeric_features]

        for f in non_numeric:
            st.warning(f"Field '{f}' is not numeric and was skipped.")

        if len(numeric_features) > 6:
            st.warning("More than 6 features selected. Only the first 6 will be shown.")
            numeric_features = numeric_features[:6]

        for i, col in enumerate(numeric_features):
            if i % 2 == 0:
                col_left, col_right = st.columns([1, 1])
            target_col = col_left if i % 2 == 0 else col_right

            with target_col:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30, edgecolor='white', color=plt.cm.tab10(i % 10))
                ax.set_title(col)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

    elif analysis_type == "Correlation matrix":
        non_numeric_fields = [f for f in selected_features if inferred_types.get(f) != "numeric"]
        if non_numeric_fields:
            st.error("All selected features must be numeric for correlation analysis. "
                     f"The following fields are not: {', '.join(non_numeric_fields)}")
        elif len(selected_features) < 2:
            st.warning("Please select at least two numeric features to compute correlation.")
        else:
            corr_data = df[selected_features].dropna()
            corr_matrix = corr_data.corr()

            st.markdown("### Correlation Table")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    elif analysis_type == "Linear Regression":
        st.markdown("## Linear Regression")

        non_numeric = [f for f in selected_features if inferred_types.get(f) != "numeric"]
        if non_numeric:
            st.error(f"All selected features must be numeric. The following are not: {', '.join(non_numeric)}")
        elif len(selected_features) < 2:
            st.warning("Please select at least two numeric features (1 dependent + ≥1 independent).")
        elif dependent_var not in selected_features:
            st.error("Dependent variable must be one of the selected fields.")
        else:
            X_cols = [col for col in selected_features if col != dependent_var]
            if len(X_cols) > 8:
                st.error("You can include up to 8 independent variables in the regression.")
            else:
                data = df[[dependent_var] + X_cols].dropna()
                warn_fields = []
                for col in [dependent_var] + X_cols:
                    ratio = df[col].isna().mean()
                    if ratio > 0.2:
                        warn_fields.append((col, ratio))
                if warn_fields:
                    st.warning("Some fields used in this regression have high missing values:\n" +
                               "\n".join([f"{col}: {ratio:.0%} missing" for col, ratio in warn_fields]))
                X = sm.add_constant(data[X_cols])
                y = data[dependent_var]

                model = sm.OLS(y, X).fit()

                eqn_parts = [f"{model.params[name]:.4f} × {name}" for name in X_cols]
                intercept = model.params["const"] if "const" in model.params else 0
                equation_latex = f"$ {dependent_var} = {intercept:.4f} + " + " + ".join(eqn_parts) + " $"
                st.markdown("### Regression Equation")
                st.markdown(equation_latex)

                summary_df = pd.DataFrame({
                    "Coefficient": model.params,
                    "P-Value": model.pvalues,
                })
                summary_df.index.name = "Variable"
                summary_df = summary_df.round(4)

                st.markdown("### Regression Summary")
                st.dataframe(summary_df)

                st.markdown(f"**R²:** {model.rsquared:.4f}")
                if len(X_cols) > 1:
                    st.markdown(f"**Adjusted R²:** {model.rsquared_adj:.4f}")

                if len(X_cols) == 1:
                    st.markdown("### Regression Plot")
                    x_vals = data[X_cols[0]]
                    y_vals = y
                    y_pred = model.predict(X)

                    fig, ax = plt.subplots()
                    ax.scatter(x_vals, y_vals, alpha=0.6, label="Data")
                    ax.plot(x_vals, y_pred, color="red", label="Regression Line")
                    ax.set_xlabel(X_cols[0])
                    ax.set_ylabel(dependent_var)
                    ax.set_title(f"{dependent_var} vs {X_cols[0]}")
                    ax.legend()
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)

    elif analysis_type == "Clustering":
        st.markdown("## K-Means Clustering")

        numeric_selected = [f for f in selected_features if inferred_types.get(f) == "numeric"]
        if len(numeric_selected) < 2:
            st.warning("Please select at least two numeric features for clustering.")
        elif cluster_k is None:
            st.warning("Please select the number of clusters.")
        else:
            try:
                warn_fields = []
                for col in numeric_selected:
                    ratio = df[col].isna().mean()
                    if ratio > 0.2:
                        warn_fields.append((col, ratio))
                if warn_fields:
                    st.warning("Some clustering fields have high missing values:\n" +
                               "\n".join([f"{col}: {ratio:.0%} missing" for col, ratio in warn_fields]))
                X_raw = df[numeric_selected].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)

                kmeans = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                X_raw_with_labels = X_raw.copy()
                X_raw_with_labels["Cluster"] = clusters
                X_raw_with_labels["Cluster Name"] = X_raw_with_labels["Cluster"].apply(lambda x: f"Cluster {x}")

                st.markdown("### Cluster Summary Table")
                cluster_summary = X_raw_with_labels.groupby("Cluster Name").agg(["mean", "count"])
                st.dataframe(cluster_summary, use_container_width=True)

                # PCA for visualization
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
                pca_df["Cluster"] = clusters
                pca_df["Cluster Name"] = pca_df["Cluster"].apply(lambda x: f"Cluster {x}")

                st.markdown("### 2D PCA Scatter Plot (Clusters)")
                fig, ax = plt.subplots()
                for name in pca_df["Cluster Name"].unique():
                    subset = pca_df[pca_df["Cluster Name"] == name]
                    ax.scatter(subset["PC1"], subset["PC2"], label=name, alpha=0.7)

                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_title("K-Means Clustering Results (PCA)")
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during clustering: {e}")

    elif analysis_type == "Frequency Table":
        st.markdown("## Frequency Tables")

        if not selected_features:
            st.warning("Please select at least one field.")
        else:
            categorical_fields = [f for f in selected_features if inferred_types.get(f) == "categorical"]
            non_cat_fields = [f for f in selected_features if f not in categorical_fields]

            if non_cat_fields:
                st.warning(f"The following fields are not categorical and will be skipped: {', '.join(non_cat_fields)}")

            if len(categorical_fields) > 3:
                st.warning("Only the first 3 categorical fields will be processed.")
                categorical_fields = categorical_fields[:3]

            for i, col in enumerate(categorical_fields):
                freq_data = df[col].fillna("(Missing)").value_counts(dropna=False).head(15).reset_index()
                if df[col].nunique(dropna=False) > 15:
                    st.info(f"Only the top 15 most frequent values are shown for '{col}'.")
                freq_data.columns = [col, "Count"]
                freq_data["Percentage"] = (freq_data["Count"] / freq_data["Count"].sum() * 100).round(2)

                left_col, right_col = st.columns([1, 1])
                with left_col:
                    st.markdown(f"### Frequency Table: {col}")
                    st.dataframe(freq_data, use_container_width=True)

                with right_col:
                    st.markdown(f"### Frequency Plot: {col}")
                    fig, ax = plt.subplots()
                    ax.barh(freq_data[col].astype(str), freq_data["Count"], color=plt.cm.tab10(i % 10))
                    ax.set_xlabel("Count")
                    ax.set_ylabel("Value")
                    ax.invert_yaxis()
                    ax.set_title(f"Top Values in '{col}'")
                    st.pyplot(fig)

    elif analysis_type == "Line Chart":
        st.markdown("## Line Charts")
        # Only render chart when Run Analysis is clicked and valid x_field is selected
        if not valid_x_fields:
            st.warning("Line charts require a time-based x-axis. No datetime or t-index fields were found.")
        elif run_analysis:
            y_fields = [f for f in selected_features if inferred_types.get(f) == "numeric"]
            if not y_fields:
                st.warning("No numeric features selected for the Y-axis.")
            else:
                if len(y_fields) > 6:
                    st.warning("More than 6 numeric features selected. Only the first 6 will be shown.")
                    y_fields = y_fields[:6]
                for i, y_field in enumerate(y_fields):
                    # Defensive: skip if not in df
                    if x_field not in df.columns or y_field not in df.columns:
                        continue
                    chart_df = df[[x_field, y_field]].dropna()
                    if chart_df.empty:
                        st.warning(f"Skipping '{y_field}' due to missing values.")
                        continue
                    chart_df = chart_df.sort_values(by=x_field)
                    if i % 2 == 0:
                        col_left, col_right = st.columns([1, 1])
                    target_col = col_left if i % 2 == 0 else col_right
                    with target_col:
                        fig, ax = plt.subplots()
                        ax.plot(chart_df[x_field], chart_df[y_field], marker='o', linewidth=1.5)
                        ax.set_title(f"{y_field} over {x_field}")
                        ax.set_xlabel(x_field)
                        ax.set_ylabel(y_field)
                        ax.grid(True, linestyle="--", alpha=0.5)
                        # Improved x-axis date formatting
                        # Only format as datetime if it's a real datetime field and not a t-index
                        if np.issubdtype(chart_df[x_field].dtype, np.datetime64) and not x_field.startswith("t - "):
                            num_days = (chart_df[x_field].max() - chart_df[x_field].min()).days
                            if num_days > 1500:
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                            elif num_days > 90:
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                            else:
                                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
                            fig.autofmt_xdate()
                        st.pyplot(fig)

    elif analysis_type == "Box Plot":
        st.markdown("### Box Plot")

        # Use selected_features from session state, ensure they are valid and numeric
        numeric_selected = [f for f in selected_features if pd.api.types.is_numeric_dtype(df[f])]
        if not selected_features:
            st.warning("Please select at least one feature for the box plot.")
        elif not numeric_selected:
            st.warning("No valid numeric features selected for the box plot.")
        else:
            if len(numeric_selected) > 6:
                st.warning("More than 6 features selected. Only the first 6 will be shown.")
                numeric_selected = numeric_selected[:6]

            cleaned_df = df[numeric_selected + ([group_col] if group_col else [])].dropna()
            if len(cleaned_df) > 25000:
                cleaned_df = cleaned_df.sample(n=10000, random_state=42)

            import seaborn as sns
            for i, col in enumerate(numeric_selected):
                fig, ax = plt.subplots(figsize=(8, 4))
                df_box = cleaned_df
                sns.boxplot(
                    data=df_box,
                    x=group_col if group_col else None,
                    y=col,
                    color='skyblue',
                    showmeans=True,
                    meanprops={
                        "marker": "o",
                        "markerfacecolor": "white",
                        "markeredgecolor": "black"
                    },
                    showfliers=show_outliers,
                    ax=ax,
                    orient="h" if not group_col else "v"
                )
                if group_col:
                    ax.set_title(f'{col} by {group_col}')
                    ax.set_xlabel(group_col)
                    ax.set_ylabel(col)
                else:
                    ax.set_title(f'Box Plot: {col}')
                    ax.set_xlabel(col)
                ax.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig)

            st.markdown("#### Summary Statistics")
            st.dataframe(cleaned_df[numeric_selected].describe().T)

# If not file_loaded, clear info and df
elif not st.session_state.get("file_loaded", False):
    # On reset or file removal, ensure sampled_data_info and cleanliness messages disappear
    st.session_state.pop("sampled_data_info", None)
    st.session_state.pop("file_info", None)
    st.session_state.pop("data_cleanliness_msgs", None)
    st.session_state["df"] = None
    st.session_state.pop("filename", None)

# (Removed redundant logic that sets selected_features from session state again to avoid mutation error)
