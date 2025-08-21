import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
import os

#FileHub imports for transfer functionality
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
for key in [
    "df", "filename", "selected_features", "analysis_type", "analysis_results",
    "feature_selector", "feature_selector_options", "analysis_type_selectbox", "dep_var_select", "x_axis_select",
    "cluster_dropdown", "boxplot_grouping", "boxplot_outliers",
    "date_filter_mode", "date_filter_range", "t_filter_range",
    "enable_group_line", "group_by_col", "group_category_values",
    "line_agg_method", "line_category_mode", "line_selected_category",
    "line_selected_categories"
]:
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
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.dates as mdates
import altair as alt
import plotly.graph_objects as go
import colorsys


with st.sidebar:
    with st.expander("**More Tools**", expanded=True):
        st.markdown("""
- [**DataBlender**](https://datablendertool.streamlit.app/)
- [**DataSampler**](https://datasamplertool.streamlit.app/)
""")
    # --------- Transfer Token Section (FileHub) ---------
    transfer_token = st.text_input("Enter transfer token to load file from FileHub", key="transfer_token_input")
    if st.button("Submit Token"):
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
                    st.error(f"Error retrieving file: {e}")
        else:
            st.warning("Please enter a token before submitting.")

# ---------- Main Layout ----------
st.title("DataWizard")
st.subheader("Upload your data to begin analysis")
st.write(
    "This tool lets you upload a CSV or Excel file and quickly view summary statistics, charts, distributions, run regressions, cluster your data, and more. "
    "You can upload your file using the button below or import data by entering a FileHub token."
)

main_file = st.file_uploader("Upload your file here", type=["csv", "xls", "xlsx"], key="main")

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
                # Clear all prior file-related session state to allow full overwrite
                st.session_state.pop("df", None)
                st.session_state.pop("filename", None)
                st.session_state.pop("sampled_data_info", None)
                st.session_state.pop("file_info", None)
                st.session_state.pop("data_cleanliness_msgs", None)
                st.session_state["upload_processed"] = False
                st.session_state["file_from_filehub"] = False
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
        # --- Analysis-first workflow ---
        # Ensure local variable exists for downstream references
        selected_features = st.session_state.get("selected_features", [])

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
                "Clustering",
                "Sankey"
            ],
            index=0,
            key="analysis_type_selectbox"
        )

        # Show Feature Selection ONLY for analyses that need it
        analyses_require_features = {
            "Summary statistics",
            "Frequency Table",
            "Histogram",
            "Box Plot",
            "Line Chart",
            "Correlation matrix",
            "Linear Regression",
            "Clustering",
        }
        if analysis_type in analyses_require_features:
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
                    key="selected_features"
                )
                st.session_state["trigger_feature_update"] = False
        # Sankey configuration (multi-level, essential inputs)
        if analysis_type == "Sankey":
            # Candidate level columns: treat categorical/object columns as eligible levels
            sankey_level_candidates = [c for c, t in inferred_types.items() if t == "categorical" and c in df.columns]
            if not sankey_level_candidates:
                st.warning("No categorical columns available to use as Sankey levels.")
            # Preserve selection order in multiselect (user's click order defines left→right)
            st.session_state.setdefault("sankey_levels", [])
            sankey_levels = st.multiselect(
                "Select 2–5 level columns (left → right order)",
                options=sankey_level_candidates,
                key="sankey_levels",
                help="Choose categorical columns. Order of selection becomes the flow order."
            )
            # Numeric value column
            sankey_value_candidates = [c for c, t in inferred_types.items() if t == "numeric" and c in df.columns]
            st.session_state.setdefault("sankey_value_col", sankey_value_candidates[0] if sankey_value_candidates else None)
            sankey_value_col = st.selectbox(
                "Select value column (non-negative numeric)",
                options=sankey_value_candidates if sankey_value_candidates else [""],
                key="sankey_value_col"
            )
            # Minimal, intuitive pruning controls
            st.session_state.setdefault("sankey_topn", 12)
            st.session_state.setdefault("sankey_group_other", True)
            st.session_state.setdefault("sankey_other_pct", 1.0)
            st.markdown("**Display options**")
            sankey_topn = st.slider("Top-N nodes per level", min_value=3, max_value=25, value=st.session_state["sankey_topn"], key="sankey_topn")
            sankey_group_other = st.checkbox("Group small items into ‘Other’", value=st.session_state["sankey_group_other"], key="sankey_group_other")
            sankey_other_pct = st.slider("Threshold for ‘Other’ (percent of total)", min_value=0.0, max_value=5.0, value=float(st.session_state["sankey_other_pct"]), step=0.1, key="sankey_other_pct")
            # --- Quick Slicer (single level filter) ---
            st.markdown("**Filter (optional)**")
            level_choices = st.session_state.get("sankey_levels", [])
            if level_choices:
                # Remember previously selected level to reset invalid defaults when level changes
                prev_level = st.session_state.get("sankey_last_slicer_level", None)
                st.session_state.setdefault("sankey_slicer_level", level_choices[0])
                slicer_level = st.selectbox(
                    "Focus on values of a level",
                    options=level_choices,
                    key="sankey_slicer_level",
                    help="Pick one level to filter on."
                )

                # If the level changed, clear the previous selections to avoid invalid defaults
                if prev_level != slicer_level:
                    st.session_state["sankey_slicer_values"] = []
                    st.session_state["sankey_last_slicer_level"] = slicer_level

                # Build available options for the selected level
                try:
                    level_uniqs = sorted(df[slicer_level].dropna().astype(str).unique().tolist())[:1000]
                except Exception:
                    level_uniqs = []

                # Sanitize session state so it only contains valid options
                st.session_state.setdefault("sankey_slicer_values", [])
                current_defaults = st.session_state.get("sankey_slicer_values", [])
                option_set = set(level_uniqs)
                # Ensure session state contains only valid options
                st.session_state["sankey_slicer_values"] = [v for v in current_defaults if str(v) in option_set]

                slicer_values = st.multiselect(
                    f"Show only rows where **{slicer_level}** is:",
                    options=level_uniqs,
                    key="sankey_slicer_values"
                )
            else:
                slicer_level, slicer_values = None, []

            # --- Color controls for Sankey ---
            st.markdown("**Coloring**")
            st.session_state.setdefault("sankey_color_by", "Source")
            st.session_state.setdefault("sankey_link_opacity", 0.35)
            color_by = st.radio(
                "Color links by",
                options=["Source", "Target"],
                horizontal=True,
                key="sankey_color_by",
                help="Use source or target node color for each link."
            )
            link_opacity = st.slider(
                "Link opacity",
                min_value=0.1,
                max_value=1.0,
                value=float(st.session_state["sankey_link_opacity"]),
                step=0.05,
                key="sankey_link_opacity"
            )
        # Dependent variable for regression (positioned directly under analysis type)
        dependent_var = None
        if analysis_type == "Linear Regression" and selected_features:
            # default to the first selected feature if none chosen yet
            if "dep_var_select" not in st.session_state and selected_features:
                st.session_state["dep_var_select"] = selected_features[0]
            dependent_var = st.selectbox(
                "Select dependent variable",
                options=selected_features,
                key="dep_var_select"
            )
        # Time-based X-axis selection for Line Chart (positioned under analysis type)
        x_field = None
        valid_x_fields = []
        if analysis_type == "Line Chart":
            valid_x_fields = [col for col in df.columns if inferred_types.get(col) == "datetime" or col.startswith("t - ")]
            if not valid_x_fields:
                st.warning("Line charts require a time-based x-axis. No datetime or t-index fields were found.")
            else:
                # default to first valid if none chosen yet
                if "x_axis_select" not in st.session_state and valid_x_fields:
                    st.session_state["x_axis_select"] = valid_x_fields[0]
                x_field = st.selectbox(
                    "Select a time-based field for the X-axis",
                    options=valid_x_fields,
                    key="x_axis_select",
                    index=0
                )
        # Date / t-index filter controls for Line Chart (shown beneath X-axis selector)
        if analysis_type == "Line Chart":
            if x_field and x_field in df.columns:
                # DATETIME FILTER MODE
                if inferred_types.get(x_field) == "datetime" and not x_field.startswith("t - "):
                    # Compute bounds
                    x_nonnull = df[x_field].dropna()
                    if not x_nonnull.empty:
                        xmin = x_nonnull.min().date()
                        xmax = x_nonnull.max().date()
                        # Presets + Custom
                        preset_options = [
                            "All", "YTD", "Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"
                        ]
                        mode = st.selectbox(
                            "Date range (optional)",
                            options=preset_options,
                            index=0,
                            key="date_filter_mode"
                        )
                        # Persist custom range
                        if "date_filter_range" not in st.session_state or not st.session_state["date_filter_range"]:
                            st.session_state["date_filter_range"] = (xmin, xmax)
                        if mode == "Custom":
                            # Always use (xmin, xmax) as the default value for the date_input
                            dr = st.date_input(
                                "Select custom date range",
                                value=(xmin, xmax),
                                min_value=xmin,
                                max_value=xmax,
                                key="custom_date_range"
                            )
                            # After assignment, ensure session_state["date_filter_range"] defaults to (xmin, xmax) if not set
                            if isinstance(dr, tuple) and len(dr) == 2:
                                st.session_state["date_filter_range"] = (dr[0], dr[1])
                            if (
                                "date_filter_range" not in st.session_state
                                or not st.session_state["date_filter_range"]
                                or not isinstance(st.session_state["date_filter_range"], tuple)
                                or len(st.session_state["date_filter_range"]) != 2
                            ):
                                st.session_state["date_filter_range"] = (xmin, xmax)
                        else:
                            # Ensure range is synced to preset on change
                            st.session_state["date_filter_range"] = (xmin, xmax)
                # T-INDEX FILTER MODE (numeric t-index)
                elif x_field.startswith("t - ") or inferred_types.get(x_field) == "numeric":
                    x_nonnull = df[x_field].dropna()
                    if not x_nonnull.empty:
                        tmin = int(np.nanmin(x_nonnull))
                        tmax = int(np.nanmax(x_nonnull))
                        if "t_filter_range" not in st.session_state or not st.session_state["t_filter_range"]:
                            st.session_state["t_filter_range"] = (tmin, tmax)
                        st.session_state["t_filter_range"] = st.slider(
                            "Index range",
                            min_value=tmin,
                            max_value=tmax,
                            value=st.session_state.get("t_filter_range", (tmin, tmax))
                        )
            # Guided workflow: offer to split lines by category if we detect duplicates per x value
            group_by_col = None
            enable_group = False
            if analysis_type == "Line Chart" and x_field and x_field in df.columns:
                # Detect duplicates on x_field (multiple rows per time value)
                try:
                    has_dupes_on_x = df[x_field].duplicated().any()
                except Exception:
                    has_dupes_on_x = False

                # Candidate categorical columns
                categorical_candidates = [c for c, t in inferred_types.items() if t == "categorical" and c in df.columns]

                if has_dupes_on_x and categorical_candidates:
                    enable_group = st.checkbox(
                        "Group by categorical variable (e.g., Country)",
                        key="enable_group_line"
                    )
                    if enable_group:
                        # Group-by selector appears immediately
                        group_by_col = st.selectbox(
                            "Group by",
                            options=categorical_candidates,
                            index=0 if st.session_state.get("group_by_col") is None or st.session_state.get("group_by_col") not in categorical_candidates else categorical_candidates.index(st.session_state.get("group_by_col")),
                            key="group_by_col"
                        )

                        # Aggregation method toggle (shows immediately on enable)
                        agg_method = st.radio(
                            "Aggregation method",
                            options=["Mean", "Sum"],
                            horizontal=True,
                            key="line_agg_method"
                        )

                        # Category mode (shows immediately on enable)
                        cat_mode = st.radio(
                            "Category mode",
                            options=["All categories (aggregate)", "Pick a category"],
                            key="line_category_mode"
                        )

                        # Category picker (multi-select up to 5) when a group-by column is chosen and mode requires it
                        if cat_mode == "Pick a category" and group_by_col:
                            uniq_vals = df[group_by_col].dropna().astype(str).unique().tolist()
                            uniq_vals = sorted(uniq_vals)[:1000]
                            # Migrate legacy single selection to list if present
                            legacy_single = st.session_state.get("line_selected_category", None)
                            if legacy_single is not None and not st.session_state.get("line_selected_categories"):
                                st.session_state["line_selected_categories"] = [str(legacy_single)] if str(legacy_single) in uniq_vals else []
                            # Default selection: previously chosen list or first up to 3
                            default_list = st.session_state.get("line_selected_categories")
                            if not isinstance(default_list, list) or not default_list:
                                default_list = uniq_vals[:3]
                            selected_categories = st.multiselect(
                                f"Select up to 5 {group_by_col} values",
                                options=uniq_vals,
                                default=default_list,
                                key="line_selected_categories"
                            )
                            if isinstance(selected_categories, list) and len(selected_categories) > 5:
                                st.warning("Limiting to the first 5 selected categories.")
                                st.session_state["line_selected_categories"] = selected_categories[:5]
                    else:
                        # Clear grouping selections when the feature is turned off
                        st.session_state["group_by_col"] = None
                        st.session_state["group_category_values"] = []
                        st.session_state["line_category_mode"] = None
                        st.session_state["line_selected_category"] = None
                        st.session_state["line_agg_method"] = None
                        st.session_state["line_selected_categories"] = []
                        st.session_state["line_selected_category"] = None
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

        # "Run Analysis" button now appears after axis selection / dependent var
        run_analysis = st.button("Run Analysis")
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
                value_counts = col_data.value_counts()
                if len(value_counts) == 1:
                    most_common = ""
                    least_common = ""
                else:
                    top_counts = value_counts[value_counts == value_counts.max()]
                    bottom_counts = value_counts[value_counts == value_counts.min()]
                    most_common = top_counts.index[0] if len(top_counts) == 1 else ""
                    least_common = bottom_counts.index[0] if len(bottom_counts) == 1 else ""

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

        # Define a color palette for dynamic feature coloring
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#FFA600']

        for i, col in enumerate(numeric_features):
            feature_color = palette[i % len(palette)]
            if i % 2 == 0:
                col_left, col_right = st.columns([1, 1])
            target_col = col_left if i % 2 == 0 else col_right

            with target_col:
                col_series = pd.to_numeric(df[col], errors='coerce').dropna()
                if col_series.empty:
                    st.warning(f"No data available for '{col}'. Skipping.")
                else:
                    vals = col_series.values
                    n = len(vals)
                    try:
                        # Central 95%: 2.5th to 97.5th percentiles
                        p_low = np.quantile(vals, 0.025)
                        p_high = np.quantile(vals, 0.975)
                    except Exception:
                        p_low, p_high = np.min(vals), np.max(vals)

                    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
                        # Fallback to automatic binning if degenerate
                        chart_df = pd.DataFrame({col: col_series})
                        chart = (
                            alt.Chart(chart_df)
                            .mark_bar()
                            .encode(
                                x=alt.X(col, bin=alt.Bin(maxbins=30), title="Value"),
                                y=alt.Y('count()', title="Frequency", scale=alt.Scale(zero=True)),
                                tooltip=[alt.Tooltip('count()', title='Count')]
                            )
                            .properties(title=f"Distribution of {col}", width='container')
                        )
                        st.altair_chart(chart, use_container_width=True)
                    else:
                        # Build 8 equal-width interior bins + 2 catch-all bins
                        interior_bins = 8
                        width = (p_high - p_low) / interior_bins if p_high > p_low else 0
                        if width <= 0:
                            chart_df = pd.DataFrame({col: col_series})
                            chart = (
                                alt.Chart(chart_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X(col, bin=alt.Bin(maxbins=30), title="Value"),
                                    y=alt.Y('count()', title="Frequency", scale=alt.Scale(zero=True)),
                                    tooltip=[alt.Tooltip('count()', title='Count')]
                                )
                                .properties(title=f"Distribution of {col}", width='container')
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            # Explicit interior edges across central 95%
                            edges = np.linspace(p_low, p_high, interior_bins + 1)
                            interior_vals = vals[(vals >= p_low) & (vals <= p_high)]
                            counts, _ = np.histogram(interior_vals, bins=edges)
                            below = int((vals < p_low).sum())
                            above = int((vals > p_high).sum())

                            # Assemble bins df (avoid +/-inf; use one width outside range)
                            rows = []
                            # Left catch-all
                            rows.append({
                                'bin_start': float(edges[0] - width),
                                'bin_end': float(edges[0]),
                                'count': below,
                                'is_tail': True
                            })
                            # Interior bins
                            for i_b in range(interior_bins):
                                b_start = float(edges[i_b])
                                b_end = float(edges[i_b + 1])
                                rows.append({
                                    'bin_start': b_start,
                                    'bin_end': b_end,
                                    'count': int(counts[i_b]),
                                    'is_tail': False
                                })
                            # Right catch-all
                            rows.append({
                                'bin_start': float(edges[-1]),
                                'bin_end': float(edges[-1] + width),
                                'count': above,
                                'is_tail': True
                            })

                            bins_df = pd.DataFrame(rows)
                            if n > 0:
                                bins_df['pct'] = bins_df['count'] / n
                            else:
                                bins_df['pct'] = 0.0

                            # Ensure non-negative counts and compute a stable y-axis max
                            bins_df['count'] = bins_df['count'].clip(lower=0)
                            max_count = max(1, int(bins_df['count'].max()))

                            chart = (
                                alt.Chart(bins_df)
                                .mark_bar(orient='vertical')
                                .encode(
                                    x=alt.X('bin_start:Q', title='Value'),
                                    x2='bin_end:Q',
                                    y=alt.Y('count:Q', title='Frequency', scale=alt.Scale(domain=[0, max_count * 1.05])),
                                    y2=alt.value(0),
                                    color=alt.condition('datum.is_tail', alt.value('#A0A0A0'), alt.value(feature_color)),
                                    tooltip=[
                                        alt.Tooltip('bin_start:Q', title='Bin start', format=',.3f'),
                                        alt.Tooltip('bin_end:Q', title='Bin end', format=',.3f'),
                                        alt.Tooltip('count:Q', title='Count', format=','),
                                        alt.Tooltip('pct:Q', title='% of total', format='.2%')
                                    ]
                                )
                                .properties(title=f"Distribution of {col}", width='container')
                            )
                            st.altair_chart(chart, use_container_width=True)

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

        if dependent_var not in selected_features:
            st.error("Dependent variable must be one of the selected fields.")
        elif inferred_types.get(dependent_var) != "numeric":
            st.error("Dependent variable must be numeric for linear regression.")
        else:
            # Filter independent vars to numeric only; warn about skipped non-numeric
            numeric_indep = [f for f in selected_features if f != dependent_var and inferred_types.get(f) == "numeric"]
            skipped_indep = [f for f in selected_features if f != dependent_var and f not in numeric_indep]
            if skipped_indep:
                st.warning("Skipping non-numeric independent variables: " + ", ".join(skipped_indep))

            if len(numeric_indep) < 1:
                st.warning("Please select at least one numeric independent variable.")
            elif len(numeric_indep) > 8:
                st.error("You can include up to 8 independent variables in the regression.")
            else:
                cols_used = [dependent_var] + numeric_indep
                data = df[cols_used].dropna()

                warn_fields = []
                for col in cols_used:
                    ratio = df[col].isna().mean()
                    if ratio > 0.2:
                        warn_fields.append((col, ratio))
                if warn_fields:
                    st.warning("Some fields used in this regression have high missing values:\n" +
                               "\n".join([f"{col}: {ratio:.0%} missing" for col, ratio in warn_fields]))

                X = sm.add_constant(data[numeric_indep])
                y = data[dependent_var]
                model = sm.OLS(y, X).fit()

                eqn_parts = [f"{model.params[name]:.4f} × {name}" for name in numeric_indep]
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
                if len(numeric_indep) > 1:
                    st.markdown(f"**Adjusted R²:** {model.rsquared_adj:.4f}")

                if len(numeric_indep) == 1:
                    st.markdown("### Regression Plot")
                    xname = numeric_indep[0]
                    x_vals = data[xname]
                    y_vals = y
                    y_pred = model.predict(X)

                    reg_df = pd.DataFrame({
                        xname: pd.to_numeric(x_vals, errors='coerce'),
                        dependent_var: pd.to_numeric(y_vals, errors='coerce'),
                        'Predicted': pd.to_numeric(y_pred, errors='coerce'),
                    }).dropna().sort_values(by=xname)

                    scatter = (
                        alt.Chart(reg_df)
                        .mark_circle(opacity=0.6, color="steelblue")
                        .encode(
                            x=alt.X(f"{xname}:Q", title=xname),
                            y=alt.Y(f"{dependent_var}:Q", title=dependent_var),
                            tooltip=[
                                alt.Tooltip(f"{xname}:Q", title=xname, format=",.3f"),
                                alt.Tooltip(f"{dependent_var}:Q", title=dependent_var, format=",.3f"),
                                alt.Tooltip("Predicted:Q", title="Predicted", format=",.3f"),
                            ],
                        )
                    )

                    line = (
                        alt.Chart(reg_df)
                        .mark_line(color="red")
                        .encode(
                            x=alt.X(f"{xname}:Q", title=xname),
                            y=alt.Y("Predicted:Q", title=dependent_var),
                        )
                    )

                    chart = (scatter + line).properties(
                        title=f"{dependent_var} vs {xname} (Regression)",
                        width='container',
                        height=600
                    )
                    st.altair_chart(chart, use_container_width=True)

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
                # Altair interactive scatter: one point per row, colored by cluster
                chart = (
                    alt.Chart(pca_df)
                    .mark_circle(opacity=0.7)
                    .encode(
                        x=alt.X('PC1:Q', title='Principal Component 1'),
                        y=alt.Y('PC2:Q', title='Principal Component 2'),
                        color=alt.Color(
                            'Cluster Name:N',
                            legend=alt.Legend(orient="right", direction="vertical", columns=1)
                        ),
                        tooltip=[
                            alt.Tooltip('Cluster Name:N', title='Cluster'),
                            alt.Tooltip('PC1:Q', title='PC1', format=',.3f'),
                            alt.Tooltip('PC2:Q', title='PC2', format=',.3f')
                        ]
                    )
                    .properties(
                        title='K-Means Clustering Results (PCA)',
                        width='container',
                        height=800
                    )
                )
                st.altair_chart(chart, use_container_width=True)

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
                    # Altair horizontal bar chart, sorted by count descending
                    chart = (
                        alt.Chart(freq_data)
                        .mark_bar()
                        .encode(
                            y=alt.Y(f"{col}:N", sort='-x', title="Value"),
                            x=alt.X("Count:Q", title="Count", scale=alt.Scale(zero=True)),
                            tooltip=[
                                alt.Tooltip(f"{col}:N", title="Value"),
                                alt.Tooltip("Count:Q", title="Count", format=","),
                                alt.Tooltip("Percentage:Q", title="% of total", format=".2f")
                            ]
                        )
                        .properties(title=f"Top Values in '{col}'", width='container')
                    )
                    st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Line Chart":
        st.markdown("## Line Charts")
        # Only render chart when Run Analysis is clicked and valid x_field is selected
        if not valid_x_fields:
            st.warning("Line charts require a time-based x-axis. No datetime or t-index fields were found.")
        elif run_analysis:
            # Build filtered dataframe according to selected filter controls
            df_filt = df.copy()
            if x_field and x_field in df_filt.columns:
                if inferred_types.get(x_field) == "datetime" and not x_field.startswith("t - "):
                    x_nonnull = df_filt[x_field].dropna()
                    if not x_nonnull.empty:
                        xmin = x_nonnull.min().date()
                        xmax = x_nonnull.max().date()
                        mode = st.session_state.get("date_filter_mode", "All")
                        if mode == "All":
                            start_dt, end_dt = xmin, xmax
                        elif mode == "YTD":
                            end_dt = xmax
                            start_dt = pd.Timestamp(year=end_dt.year, month=1, day=1).date()
                        elif mode == "Last 7 days":
                            end_dt = xmax
                            start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=7)).date()
                        elif mode == "Last 30 days":
                            end_dt = xmax
                            start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=30)).date()
                        elif mode == "Last 90 days":
                            end_dt = xmax
                            start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=90)).date()
                        elif mode == "Last 365 days":
                            end_dt = xmax
                            start_dt = (pd.Timestamp(end_dt) - pd.Timedelta(days=365)).date()
                        elif mode == "Custom":
                            start_dt, end_dt = st.session_state.get("date_filter_range", (xmin, xmax))
                        else:
                            start_dt, end_dt = xmin, xmax
                        mask = df_filt[x_field].dt.date.between(start_dt, end_dt)
                        df_filt = df_filt.loc[mask]
                else:
                    # t-index / numeric x_field
                    tmin, tmax = st.session_state.get("t_filter_range", (None, None))
                    if tmin is not None and tmax is not None:
                        mask = df_filt[x_field].between(tmin, tmax)
                        df_filt = df_filt.loc[mask]

            if df_filt.empty:
                st.warning("No data in the selected range.")
            else:
                y_fields = [f for f in selected_features if inferred_types.get(f) == "numeric"]
                if not y_fields:
                    st.warning("No numeric features selected for the Y-axis.")
                else:
                    # Check if the guided grouping is enabled and valid
                    enable_group = st.session_state.get("enable_group_line", False)
                    group_by_col = st.session_state.get("group_by_col", None)
                    # selected_categories = st.session_state.get("group_category_values", []) # No longer used

                    if enable_group and group_by_col and group_by_col in df_filt.columns:
                        # Determine aggregation function
                        agg_method = st.session_state.get("line_agg_method", "Mean")
                        agg_fn = "sum" if agg_method == "Sum" else "mean"
                        cat_mode = st.session_state.get("line_category_mode", "All categories (aggregate)")
                        # selected_category = st.session_state.get("line_selected_category", None)
                        selected_categories = st.session_state.get("line_selected_categories", [])

                        if len(y_fields) > 6:
                            st.warning("More than 6 numeric features selected. Only the first 6 will be shown.")
                            y_fields = y_fields[:6]

                        for i, y_field in enumerate(y_fields):
                            # Prepare data per measure
                            work_df = df_filt[[x_field, group_by_col, y_field]].dropna()
                            if work_df.empty:
                                st.warning(f"No data for '{y_field}' after filtering.")
                                continue

                            if cat_mode == "Pick a category" and selected_categories:
                                # Filter to chosen categories; aggregate duplicates on the same x per category
                                sub = work_df[work_df[group_by_col].astype(str).isin([str(v) for v in selected_categories])]
                                if sub.empty:
                                    st.warning("No rows for the selected categories in '{y_field}'.")
                                    continue
                                agg_df = sub.groupby([x_field, group_by_col], as_index=False)[y_field].agg(agg_fn)
                                title_suffix = f"for selected {group_by_col} values"
                            else:
                                # Aggregate across all categories per x
                                agg_df = work_df.groupby([x_field], as_index=False)[y_field].agg(agg_fn)
                                title_suffix = "(All categories)"

                            agg_df = agg_df.sort_values(by=x_field)

                            # Layout two charts per row like before
                            if i % 2 == 0:
                                col_left, col_right = st.columns([1, 1])
                            target_col = col_left if i % 2 == 0 else col_right
                            with target_col:
                                # Determine Altair types
                                x_type = 'temporal' if (np.issubdtype(agg_df[x_field].dtype, np.datetime64) and not x_field.startswith("t - ")) else 'quantitative'
                                base = alt.Chart(agg_df).mark_line(point=True).encode(
                                    x=alt.X(x_field, type=x_type, title=x_field),
                                    y=alt.Y(y_field, type='quantitative', title=y_field)
                                ).properties(
                                    title=f"{y_field} over {x_field} {title_suffix}",
                                    width='container'
                                )
                                if (cat_mode == "Pick a category" and selected_categories) and (group_by_col in agg_df.columns):
                                    chart = base.encode(
                                        color=alt.Color(
                                            group_by_col,
                                            legend=alt.Legend(
                                                orient="right",
                                                direction="vertical",
                                                columns=1
                                            )
                                        )
                                    )
                                else:
                                    chart = base.encode(color=alt.value('steelblue'))
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        # Original behavior: multiple numeric series in columns plotted against x_field
                        if len(y_fields) > 6:
                            st.warning("More than 6 numeric features selected. Only the first 6 will be shown.")
                            y_fields = y_fields[:6]
                        for i, y_field in enumerate(y_fields):
                            if x_field not in df_filt.columns or y_field not in df_filt.columns:
                                continue
                            chart_df = df_filt[[x_field, y_field]].dropna()
                            if chart_df.empty:
                                st.warning(f"Skipping '{y_field}' due to missing values.")
                                continue
                            chart_df = chart_df.sort_values(by=x_field)
                            if i % 2 == 0:
                                col_left, col_right = st.columns([1, 1])
                            target_col = col_left if i % 2 == 0 else col_right
                            with target_col:
                                x_type = 'temporal' if (np.issubdtype(chart_df[x_field].dtype, np.datetime64) and not x_field.startswith("t - ")) else 'quantitative'
                                chart = alt.Chart(chart_df).mark_line(point=True).encode(
                                    x=alt.X(x_field, type=x_type, title=x_field),
                                    y=alt.Y(y_field, type='quantitative', title=y_field),
                                    color=alt.value('steelblue')
                                ).properties(
                                    title=f"{y_field} over {x_field}",
                                    width='container'
                                )
                                st.altair_chart(chart, use_container_width=True)

    elif analysis_type == "Sankey":
        st.markdown("## Sankey Flow")

        # Fetch UI state
        levels = st.session_state.get("sankey_levels", [])
        value_col = st.session_state.get("sankey_value_col", None)
        topn = int(st.session_state.get("sankey_topn", 12) or 12)
        group_other = bool(st.session_state.get("sankey_group_other", True))
        other_pct = float(st.session_state.get("sankey_other_pct", 1.0) or 1.0)

        # Validations
        if len(levels) < 2:
            st.error("Please select at least two level columns.")
        elif value_col is None or value_col == "":
            st.error("Please select a numeric value column.")
        else:
            try:
                work = df.copy()
                # Coerce numeric values and drop invalids
                work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
                work = work.dropna(subset=levels + [value_col])
                # Disallow negatives in multi-level mode for MVP
                work = work[work[value_col] >= 0]
                # Apply Quick Slicer filter if provided
                slicer_level = st.session_state.get("sankey_slicer_level")
                slicer_values = st.session_state.get("sankey_slicer_values", [])
                if slicer_level and slicer_values:
                    work = work[work[slicer_level].astype(str).isin([str(v) for v in slicer_values])]

                if work.empty:
                    st.warning("No data available after cleaning (missing or negative values removed).")
                else:
                    total_value = work[value_col].sum()
                    # Compute throughput per level for Top-N
                    level_maps = {}
                    for L in levels:
                        s = work.groupby(L, dropna=False)[value_col].sum().sort_values(ascending=False)
                        if topn is not None and topn > 0:
                            keep = set(s.iloc[:topn].index)
                        else:
                            keep = set(s.index)
                        # Apply ‘Other’ by percentage threshold on total flow
                        if group_other and other_pct > 0:
                            thresh = (other_pct / 100.0) * total_value
                            keep = keep.union(set(s[s >= thresh].index))
                        # Build mapping: raw label -> kept label or 'Other'
                        mapping = {label: (label if label in keep else "Other") for label in s.index}
                        level_maps[L] = mapping

                    # Apply mappings to a pruned copy
                    pruned = work.copy()
                    for L in levels:
                        pruned[L] = pruned[L].map(level_maps[L]).fillna("Other")

                    # Aggregate links between adjacent levels
                    link_frames = []
                    for i in range(len(levels) - 1):
                        a, b = levels[i], levels[i + 1]
                        g = (
                            pruned.groupby([a, b], as_index=False)[value_col]
                            .sum()
                            .rename(columns={a: "source", b: "target", value_col: "value"})
                            .assign(level_from=i, level_to=i + 1)
                        )
                        g = g[g["value"] > 0]
                        link_frames.append(g)

                    if not link_frames:
                        st.warning("No links to display after aggregation.")
                    else:
                        links_all = pd.concat(link_frames, ignore_index=True)

                        # Build nodes with per-level namespace
                        nodes_df = (
                            pd.concat([
                                links_all[["level_from", "source"]].rename(columns={"level_from": "level", "source": "label"}),
                                links_all[["level_to", "target"]].rename(columns={"level_to": "level", "target": "label"}),
                            ])
                            .drop_duplicates()
                            .sort_values(["level", "label"]).reset_index(drop=True)
                        )
                        nodes_df["node_index"] = range(len(nodes_df))

                        # --- Build per-node colors (reuse hues within a level, vary lightness) ---
                        # Create ordered lists of labels per level
                        level_to_labels = {}
                        for lvl in sorted(nodes_df["level"].unique()):
                            labels_lvl = nodes_df.loc[nodes_df["level"] == lvl, "label"].astype(str).tolist()
                            level_to_labels[int(lvl)] = labels_lvl

                        def rgb_to_hex(r, g, b):
                            return "#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255))

                        node_color_map = {}
                        for lvl, labels_lvl in level_to_labels.items():
                            n = max(1, len(labels_lvl))
                            # Base palette count per cycle (hues reused beyond this)
                            cycle = 10
                            for i, label in enumerate(labels_lvl):
                                # Hue cycles every `cycle`; lightness steps every cycle to create variation
                                hue_index = i % cycle
                                cycle_index = i // cycle
                                h = (hue_index / cycle) % 1.0
                                # Saturation constant; lightness varies by level and cycle_index
                                s = 0.60
                                # Start lightness around 0.55, adjust slightly by level and cycle
                                base_l = 0.55 + 0.04 * (lvl % 3)
                                l = max(0.30, min(0.80, base_l - 0.06 * cycle_index))
                                r, g, b = colorsys.hls_to_rgb(h, l, s)
                                node_color_map[(lvl, label)] = rgb_to_hex(r, g, b)
                            # Ensure 'Other' (if present) is neutral gray
                            if "Other" in labels_lvl:
                                node_color_map[(lvl, "Other")] = "#B0B0B0"

                        # Construct node color array in nodes_df order
                        node_colors = [node_color_map[(int(r.level), str(r.label))] for r in nodes_df.itertuples(index=False)]

                        # Maps to indices
                        index_map = {(r.level, r.label): int(r.node_index) for r in nodes_df.itertuples(index=False)}
                        links_all["source_idx"] = links_all.apply(lambda r: index_map[(r.level_from, r.source)], axis=1)
                        links_all["target_idx"] = links_all.apply(lambda r: index_map[(r.level_to, r.target)], axis=1)

                        # --- Build per-link colors from chosen node side ---
                        color_by_setting = st.session_state.get("sankey_color_by", "Source")
                        link_alpha = float(st.session_state.get("sankey_link_opacity", 0.35) or 0.35)

                        # Helper to convert HEX + alpha to RGBA string
                        def hex_to_rgba(hex_color, alpha=0.35):
                            hex_color = hex_color.lstrip('#')
                            r = int(hex_color[0:2], 16)
                            g = int(hex_color[2:4], 16)
                            b = int(hex_color[4:6], 16)
                            return f"rgba({r}, {g}, {b}, {alpha})"

                        # Map node index to its color
                        idx_to_color = {int(r.node_index): node_color_map[(int(r.level), str(r.label))] for r in nodes_df.itertuples(index=False)}

                        link_colors = []
                        for r in links_all.itertuples(index=False):
                            node_idx = int(r.source_idx) if color_by_setting == "Source" else int(r.target_idx)
                            base_hex = idx_to_color.get(node_idx, "#999999")
                            link_colors.append(hex_to_rgba(base_hex, link_alpha))

                        # Prepare labels and links for Plotly
                        node_labels = nodes_df["label"].astype(str).tolist()
                        source_idx = links_all["source_idx"].astype(int).tolist()
                        target_idx = links_all["target_idx"].astype(int).tolist()
                        values = links_all["value"].astype(float).tolist()

                        # Optional hover text with level info
                        node_hovers = [f"Level {int(lvl)+1}: {lbl}" for lvl, lbl in nodes_df[["level", "label"]].itertuples(index=False, name=None)]

                        fig = go.Figure(
                            data=[
                                go.Sankey(
                                    node=dict(
                                        label=node_labels,
                                        hovertemplate="%{label}<extra></extra>",
                                        color=node_colors,
                                    ),
                                    link=dict(
                                        source=source_idx,
                                        target=target_idx,
                                        value=values,
                                        color=link_colors,
                                        hovertemplate="%{source.label} → %{target.label}: %{value}<extra></extra>",
                                    ),
                                    arrangement="snap"
                                )
                            ]
                        )
                        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)

                        # Diagnostics & downloads
                        # Build filter summary
                        filter_bits = []
                        if slicer_level and slicer_values:
                            filter_bits.append(f"{slicer_level} in {len(slicer_values)} selected")
                        filter_suffix = (" | Filters: " + " • ".join(filter_bits)) if filter_bits else ""
                        st.caption(
                            f"Levels: {' → '.join(levels)} | Top-N per level: {topn} | Other threshold: {other_pct:.1f}% of total | Rows: {len(work):,} | Total value: {total_value:,.2f}{filter_suffix}"
                        )

                        # Download links data
                        links_display = links_all[["level_from", "level_to", "source", "target", "value"]].copy()
                        links_csv = links_display.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download links CSV",
                            data=links_csv,
                            file_name="sankey_links.csv",
                            mime="text/csv",
                        )

                        # Download nodes data
                        nodes_display = nodes_df[["level", "label", "node_index"]].copy()
                        nodes_csv = nodes_display.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download nodes CSV",
                            data=nodes_csv,
                            file_name="sankey_nodes.csv",
                            mime="text/csv",
                        )
            except Exception as e:
                st.error(f"Error building Sankey: {e}")

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


