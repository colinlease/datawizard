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
        messages.append("Dataset is extremely clean. No missing values, well-defined structure.")
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")

# ---------- Sidebar Upload ----------
st.sidebar.header("Upload Data")
sidebar_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"], key="sidebar")

# ---------- Main Layout ----------
st.title("DataWizard")
st.subheader("Upload your data to begin analysis")
st.write(
    "This tool lets you upload a CSV or Excel file and quickly view summary statistics, charts, distributions, run regressions, cluster your data, and more. "
    "You can upload your file using the button below or from the sidebar."
)

main_file = st.file_uploader("Upload your file here", type=["csv", "xls", "xlsx"], key="main")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
        <strong>Need to combine multiple datasets before analysis?</strong><br>
        Use our companion tool, <a href='https://datablender.streamlit.app/' target='_blank'>DataBlender</a>.
    </div>
    """,
    unsafe_allow_html=True
)
uploaded_file = main_file if main_file is not None else sidebar_file

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if df.shape[0] > 75000:
            st.error("File exceeds 75,000 row limit. Please upload a smaller file.")
        else:
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            # Reset selected features in session state when a new file is uploaded
            st.session_state.selected_features = []

            inferred_types = {}
            for col in df.columns:
                dtype = pd.api.types.infer_dtype(df[col], skipna=True)
                if dtype in ["string", "categorical", "object"]:
                    # Try parsing to datetime
                    parsed_dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    non_null_ratio = parsed_dates.notnull().mean()
                    if non_null_ratio > 0.9:
                        df[col] = parsed_dates
                        inferred_types[col] = "datetime"
                    else:
                        inferred_types[col] = "categorical"
                elif dtype in ["integer", "floating", "mixed-integer-float"]:
                    inferred_types[col] = "numeric"
                elif dtype.startswith("datetime"):
                    inferred_types[col] = "datetime"
                else:
                    inferred_types[col] = "other"

            cleanliness_messages = evaluate_data_cleanliness(df, inferred_types)

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

            st.sidebar.markdown("---")
            st.sidebar.markdown("### File Summary")
            if cleanliness_messages:
                for msg in cleanliness_messages:
                    st.sidebar.info(f"**Data Quality Check**\n\n{msg}")
            st.sidebar.write(f"**Name:** {uploaded_file.name}")
            st.sidebar.write(f"**Rows:** {df.shape[0]}")
            st.sidebar.write(f"**Columns:** {df.shape[1]}")
            st.sidebar.write(f"**Missing Values:** {int(df.isna().sum().sum())}")

            type_counts = Counter(inferred_types.values())
            st.sidebar.markdown("### Column Types")
            for t, count in type_counts.items():
                label = t.capitalize() if t != "other" else "Other/Unknown"
                st.sidebar.write(f"{label}: {count}")

            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.markdown("### Feature Selection")

                # Filter allowed features
                allowed_feature_types = {"categorical", "datetime", "numeric"}
                allowed_features = [col for col in df.columns if inferred_types.get(col) in allowed_feature_types or col.startswith("t - ")]

                # Initialize session state if needed
                if "selected_features" not in st.session_state:
                    st.session_state.selected_features = []

                # UI layout: Buttons first
                feature_col1, feature_col2 = st.columns([3, 1])
                with feature_col2:
                    if st.button("Select All"):
                        st.session_state.selected_features = allowed_features.copy()  # use copy to avoid reference issues
                    if st.button("Clear All"):
                        st.session_state.selected_features = []

                with feature_col1:
                    selected_features = st.multiselect(
                        "Select one or more features",
                        options=allowed_features,
                        default=st.session_state.selected_features
                    )

                if selected_features != st.session_state.selected_features:
                    st.session_state.selected_features = selected_features

                st.markdown("### Analysis Type")
                analysis_type = st.selectbox(
                    "Select one type of analysis",
                    ["Summary statistics", "Histogram", "Correlation matrix", "Linear Regression", "Clustering", "Line Chart"]
                )

                # Add below analysis_type selectbox
                cluster_k = None
                if analysis_type == "Clustering":
                    cluster_k = st.selectbox(
                        "Select number of clusters (k)",
                        options=list(range(2, 11)),
                        index=1,  # Default is 3 (index 1 of range 2–10)
                        key="cluster_dropdown"
                    )

                # Reset clustering run flag if analysis type changes
                if "last_analysis_type" not in st.session_state:
                    st.session_state.last_analysis_type = None

                if st.session_state.last_analysis_type != analysis_type:
                    st.session_state.run_clustering = False
                    st.session_state.last_analysis_type = analysis_type

                dependent_var = None
                if analysis_type == "Linear Regression" and selected_features:
                    dependent_var = st.selectbox(
                        "Select dependent variable",
                        options=selected_features,
                        key="dep_var_select"
                    )

                run_analysis = st.button("Run Analysis")

            with right_col:
                st.markdown("### Data Preview (First 10 Rows)")
                st.dataframe(df.head(10), use_container_width=True)

            if run_analysis:
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
                    st.markdown("## Correlation Matrix")

                    non_numeric_fields = [f for f in selected_features if inferred_types.get(f) != "numeric"]
                    if non_numeric_fields:
                        st.error("All selected features must be numeric for correlation analysis. "
                                 f"The following fields are not: {', '.join(non_numeric_fields)}")
                    elif len(selected_features) < 2:
                        st.warning("Please select at least two numeric features to compute correlation.")
                    else:
                        corr_data = df[selected_features].dropna()
                        corr_matrix = corr_data.corr()

                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        fig, ax = plt.subplots(figsize=(min(10, 1 + 0.6 * len(selected_features)), 6))
                        cax = ax.imshow(
                            corr_matrix,
                            cmap='coolwarm',
                            vmin=-1,
                            vmax=1
                        )
                        ax.set_xticks(np.arange(len(corr_matrix.columns)))
                        ax.set_yticks(np.arange(len(corr_matrix.index)))
                        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                        ax.set_yticklabels(corr_matrix.index)

                        for i in range(len(corr_matrix)):
                            for j in range(len(corr_matrix)):
                                if mask[i, j]:
                                    continue
                                ax.text(
                                    j, i,
                                    f"{corr_matrix.iloc[i, j]:.2f}",
                                    ha='center',
                                    va='center',
                                    color="black",
                                    fontsize=8
                                )

                        ax.set_title("Correlation Heatmap")
                        fig.colorbar(cax, ax=ax, shrink=0.75)
                        st.pyplot(fig)

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

                elif analysis_type == "Line Chart":
                    st.markdown("## Line Charts")

                    # Identify valid x-axis options
                    valid_x_fields = [col for col in df.columns if inferred_types.get(col) == "datetime" or col.startswith("t - ")]

                    if not valid_x_fields:
                        st.warning("Line charts require a time-based x-axis. No datetime or t-index fields were found.")
                    else:
                        x_field = st.selectbox("Select a time-based field for the X-axis", options=valid_x_fields, key="x_axis_select")
                        y_fields = [f for f in selected_features if inferred_types.get(f) == "numeric"]

                        if not y_fields:
                            st.warning("No numeric features selected for the Y-axis.")
                        else:
                            if len(y_fields) > 6:
                                st.warning("More than 6 numeric features selected. Only the first 6 will be shown.")
                                y_fields = y_fields[:6]

                            for i, y_field in enumerate(y_fields):
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
                                    st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
