# DataWizard

**DataWizard** is a modular Streamlit application for lightweight exploratory data analysis. It allows users to upload, examine, and analyze datasets (CSV or Excel) using built-in tools for summary statistics, visualization, regression, clustering, and cross-application file transfer via the FileHub microservice. The application is optimized for usability, flexibility, and compatibility across large datasets (up to 75,000 rows).

## Features

- **Data Ingestion**
  - Supports `.csv`, `.xlsx`, `.xls` file formats
  - Maximum total row limit: 75,000 rows
  - Supports joining or unioning two uploaded datasets

- **Exploratory Data Analysis**
  - Preview table and basic structure of the uploaded dataset
  - Summary statistics: mean, median, min, max, standard deviation
  - Categorical frequency: most/least common values
  - Trend analysis and seasonality detection (if applicable)

- **Correlation and Regression**
  - Correlation matrix with upper triangle masking
  - Simple linear regression with coefficient display, p-values, R², and visualization
  - Support for time-based regression if a datetime column is present

- **Clustering**
  - K-means clustering with user-defined number of clusters
  - Dimensionality reduction using PCA for 2D visualization
  - Optional custom labels for each cluster

- **FileHub Integration**
  - Import datasets using a secure token issued by FileHub
  - Export any processed dataset to FileHub for downstream usage in other apps
  - Tokens expire after a fixed TTL (default: 15 minutes)

- **Interface and Logic**
  - Clean session state management for user selections and transfers
  - UI adapts based on available features and input types
  - Upload or token-based access supported as interchangeable workflows

## File Structure