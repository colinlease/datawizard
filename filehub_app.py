# filehub.py
# Microservice module for secure file transfer between Streamlit apps via S3

import boto3
import uuid
import os
import pandas as pd
from io import StringIO
from datetime import datetime
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# Prefer Streamlit secrets for Streamlit Cloud compatibility, fallback to env vars
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", os.getenv("S3_BUCKET_NAME", "your-bucket-name"))
S3_REGION = st.secrets.get("S3_REGION", os.getenv("S3_REGION", "us-east-1"))

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)

def generate_token():
    return str(uuid.uuid4())[:8]

def upload_dataframe(df: pd.DataFrame, source_app: str, original_filename: str) -> str:
    token = generate_token()
    sanitized_filename = original_filename.replace(" ", "_").replace("/", "-")
    object_key = f"{source_app}/{token}__{sanitized_filename}.csv"

    csv_buffer = StringIO()
    include_index = df.index.name is not None
    df.to_csv(csv_buffer, index=include_index)

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=object_key,
        Body=csv_buffer.getvalue(),
        ContentType="text/csv",
        Tagging="ttl=900"  # 15 minutes in seconds
    )

    return token

def find_file_by_token(source_app: str, token: str):
    prefix = f"{source_app}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
    
    for obj in response.get("Contents", []):
        if f"{token}__" in obj["Key"]:
            return obj["Key"]
    return None

def download_dataframe(token: str, source_app: str = None) -> tuple[pd.DataFrame, str]:
    source_apps = [source_app] if source_app else ["DataWizard", "DataBlender", "DataSampler"]
    for app in source_apps:
        key = find_file_by_token(app, token)
        if key:
            response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
            content = response["Body"].read().decode("utf-8")
            df = pd.read_csv(StringIO(content))
            original_name = key.split("__", 1)[-1].replace(".csv", "")
            return df, original_name
    raise FileNotFoundError("No file found for provided token.")

def list_active_filehub_objects_ui():
    st.header("📂 Current Files in FileHub (S3)")
    st.markdown("<br><br>", unsafe_allow_html=True)

    response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
    if "Contents" not in response:
        st.info("No files currently stored.")
        return

    now = datetime.utcnow()
    active_files = [
        obj for obj in response["Contents"]
        if (now - obj["LastModified"].replace(tzinfo=None)).total_seconds() < 900
    ]

    total_bytes = sum(obj["Size"] for obj in active_files)
    total_mb = total_bytes / (1024 * 1024)
    st.markdown(f"**Total Active File Size:** `{total_mb:.2f} MB`")
    st.markdown(f"**Active File Count:** `{len(active_files)}`")
    st.markdown("<br>", unsafe_allow_html=True)

    for obj in sorted(active_files, key=lambda x: x["LastModified"], reverse=True):
        key = obj["Key"]
        last_modified = obj["LastModified"].replace(tzinfo=None)
        age = (now - last_modified).total_seconds()
        time_remaining = max(0, 900 - int(age))  # 15 min TTL = 900 sec
        token_masked_key = key
        if "/" in key and "__" in key:
            prefix, rest = key.split("/", 1)
            token_and_filename = rest.split("__", 1)
            if len(token_and_filename) == 2:
                token, filename = token_and_filename
                if len(token) == 8:
                    masked_token = "XXXXXX" + token[-2:]
                    token_masked_key = f"{prefix}/{masked_token}__{filename}"
        file_size_mb = obj["Size"] / (1024 * 1024)
        col1, col2, col3 = st.columns([6, 2, 3])
        col1.markdown(f"**{token_masked_key}**")
        col2.markdown(f"`{file_size_mb:.2f} MB`")
        color = "green"
        if time_remaining < 180:
            color = "red"
        elif time_remaining < 450:
            color = "orange"
        col3.markdown(
            f"<span style='font-family:monospace'>Expires in <span style='color:{color}'>{time_remaining}</span> sec</span>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    st.set_page_config(page_title="Admin Console – FileHub Backend Transfers")
    st_autorefresh(interval=1000, key="auto-refresh")
    list_active_filehub_objects_ui()