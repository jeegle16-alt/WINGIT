# workflow/steps/register.py
"""
Register ModelPackage (Inference Pipeline: preprocess(sklearn) -> xgboost)

- model.tar.gz (train output)을 풀어서
  1) preprocess_model.tar.gz : featurizer.joblib + (target_transform.json, route_stats.json optional) + code/
  2) xgb_model.tar.gz        : xgboost-model
  로 분리한 뒤 ModelPackage에 2개 컨테이너로 등록한다.
"""

import argparse
import os
import tarfile
import tempfile
import time
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError


def _ensure_model_package_group(sm, group_name: str):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    sm.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=f"Model group for {group_name}",
    )


def _parse_s3_uri(uri: str):
    p = urlparse(uri)
    if p.scheme != "s3":
        raise ValueError(f"Not an s3 uri: {uri}")
    return p.netloc, p.path.lstrip("/")


def _download_if_s3(s3_client, path_or_s3: str, local_dir: str) -> str:
    if path_or_s3.startswith("s3://"):
        bucket, key = _parse_s3_uri(path_or_s3)
        local_path = os.path.join(local_dir, os.path.basename(key) or "model.tar.gz")
        s3_client.download_file(bucket, key, local_path)
        return local_path
    return path_or_s3


def _upload_file(s3_client, local_path: str, bucket: str, key: str) -> str:
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _safe_extract_tar(tar_path: str, extract_dir: str):
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)


def _make_tar_gz(tar_path: str, base_dir: str, members_relpaths: list[str]):
    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in members_relpaths:
            abs_path = os.path.join(base_dir, rel)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Expected file/dir not found: {abs_path}")
            tf.add(abs_path, arcname=rel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tar", required=True)  # s3://... or local
    p.add_argument("--model-group", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--role-arn", required=True)

    p.add_argument("--preprocess-image-uri", required=True)
    p.add_argument("--xgb-image-uri", required=True)

    p.add_argument("--metrics-s3-uri", default="")

    p.add_argument("--artifact-bucket", required=True)
    p.add_argument("--artifact-prefix", required=True)

    p.add_argument("--auto-approve", action="store_true")
    args = p.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    _ensure_model_package_group(sm, args.model_group)

    suffix = time.strftime("%Y%m%d-%H%M%S")
    exec_arn = os.getenv("SM_PIPELINE_EXECUTION_ARN", "")
    if exec_arn:
        suffix = suffix + "-" + exec_arn.split("/")[-1][-8:]

    with tempfile.TemporaryDirectory() as td:
        local_model_tar = _download_if_s3(s3, args.model_tar, td)

        extracted = os.path.join(td, "extracted")
        os.makedirs(extracted, exist_ok=True)
        _safe_extract_tar(local_model_tar, extracted)

        xgb_model = os.path.join(extracted, "xgboost-model")
        fz = os.path.join(extracted, "featurizer.joblib")
        code_dir = os.path.join(extracted, "code")
        infer_py = os.path.join(code_dir, "inference.py")

        if not os.path.exists(xgb_model):
            raise FileNotFoundError(f"Missing xgboost-model: {xgb_model}")
        if not os.path.exists(fz):
            raise FileNotFoundError(f"Missing featurizer.joblib: {fz}")
        if not os.path.exists(infer_py):
            raise FileNotFoundError(f"Missing code/inference.py: {infer_py}")

        tt = os.path.join(extracted, "target_transform.json")
        rs = os.path.join(extracted, "route_stats.json")
        has_tt = os.path.exists(tt)
        has_rs = os.path.exists(rs)

        preprocess_tar = os.path.join(td, "preprocess_model.tar.gz")
        preprocess_members = ["featurizer.joblib", "code"]
        if has_tt:
            preprocess_members.append("target_transform.json")
        if has_rs:
            preprocess_members.append("route_stats.json")

        _make_tar_gz(preprocess_tar, extracted, preprocess_members)

        xgb_tar = os.path.join(td, "xgb_model.tar.gz")
        _make_tar_gz(xgb_tar, extracted, ["xgboost-model"])

        prefix = args.artifact_prefix.rstrip("/")
        preprocess_key = f"{prefix}/{suffix}/preprocess_model.tar.gz"
        xgb_key = f"{prefix}/{suffix}/xgb_model.tar.gz"

        preprocess_model_data_url = _upload_file(s3, preprocess_tar, args.artifact_bucket, preprocess_key)
        xgb_model_data_url = _upload_file(s3, xgb_tar, args.artifact_bucket, xgb_key)

    model_metrics = None
    if args.metrics_s3_uri and args.metrics_s3_uri.startswith("s3://"):
        model_metrics = {
            "ModelQuality": {
                "Statistics": {"ContentType": "application/json", "S3Uri": args.metrics_s3_uri}
            }
        }

    kwargs = dict(
        ModelPackageGroupName=args.model_group,
        ModelPackageDescription="Flight price prediction inference pipeline (preprocess -> xgboost)",
        InferenceSpecification={
            "Containers": [
                {"Image": args.preprocess_image_uri, "ModelDataUrl": preprocess_model_data_url},
                {"Image": args.xgb_image_uri, "ModelDataUrl": xgb_model_data_url},
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["text/csv", "application/json", "text/plain"],
        },
        ModelApprovalStatus="PendingManualApproval",
    )
    if model_metrics:
        kwargs["ModelMetrics"] = model_metrics

    resp = sm.create_model_package(**kwargs)
    arn = resp["ModelPackageArn"]

    print("✅ Model registered:", arn)
    print("  - preprocess ModelDataUrl:", preprocess_model_data_url)
    print("  - xgb       ModelDataUrl:", xgb_model_data_url)

    if args.auto_approve:
        sm.update_model_package(
            ModelPackageArn=arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription="Auto-approved by pipeline.",
        )
        print("✅ Approved:", arn)


if __name__ == "__main__":
    main()
