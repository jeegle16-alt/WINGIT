# workflow/steps/register.py
import argparse
import os
import time

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


def _upload_if_local(s3_client, local_path: str, bucket: str, key: str) -> str:
    if local_path.startswith("s3://"):
        return local_path
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _wait_approval(sm, model_package_arn: str, desired: str = "Approved", timeout: int = 120, poll: int = 3):
    """Wait until ModelApprovalStatus becomes desired (handles eventual consistency)."""
    start = time.time()
    while True:
        d = sm.describe_model_package(ModelPackageName=model_package_arn)
        status = d.get("ModelApprovalStatus")
        if status == desired:
            return
        if time.time() - start > timeout:
            raise RuntimeError(
                f"Timed out waiting for approval status '{desired}'. Current='{status}', arn='{model_package_arn}'"
            )
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tar", required=True)        # local path OR s3 uri
    parser.add_argument("--model-group", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)

    parser.add_argument("--metrics-s3-uri", required=False, default="")

    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--artifact-prefix", required=True)  # e.g. flights/artifacts/FlightPriceXGB

    # ✅ for end-to-end pipeline
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve the created model package so Deploy can run in the same pipeline execution.",
    )

    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    _ensure_model_package_group(sm, args.model_group)

    # 충돌 방지: timestamp + (가능하면 pipeline execution id 일부)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    exec_arn = os.getenv("SM_PIPELINE_EXECUTION_ARN", "")
    if exec_arn:
        suffix = suffix + "-" + exec_arn.split("/")[-1][-8:]

    model_key = f"{args.artifact_prefix.rstrip('/')}/{suffix}/model.tar.gz"
    model_data_url = _upload_if_local(
        s3_client=s3,
        local_path=args.model_tar,
        bucket=args.artifact_bucket,
        key=model_key,
    )

    model_metrics = None
    if args.metrics_s3_uri and args.metrics_s3_uri.startswith("s3://"):
        model_metrics = {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": args.metrics_s3_uri,
                }
            }
        }

    kwargs = dict(
        ModelPackageGroupName=args.model_group,
        ModelPackageDescription="Flight price prediction XGBoost model",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": args.image_uri,
                    "ModelDataUrl": model_data_url,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        # 기본값은 Pending. auto-approve면 바로 Approved로 올릴 것.
        ModelApprovalStatus="PendingManualApproval",
    )

    if model_metrics:
        kwargs["ModelMetrics"] = model_metrics

    response = sm.create_model_package(**kwargs)
    model_package_arn = response["ModelPackageArn"]

    print("✅ Model registered")
    print("ModelPackageArn:", model_package_arn)
    print("ModelDataUrl:", model_data_url)
    print("ApprovalStatus: PendingManualApproval")

    # ✅ Auto approve for one-shot pipeline
    if args.auto_approve:
        print("[register] auto-approving model package:", model_package_arn)
        sm.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus="Approved",
            ApprovalDescription="Auto-approved by pipeline for end-to-end execution.",
        )
        _wait_approval(sm, model_package_arn, desired="Approved", timeout=180, poll=3)
        print("✅ ApprovalStatus: Approved")


if __name__ == "__main__":
    main()
