# workflow/steps/register.py
import argparse
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
    # ProcessingStep input으로 들어오는 경로는 보통 local path임 (/opt/ml/...)
    # s3://면 그대로 쓰고, 아니면 S3로 업로드해서 S3 URI를 반환
    if local_path.startswith("s3://"):
        return local_path

    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tar", required=True)        # local path OR s3 uri
    parser.add_argument("--model-group", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)

    # (선택) test step에서 만든 metrics JSON이 S3에 있을 때만 넣기
    parser.add_argument("--metrics-s3-uri", required=False, default="")

    # (필수) 업로드 목적지
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--artifact-prefix", required=True)  # e.g. flights/artifacts/FlightPriceXGB
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    # 1) ModelPackageGroup 보장
    _ensure_model_package_group(sm, args.model_group)

    # 2) model.tar.gz 를 S3로 업로드(로컬 경로면)
    model_key = f"{args.artifact_prefix.rstrip('/')}/model.tar.gz"
    model_data_url = _upload_if_local(
        s3_client=s3,
        local_path=args.model_tar,
        bucket=args.artifact_bucket,
        key=model_key,
    )

    # 3) metrics가 유효할 때만 ModelMetrics 포함
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
        ModelApprovalStatus="PendingManualApproval",  # 팀 합의대로면 여기 유지
    )

    if model_metrics:
        kwargs["ModelMetrics"] = model_metrics

    response = sm.create_model_package(**kwargs)

    print("✅ Model registered")
    print("ModelPackageArn:", response["ModelPackageArn"])
    print("ModelDataUrl:", model_data_url)
    print("ApprovalStatus: PendingManualApproval")


if __name__ == "__main__":
    main()
