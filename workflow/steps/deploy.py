# workflow/steps/deploy.py
import argparse
import boto3
import time
from typing import Optional
from botocore.exceptions import ClientError


def _endpoint_status(sm, endpoint_name: str) -> Optional[str]:
    try:
        return sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    except ClientError as e:
        # endpoint가 없을 때도 ValidationException으로 떨어짐
        if e.response["Error"]["Code"] == "ValidationException":
            return None
        raise


def _wait_deleted(sm, endpoint_name: str, poll=10, timeout=900):
    start = time.time()
    while True:
        st = _endpoint_status(sm, endpoint_name)
        if st is None:
            return
        if time.time() - start > timeout:
            raise RuntimeError("Timed out waiting endpoint deletion")
        time.sleep(poll)


def _wait_in_service(sm, endpoint_name: str, poll=30, timeout=3600):
    start = time.time()
    while True:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        st = desc["EndpointStatus"]
        if st == "InService":
            return
        if st == "Failed":
            reason = desc.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint failed: {reason}")
        if time.time() - start > timeout:
            raise RuntimeError("Timed out waiting endpoint InService")
        time.sleep(poll)


def _get_latest_approved_package(sm, model_group: str) -> str:
    """
    Deploy는 반드시 Approved 모델만 사용해야 함.
    (PendingManualApproval이면 CreateModel 단계에서 막힘)
    """
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_group,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=50,
    )

    for p in resp.get("ModelPackageSummaryList", []):
        arn = p["ModelPackageArn"]
        d = sm.describe_model_package(ModelPackageName=arn)

        # 가장 최근 Approved+Completed를 선택
        if d.get("ModelApprovalStatus") == "Approved" and d.get("ModelPackageStatus") == "Completed":
            return arn

    raise RuntimeError(
        f"No Approved+Completed ModelPackage found in group: {model_group}. "
        f"Console에서 가장 최신 패키지를 Approved로 바꾼 뒤 다시 실행해."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-group", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--initial-instance-count", type=int, default=1)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    sm = boto3.client("sagemaker", region_name=args.region)

    # 1) Approved model package 선택
    model_package_arn = _get_latest_approved_package(sm, args.model_group)
    print("[deploy] using APPROVED model package:", model_package_arn)

    # 2) model/config은 매번 새 이름(충돌 방지)
    suffix = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.endpoint_name}-model-{suffix}"
    cfg_name = f"{args.endpoint_name}-cfg-{suffix}"

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role_arn,
        PrimaryContainer={"ModelPackageName": model_package_arn},
    )

    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": args.instance_type,
                "InitialInstanceCount": args.initial_instance_count,
            }
        ],
    )

    # 3) endpoint 상태에 따라 create / update / recreate
    st = _endpoint_status(sm, args.endpoint_name)

    if st is None:
        print("[deploy] endpoint not found -> create:", args.endpoint_name)
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)

    elif st == "Failed":
        print("[deploy] endpoint exists but Failed -> delete & recreate:", args.endpoint_name)
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        _wait_deleted(sm, args.endpoint_name)
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)

    else:
        print(f"[deploy] endpoint exists (status={st}) -> update:", args.endpoint_name)
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=cfg_name)

    print("🚀 Endpoint deployment started:", args.endpoint_name)

    if args.wait:
        _wait_in_service(sm, args.endpoint_name)
        print("✅ Endpoint InService:", args.endpoint_name)


if __name__ == "__main__":
    main()
