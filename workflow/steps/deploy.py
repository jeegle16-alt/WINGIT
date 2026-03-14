# workflow/steps/deploy.py
"""
Deploy latest COMPLETED ModelPackage (2-container inference pipeline) to a SageMaker Endpoint.

- Container0 (sklearn inference): runs inference.py to convert JSON -> CSV
- Container1 (xgboost inference): runs built-in serving to predict from CSV
"""

import argparse
import time
from typing import Optional, List, Dict, Any

import boto3
from botocore.exceptions import ClientError


def _endpoint_status(sm, endpoint_name: str) -> Optional[str]:
    try:
        return sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return None
        raise


def _wait_deleted(sm, endpoint_name: str, poll: int = 10, timeout: int = 900):
    start = time.time()
    while True:
        st = _endpoint_status(sm, endpoint_name)
        if st is None:
            return
        if time.time() - start > timeout:
            raise RuntimeError(f"Timed out waiting endpoint deletion: {endpoint_name}")
        time.sleep(poll)


def _wait_in_service(sm, endpoint_name: str, poll: int = 30, timeout: int = 3600):
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
            raise RuntimeError(f"Timed out waiting endpoint InService: {endpoint_name}")
        time.sleep(poll)


def _get_latest_completed_package(sm, model_group: str) -> str:
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_group,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=50,
    )

    for p in resp.get("ModelPackageSummaryList", []):
        arn = p["ModelPackageArn"]
        d = sm.describe_model_package(ModelPackageName=arn)
        if d.get("ModelPackageStatus") == "Completed":
            approval = d.get("ModelApprovalStatus")
            print(f"[deploy] picked package: {arn} (Approval={approval}, Status=Completed)")
            return arn

    raise RuntimeError(f"No Completed ModelPackage found in group: {model_group}")


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

    # 1) pick model package
    model_package_arn = _get_latest_completed_package(sm, args.model_group)
    print("[deploy] using model package:", model_package_arn)

    # 2) read 2 containers from package
    pkg = sm.describe_model_package(ModelPackageName=model_package_arn)
    containers: List[Dict[str, Any]] = pkg.get("InferenceSpecification", {}).get("Containers", [])
    if len(containers) < 2:
        raise RuntimeError(
            f"Expected >=2 containers in ModelPackage InferenceSpecification, got {len(containers)}: {model_package_arn}"
        )

    c0 = containers[0]
    c1 = containers[1]

    for i, c in enumerate([c0, c1]):
        if not c.get("Image") or not c.get("ModelDataUrl"):
            raise RuntimeError(f"Container[{i}] missing Image/ModelDataUrl in ModelPackage: {c}")

    # 3) create model + endpoint config
    suffix = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{args.endpoint_name}-model-{suffix}"
    cfg_name = f"{args.endpoint_name}-cfg-{suffix}"

    print("[deploy] creating model:", model_name)

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role_arn,
        Containers=[
            # Container 0: sklearn preprocess
            {
                "ContainerHostname": "sklearn-preprocess",
                "Image": c0["Image"],
                "ModelDataUrl": c0["ModelDataUrl"],
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
            },
            # Container 1: xgboost
            {
                "ContainerHostname": "xgboost",
                "Image": c1["Image"],
                "ModelDataUrl": c1["ModelDataUrl"],
            },
        ],
    )

    print("[deploy] creating endpoint config:", cfg_name)
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

    # 4) create/update endpoint
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
