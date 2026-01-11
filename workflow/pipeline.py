# workflow/pipeline.py
from __future__ import annotations
from pathlib import Path

import sagemaker
from sagemaker.session import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.functions import Join

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve
from sagemaker.xgboost.estimator import XGBoost


THIS_DIR = Path(__file__).resolve().parent
STEPS_DIR = THIS_DIR / "steps"

# Versions
XGB_VERSION = "1.7-1"
SKLEARN_VERSION = "1.2-1"
DEFAULT_INSTANCE = "ml.m5.large"

# S3 (고정 버킷)
STUDIO_BUCKET = "sagemaker-studio-779874677565-8d22e350"
RAW_DATA_S3_URI = f"s3://{STUDIO_BUCKET}/flights/raw/raw.csv"

# Model Package Group
MODEL_GROUP_NAME = "FlightPriceXGB"

# Artifacts (register에서 model.tar.gz 업로드할 위치)
ARTIFACT_PREFIX_DEFAULT = f"flights/artifacts/{MODEL_GROUP_NAME}"


def get_pipeline(
    region: str | None = None,
    role: str | None = None,
    pipeline_name: str = "flight-price-xgb-pipeline",
) -> Pipeline:
    region = region or Session().boto_region_name
    role = role or sagemaker.get_execution_role()
    pipeline_sess = PipelineSession()

    # Pipeline outputs (S3)
    processed_s3 = f"s3://{STUDIO_BUCKET}/flights/processed"
    test_output_s3 = f"s3://{STUDIO_BUCKET}/flights/test-output"
    training_output_s3 = f"s3://{STUDIO_BUCKET}/flights/training-output"

    # ---------------------------
    # Pipeline Parameters
    # ---------------------------
    param_endpoint_name = ParameterString(
        name="EndpointName",
        default_value="flight-price-xgb-endpoint",
    )

    param_instance_type = ParameterString(
        name="InstanceType",
        default_value=DEFAULT_INSTANCE,
    )

    param_processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=1,
    )

    param_train_instance_count = ParameterInteger(
        name="TrainInstanceCount",
        default_value=1,
    )

    param_deploy_initial_instance_count = ParameterInteger(
        name="DeployInitialInstanceCount",
        default_value=1,
    )

    # register 업로드 목적지
    param_artifact_bucket = ParameterString(
        name="ArtifactBucket",
        default_value=STUDIO_BUCKET,
    )

    param_artifact_prefix = ParameterString(
        name="ArtifactPrefix",
        default_value=ARTIFACT_PREFIX_DEFAULT,
    )

    # ---------------------------
    # Images
    # ---------------------------
    sklearn_image = retrieve(
        framework="sklearn",
        region=region,
        version=SKLEARN_VERSION,
        py_version="py3",
        instance_type=DEFAULT_INSTANCE,
        image_scope="training",
    )

    xgb_train_image = retrieve(
        framework="xgboost",
        region=region,
        version=XGB_VERSION,
        py_version="py3",
        instance_type=DEFAULT_INSTANCE,
        image_scope="training",
    )

    xgb_infer_image = retrieve(
        framework="xgboost",
        region=region,
        version=XGB_VERSION,
        py_version="py3",
        instance_type=DEFAULT_INSTANCE,
        image_scope="inference",
    )

    # ---------------------------
    # Step 1) Preprocess
    # ---------------------------
    preprocess_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=pipeline_sess,
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess_processor,
        code=str(STEPS_DIR / "preprocess.py"),
        inputs=[
            # ✅ destination은 "디렉토리"여야 한다.
            ProcessingInput(
                source=RAW_DATA_S3_URI,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output",
                destination=processed_s3,
            )
        ],
        job_arguments=[
            "--input-file",
            "/opt/ml/processing/input/raw.csv",
            "--output-dir",
            "/opt/ml/processing/output",
        ],
    )

    train_s3 = Join(on="/", values=[processed_s3, "train_numeric.csv"])
    val_s3 = Join(on="/", values=[processed_s3, "validation_numeric.csv"])

    # ---------------------------
    # Step 2) Train
    # ---------------------------
    estimator = XGBoost(
        entry_point="train.py",
        source_dir=str(STEPS_DIR),
        role=role,
        instance_count=param_train_instance_count,
        instance_type=param_instance_type,
        framework_version=XGB_VERSION,
        py_version="py3",
        output_path=training_output_s3,
        sagemaker_session=pipeline_sess,
        hyperparameters={
            "objective": "reg:squarederror",
            "eval-metric": "rmse",
            "n-estimators": 300,
            "learning-rate": 0.05,
            "max-depth": 6,
            "subsample": 0.8,
            "colsample-bytree": 0.8,
            "min-child-weight": 5.0,
            "reg-alpha": 0.1,
            "reg-lambda": 1.0,
            "early-stopping-rounds": 30,
            "seed": 42,
            "tree-method": "hist",
            "max-bin": 256,
        },
    )

    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(s3_data=train_s3, content_type="text/csv"),
            "validation": TrainingInput(s3_data=val_s3, content_type="text/csv"),
        },
        depends_on=[step_preprocess.name],
    )

    # ---------------------------
    # Step 3) Test
    # ---------------------------
    test_processor = ScriptProcessor(
        image_uri=xgb_train_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=pipeline_sess,
    )

    step_test = ProcessingStep(
        name="Test",
        processor=test_processor,
        code=str(STEPS_DIR / "test.py"),
        inputs=[
            # model.tar.gz 받기
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            # preprocess output 받기 (test_numeric.csv 포함)
            ProcessingInput(
                source=processed_s3,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="test_output",
                source="/opt/ml/processing/output",
                destination=test_output_s3,
            )
        ],
        job_arguments=[
            "--model-tar",
            "/opt/ml/processing/model/model.tar.gz",
            "--test-dir",
            "/opt/ml/processing/test",
            "--test-file",
            "test_numeric.csv",
            "--output-dir",
            "/opt/ml/processing/output",
        ],
        depends_on=[step_train.name],
    )

    # ---------------------------
    # Step 4) Register  (Model Registry)
    # ---------------------------
    register_processor = ScriptProcessor(
        image_uri=xgb_train_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=pipeline_sess,
    )

    step_register = ProcessingStep(
        name="Register",
        processor=register_processor,
        code=str(STEPS_DIR / "register.py"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/model",
            ),
            # 테스트 메트릭도 같이 받아서 metrics-s3-uri로 넣고 싶으면
            # S3로 올라간 test_output을 직접 참조하면 됨 (아래 job_arguments에 metrics-s3-uri 사용)
        ],
        job_arguments=[
            "--model-tar",
            "/opt/ml/model/model.tar.gz",
            "--model-group",
            MODEL_GROUP_NAME,
            "--region",
            region,
            "--image-uri",
            xgb_infer_image,
            "--role-arn",
            role,
            "--artifact-bucket",
            param_artifact_bucket.to_string(),
            "--artifact-prefix",
            param_artifact_prefix.to_string(),
            # (선택) test metrics를 model registry에 연결하려면 실제 S3 URI를 넣어야 함
            # "--metrics-s3-uri",
            # Join(on="/", values=[test_output_s3, "test_metrics.json"]),
        ],
        depends_on=[step_test.name],
    )

    # ---------------------------
    # Step 5) Deploy
    # ---------------------------
    deploy_processor = ScriptProcessor(
        image_uri=xgb_train_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=pipeline_sess,
    )

    step_deploy = ProcessingStep(
        name="Deploy",
        processor=deploy_processor,
        code=str(STEPS_DIR / "deploy.py"),
        job_arguments=[
            "--model-group",
            MODEL_GROUP_NAME,
            "--endpoint-name",
            param_endpoint_name.to_string(),
            "--region",
            region,
            "--instance-type",
            param_instance_type.to_string(),
            "--initial-instance-count",
            param_deploy_initial_instance_count.to_string(),
            "--role-arn",
            role,
            "--wait",
        ],
        depends_on=[step_register.name],
    )

    # ---------------------------
    # Pipeline
    # ---------------------------
    return Pipeline(
        name=pipeline_name,
        parameters=[
            param_endpoint_name,
            param_instance_type,
            param_processing_instance_count,
            param_train_instance_count,
            param_deploy_initial_instance_count,
            param_artifact_bucket,
            param_artifact_prefix,
        ],
        steps=[
            step_preprocess,
            step_train,
            step_test,
            step_register,
            step_deploy,
        ],
        sagemaker_session=pipeline_sess,
    )
