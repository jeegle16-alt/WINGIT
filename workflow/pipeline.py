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

# -----------------------------------------------------------------------------
# Versions / Defaults
# -----------------------------------------------------------------------------
XGB_VERSION = "1.7-1"
SKLEARN_VERSION = "1.2-1"
DEFAULT_INSTANCE = "ml.m5.large"

# -----------------------------------------------------------------------------
# Your Studio bucket / project folder
# -----------------------------------------------------------------------------
STUDIO_BUCKET = "sagemaker-studio-403472739383-7417e040"
FOLDER = "flights"

# Raw data location (object MUST be named raw_data.csv if you reference it as below)
RAW_DATA_S3_URI = f"s3://{STUDIO_BUCKET}/{FOLDER}/raw_data/raw_data.csv"

# Model Registry
MODEL_GROUP_NAME = "FlightPriceXGB"
ARTIFACT_PREFIX_DEFAULT = f"{FOLDER}/artifacts/{MODEL_GROUP_NAME}"

# MLflow (Managed Tracking Server ARN)
DEFAULT_MLFLOW_TRACKING_URI = (
    "arn:aws:sagemaker:us-east-1:403472739383:mlflow-tracking-server/flights"
)


def get_pipeline(
    region: str | None = None,
    role: str | None = None,
    pipeline_name: str = "flight-price-xgb-pipeline",
) -> Pipeline:
    # -------------------------------------------------------------------------
    # Region / Role / Sessions
    # -------------------------------------------------------------------------
    region = region or Session().boto_region_name
    role = role or sagemaker.get_execution_role()

    # Pin default bucket to Studio bucket (prevents auto-creating sagemaker-<region>-<acct> in many cases)
    base_sess = sagemaker.session.Session(default_bucket=STUDIO_BUCKET)

    # PipelineSession used for pipeline definition/compile/submit
    pipeline_sess = PipelineSession()

    # -------------------------------------------------------------------------
    # S3 paths (RESULTS under flights/)
    # -------------------------------------------------------------------------
    processed_s3 = f"s3://{STUDIO_BUCKET}/{FOLDER}/processed"
    test_output_s3 = f"s3://{STUDIO_BUCKET}/{FOLDER}/test-output"
    training_output_s3 = f"s3://{STUDIO_BUCKET}/{FOLDER}/training-output"

    # We can force training code upload under flights/_staging (Estimator supports code_location)
    code_loc = f"s3://{STUDIO_BUCKET}/{FOLDER}/_staging"

    # -------------------------------------------------------------------------
    # Pipeline Parameters
    # -------------------------------------------------------------------------
    param_endpoint_name = ParameterString("EndpointName", default_value="flight-price-xgb-endpoint")

    param_instance_type = ParameterString("InstanceType", default_value=DEFAULT_INSTANCE)
    param_processing_instance_count = ParameterInteger("ProcessingInstanceCount", default_value=1)
    param_train_instance_count = ParameterInteger("TrainInstanceCount", default_value=1)
    param_deploy_initial_instance_count = ParameterInteger("DeployInitialInstanceCount", default_value=1)

    param_artifact_bucket = ParameterString("ArtifactBucket", default_value=STUDIO_BUCKET)
    param_artifact_prefix = ParameterString("ArtifactPrefix", default_value=ARTIFACT_PREFIX_DEFAULT)

    param_mlflow_tracking_uri = ParameterString("MLflowTrackingURI", default_value=DEFAULT_MLFLOW_TRACKING_URI)
    param_mlflow_experiment = ParameterString("MLflowExperimentName", default_value="FlightPriceXGB")

    mlflow_env = {
        "MLFLOW_TRACKING_URI": param_mlflow_tracking_uri,
        "MLFLOW_EXPERIMENT_NAME": param_mlflow_experiment,
    }

    # -------------------------------------------------------------------------
    # Images (resolved at compile time; keep instance_type constant here)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 1) Preprocess
    # NOTE: ScriptProcessor in your SDK does NOT support code_location.
    # We still pin sagemaker_session to base_sess and keep outputs under flights/.
    # -------------------------------------------------------------------------
    preprocess_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=base_sess,  # ✅ default bucket pinned here
        base_job_name=f"{FOLDER}-preprocess",
        env={**mlflow_env, "MLFLOW_RUN_NAME": "preprocess", "MLFLOW_NESTED": "0"},
    )

    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=preprocess_processor,
        code=str(STEPS_DIR / "preprocess.py"),
        inputs=[
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
            "/opt/ml/processing/input/raw_data.csv",
            "--output-dir",
            "/opt/ml/processing/output",
            "--outlier-method",
            "clip",
            "--scale-numeric",
        ],
    )

    # Explicit DAG: downstream uses the actual preprocess output URI
    processed_uri = step_preprocess.properties.ProcessingOutputConfig.Outputs[
        "processed"
    ].S3Output.S3Uri

    train_s3 = Join(on="/", values=[processed_uri, "train_numeric.csv"])
    val_s3 = Join(on="/", values=[processed_uri, "validation_numeric.csv"])
    test_s3 = Join(on="/", values=[processed_uri, "test_numeric.csv"])

    # -------------------------------------------------------------------------
    # Step 2) Train
    # Estimator supports code_location -> training code upload goes under flights/_staging
    # -------------------------------------------------------------------------
    estimator = XGBoost(
        entry_point="train.py",
        source_dir=str(STEPS_DIR),
        role=role,
        instance_type=param_instance_type,
        instance_count=param_train_instance_count,
        framework_version=XGB_VERSION,
        output_path=training_output_s3,
        sagemaker_session=base_sess,
        code_location=code_loc,  # ✅ works for estimator
        base_job_name=f"{FOLDER}-xgb",
        hyperparameters={
            "objective": "reg:squarederror",
            "n-estimators": 300,
            "learning-rate": 0.05,
            "max-depth": 6,
            "subsample": 0.8,
            "colsample-bytree": 0.8,
            "min-child-weight": 5,
            "reg-alpha": 0.1,
            "reg-lambda": 1.0,
            "early-stopping-rounds": 30,
        },
        environment={
            **mlflow_env,
            "MLFLOW_RUN_NAME": "train",
            "MLFLOW_NESTED": "0",
        },
    )

    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(s3_data=train_s3, content_type="text/csv"),
            "validation": TrainingInput(s3_data=val_s3, content_type="text/csv"),
        },
        depends_on=[step_preprocess],
    )

    # -------------------------------------------------------------------------
    # Step 3) Test
    # NOTE: ScriptProcessor code_location not available in your SDK.
    # -------------------------------------------------------------------------
    test_processor = ScriptProcessor(
        image_uri=xgb_train_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=base_sess,
        base_job_name=f"{FOLDER}-test",
        env={**mlflow_env, "MLFLOW_RUN_NAME": "test", "MLFLOW_NESTED": "0"},
    )

    step_test = ProcessingStep(
        name="Test",
        processor=test_processor,
        code=str(STEPS_DIR / "test.py"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=processed_uri,
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
        depends_on=[step_train],
        job_arguments=[
            "--model-tar",
            "/opt/ml/processing/model/model.tar.gz",
            "--test-dir",
            "/opt/ml/processing/test",
            "--test-file",
            "test_numeric.csv",
            "--output-dir",
            "/opt/ml/processing/output",
            "--extract-dir",
            "/opt/ml/processing/model_extracted",
            "--target-transform",
            "log1p",
        ],
    )

    # Explicit DAG: Register uses Test output URI (so it depends on Test)
    test_output_uri = step_test.properties.ProcessingOutputConfig.Outputs[
        "test_output"
    ].S3Output.S3Uri
    metrics_s3_uri = Join(on="/", values=[test_output_uri, "test_metrics.json"])

    # -------------------------------------------------------------------------
    # Step 4) Register
    # NOTE: ScriptProcessor code_location not available in your SDK.
    # -------------------------------------------------------------------------
    register_processor = ScriptProcessor(
        image_uri=xgb_train_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=base_sess,
        base_job_name=f"{FOLDER}-register",
        env={**mlflow_env, "MLFLOW_RUN_NAME": "register", "MLFLOW_NESTED": "0"},
    )

    step_register = ProcessingStep(
        name="Register",
        processor=register_processor,
        code=str(STEPS_DIR / "register.py"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
        ],
        depends_on=[step_test],
        job_arguments=[
            "--model-tar",
            "/opt/ml/processing/model/model.tar.gz",
            "--model-group",
            MODEL_GROUP_NAME,
            "--region",
            region,
            "--image-uri",
            xgb_infer_image,
            "--role-arn",
            role,
            "--artifact-bucket",
            param_artifact_bucket,
            "--artifact-prefix",
            param_artifact_prefix,
            "--metrics-s3-uri",
            metrics_s3_uri,
            "--auto-approve", 
        ],
    )

    # -------------------------------------------------------------------------
    # Step 5) Deploy
    # NOTE: ScriptProcessor code_location not available in your SDK.
    # -------------------------------------------------------------------------
    deploy_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role,
        instance_type=param_instance_type,
        instance_count=param_processing_instance_count,
        sagemaker_session=base_sess,
        base_job_name=f"{FOLDER}-deploy",
        env={**mlflow_env, "MLFLOW_RUN_NAME": "deploy", "MLFLOW_NESTED": "0"},
    )

    step_deploy = ProcessingStep(
        name="Deploy",
        processor=deploy_processor,
        code=str(STEPS_DIR / "deploy.py"),
        depends_on=[step_register],
        job_arguments=[
            "--model-group",
            MODEL_GROUP_NAME,
            "--endpoint-name",
            param_endpoint_name,
            "--region",
            region,
            "--instance-type",
            param_instance_type,
            "--initial-instance-count",
            param_deploy_initial_instance_count.to_string(),
            "--role-arn",
            role,
            "--wait",
        ],
    )

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------
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
            param_mlflow_tracking_uri,
            param_mlflow_experiment,
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
