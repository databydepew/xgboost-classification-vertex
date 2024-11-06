
import argparse
import json
from kfp.dsl import executor

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

def xgboost_training(
    dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    project_id: str,
    region: str
):
    """Trains an XGBoost classifier.

    Args:
        dataset: The training dataset.

    Returns:
        model: The model artifact stores the model.joblib file.
        metrics: The metrics of the trained model.
    """
    import os

    import joblib
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import (accuracy_score, precision_recall_curve,
                                 roc_auc_score)
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         train_test_split)
    from sklearn.preprocessing import LabelEncoder

    # Load the training census dataset
    with open(dataset.path, "r") as train_data:
        raw_data = pd.read_csv(train_data)

    CATEGORICAL_COLUMNS = (

    )
    LABEL_COLUMN = "refinance"
    POSITIVE_VALUE = 1

    # Convert data in categorical columns to numerical values
    encoders = {col: LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        raw_data[col] = encoders[col].fit_transform(raw_data[col])

    X = raw_data.drop([LABEL_COLUMN], axis=1).values
    y = raw_data[LABEL_COLUMN] == POSITIVE_VALUE

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    _ = xgb.DMatrix(X_train, label=y_train)
    _ = xgb.DMatrix(X_test, label=y_test)

    params = {
        "reg_lambda": [0, 1],
        "gamma": [1, 1.5, 2, 2.5, 3],
        "max_depth": [2, 3, 4, 5, 10, 20],
        "learning_rate": [0.1, 0.01],
    }

    xgb_model = xgb.XGBClassifier(
        n_estimators=50,
        objective="binary:hinge",
        silent=True,
        nthread=1,
        eval_metric="auc",
    )

    folds = 5
    param_comb = 20

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=params,
        n_iter=param_comb,
        scoring="precision",
        n_jobs=4,
        cv=skf.split(X_train, y_train),
        verbose=4,
        random_state=42,
    )

    random_search.fit(X_train, y_train)
    xgb_model_best = random_search.best_estimator_
    predictions = xgb_model_best.predict(X_test)
    score = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    _ = precision_recall_curve(y_test, predictions)

    metrics.log_metric("accuracy", (score * 100.0))
    metrics.log_metric("framework", "xgboost")
    metrics.log_metric("dataset_size", len(raw_data))
    metrics.log_metric("AUC", auc)

    # Export the model to a file
    os.makedirs(model.path, exist_ok=True)
    joblib.dump(xgb_model_best, os.path.join(model.path, "model.joblib"))
    print("Model saved to {}".format(os.path.join(model.path, "model.joblib")))
    
    deploy_model_op(os.path.join(model.path, "model.joblib"), project_id=project_id, region=region)
    
def deploy_model(
    model_directory: str,
    project_id: str,
    region: str
):
    """Custom component that uploads a saved model from GCS to Vertex Model Registry
       and deploys the model to an endpoint for online prediction.

    Args:
        model_directory: GS location of saved model.
        project_id: Project_id.
        region: Region.
    """
    import pprint as pp
    import random

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    # Check if model exists
    models = aiplatform.Model.list()
    model_name = 'xgboost-classification-model'
    if 'xgboost-classification-model' in (m.name for m in models):
        parent_model = model_name
        model_id = None
        is_default_version=False
        version_aliases=['experimental', 'challenger', 'custom-training', 'decision-tree']
        version_description='challenger version'
    else:
        parent_model = None
        model_id = model_name
        is_default_version=True
        version_aliases=['champion', 'custom-training', 'decision-tree']
        version_description='first version'

    serving_container = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest'
    uploaded_model = aiplatform.Model.upload(
        artifact_uri=model_directory,
        model_id=model_id,
        display_name=model_name,
        parent_model=parent_model,
        is_default_version=is_default_version,
        version_aliases=version_aliases,
        version_description=version_description,
        serving_container_image_uri=serving_container,
        serving_container_ports=[8080],
        labels={'created_by': 'qarik-team'},
    )

    endpoint = uploaded_model.deploy(
        machine_type='n1-standard-4',
        deployed_model_display_name='deployed-classification-model')

# 

def main():
    """Main executor."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor.Executor(
        executor_input=executor_input,
        function_to_execute=function_to_execute).execute()

if __name__ == '__main__':
    main()
