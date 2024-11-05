#!/bin/bash 

# Publishes a message to a Pub/Sub topic to invoke the
# pipeline job submission service.
# This script should run from the AutoMLOps/ directory
# Change directory in case this is not the script root.

gcloud pubsub topics publish xgboost-classification-queueing-svc --message "$(cat pipelines/runtime_parameters/pipeline_parameter_values.json)"