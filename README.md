# XGBoost Classification on Google Vertex AI

This repository provides an automated MLOps pipeline for training and deploying an XGBoost classification model using Google Cloud's Vertex AI. This pipeline is generated through the AutoMLOps framework, which facilitates the transition from data science notebooks to production environments.

## Project Structure

The core components of the repository are organized into distinct directories, each serving a specific role:

- **components/**: Custom components for the Vertex AI pipeline.
  - **component_base/**: Contains the Dockerfile and Python scripts for component implementation.
  - **<component_name>/**: Contains YAML specifications for each component.

- **pipelines/**: Vertex AI pipeline definitions and scripts.
  - **pipeline.py**: Main script that compiles and uploads the pipeline specifications.
  - **pipeline_runner.py**: Script for submitting the pipeline to Vertex AI.

- **services/submission_service/**: Contains the REST API service to submit pipeline jobs to Vertex AI.
  - **main.py**: Source code for deploying the REST API service.

- **scripts/**: Utility scripts for building, running, and managing the pipeline.
  - **build_components.sh**: Builds Docker images for components and pushes them to the registry.
  - **run_pipeline.sh**: Submits the pipeline job to Vertex AI for execution.
  - **run_all.sh**: Executes a sequence of building, compiling, and running the pipeline.
  - **publish_to_topic.sh**: Publishes messages to a Pub/Sub topic, triggering pipeline submission.

- **configs/**: Configuration files for customizing the pipeline environment.
  - **defaults.yaml**: Contains the default runtime configurations necessary for the pipeline execution.

- **provision/**: Contains provisioning scripts for setting up necessary infrastructure.
  - **provision_resources.sh**: Sets up Google Cloud infrastructure such as service accounts and APIs.

## Workflow

1. **Component Development**: Under `components/component_base/`, develop individual components in Python. Compose Dockerfiles to containerize these components.

2. **Component Building**: Use `scripts/build_components.sh` to package and push these components to the Artifact Registry.

3. **Pipeline Definition**: Define the pipeline structure using `pipelines/pipeline.py`, which arranges the flow of data and tasks.

4. **Pipeline Execution**: Submit and run your pipeline on Vertex AI using `scripts/run_pipeline.sh` or combine steps with `scripts/run_all.sh`.

5. **Job Submission Service**: The service within `services/submission_service/main.py` provides a REST API to automate submission of jobs to Vertex AI, allowing for remote execution.

6. **Configuration Management**: Utilize `configs/defaults.yaml` to manage and adapt configurations, ensuring pipeline compatibility with the Google Cloud environment.

## Automated MLOps with AutoMLOps

The AutoMLOps framework accelerates MLOps automation by:

- Generating a production-ready codebase, from Jupyter Notebooks.
- Establishing CICD pipelines to handle pipeline versioning, promotion, and deployment.
- Supporting recurring job scheduling using Cloud Scheduler.

## Prerequisites

Ensure you have enabled necessary Google Cloud services (Vertex AI, Cloud Build, Pub/Sub) and set up your environment with necessary permissions.

Visit [these slides](https://github.com/GoogleCloudPlatform/automlops/blob/main/AutoMLOps_Implementation_Guide_External.pdf) for in-depth guidance.

## Conclusion

This setup provides a robust platform for deploying XGBoost models in a scalable and reproducible manner using Google Cloud's Vertex AI, streamlining the MLOps lifecycle from research to production.