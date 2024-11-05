
"""Kubeflow Pipeline Definition"""

import argparse
from typing import *
import os

from google.cloud import storage
import kfp
from kfp import compiler, dsl
from kfp.dsl import *
import yaml

def upload_pipeline_spec(gs_pipeline_job_spec_path: str,
                         pipeline_job_spec_path: str,
                         storage_bucket_name: str):
    '''Upload pipeline job spec from local to GCS'''
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(storage_bucket_name)
    filename = '/'.join(gs_pipeline_job_spec_path.split('/')[3:])
    blob = bucket.blob(filename)
    blob.upload_from_filename(pipeline_job_spec_path)

def load_custom_component(component_name: str):
    component_path = os.path.join('components',
                                component_name,
                              'component.yaml')
    return kfp.components.load_component_from_file(component_path)

def create_training_pipeline(pipeline_job_spec_path: str):
    create_bq_view = load_custom_component(component_name='create_bq_view')
    xgboost_training = load_custom_component(component_name='xgboost_training')
    deploy_xgboost_model = load_custom_component(component_name='deploy_xgboost_model')
    export_dataset = load_custom_component(component_name='export_dataset')

    @dsl.pipeline(
        name='classification-demo-pipeline',
        description='Classification demo pipeline',
    )
    def pipeline(
        project_id: str,
        dataset_id: str,
        view_name: str,
        table_name: str,
        location: str,
        pipeline_root: str,
    ):
        """A demo pipeline."""

        create_input_view_task = create_bq_view(
            project_id=project_id,
            dataset_id=dataset_id,
            view_name=view_name,
            table_name=table_name,
        )

        export_dataset_task = (
            export_dataset(
                project_id=project_id,
                dataset_id=dataset_id,
                view_name=view_name,
            )
            .after(create_input_view_task)
            .set_caching_options(False)
        )

        training_task = xgboost_training(
            dataset=export_dataset_task.outputs["dataset"],
        )

        _ = deploy_xgboost_model(
            project_id=config.PROJECT_ID,
            model=training_task.outputs["model"],
        )

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_job_spec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       help='The config file for setting default values.')

    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    create_training_pipeline(
        pipeline_job_spec_path=config['pipelines']['pipeline_job_spec_path'])

    upload_pipeline_spec(
        gs_pipeline_job_spec_path=config['pipelines']['gs_pipeline_job_spec_path'],
        pipeline_job_spec_path=config['pipelines']['pipeline_job_spec_path'],
        storage_bucket_name=config['gcp']['storage_bucket_name'])
