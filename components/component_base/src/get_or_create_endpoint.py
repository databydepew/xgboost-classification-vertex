import kfp
from kfp import dsl
from kfp.dsl import executor
from typing import *
from google.cloud import aiplatform
import argparse
import json

def get_or_create_endpoint(project_id: str, 
                           region: str, 
                           endpoint_id: str, 
                           display_name: str) -> str:
    from google.cloud import aiplatform
    """
    Get or create an endpoint in Vertex AI.

    Args:
        project_id (str): The GCP project ID.
        region (str): The GCP region.
        endpoint_id (str): The ID of the endpoint.
        display_name (str): The display name of the endpoint.

    Returns:
        str: The resource name of the endpoint.
    """
    aiplatform.init(project=project_id, location=region)

    try:
        # Try to get the existing endpoint
        endpoint = aiplatform.Endpoint(endpoint_id)
        endpoint_resource_name = endpoint.resource_name
        print(f"Endpoint '{endpoint_id}' already exists.")
    except Exception as e:
        # If the endpoint does not exist, create a new one
        print(f"Endpoint '{endpoint_id}' does not exist. Creating a new one.")
        endpoint = aiplatform.Endpoint.create(endpoint_id=endpoint_id, display_name=display_name)
        endpoint_resource_name = endpoint.resource_name
        print(f"Endpoint '{endpoint_id}' created successfully.")

    return endpoint_resource_name


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

