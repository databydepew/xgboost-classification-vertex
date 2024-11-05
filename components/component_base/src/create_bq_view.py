
import argparse
import json
from kfp.dsl import executor

import kfp
from kfp import dsl
from kfp.dsl import *
from typing import *

def create_bq_view(
    project_id: str,
    dataset_id: str,
    view_name: str,
    table_name: str,
):
    """Creates a BigQuery view on a table.

    Args:
        project_id: The Project ID.
        dataset_id: The BigQuery Dataset ID. Must be pre-created in the project.
        view_name: The BigQuery view name.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)

    create_or_replace_view = """
        CREATE OR REPLACE VIEW
        `{dataset_id}.{view_name}` AS
        SELECT
        interest_rate,
        loan_amount,
        loan_balance,
        loan_to_value_ratio,
        credit_score,
        debt_to_income_ratio,
        income,
        loan_term,
        loan_age,
        home_value,
        current_rate,
        rate_spread,
        refinance
        FROM
        `{project_id}.{dataset_id}.{table_name}`
    """.format(
        project_id=project_id, dataset_id=dataset_id, view_name=view_name, table_name=table_name
    )


    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query=create_or_replace_view, job_config=job_config)
    query_job.result()

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
