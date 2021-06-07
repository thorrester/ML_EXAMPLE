import os
from threading import Lock

import pandas as pd
from google.cloud import bigquery, bigquery_storage_v1beta1

lock = Lock()  # Lock for instantiating BigQuery client

_client = {}  # holds singleton instance of BigQuery client

storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()

def get_client() -> bigquery.Client:
    """Return singleton instance of BigQuery client

    The project ID is read from environment variable `GOOGLE_BQ_PROJECT_ID`.
    If not set, falls back to the default inferred from the environment.

    :return: Singleton instance of BigQuery client
    """
    if not _client.get('instance', None):
        with lock:
            if not _client.get('instance', None):
                _client['instance'] = bigquery.Client(project=os.environ.get('GOOGLE_BIGQUERY_PROJECT_ID', 'analytics-store-ops-thd'))

    return _client['instance']


def query(query_sql: str) -> pd.DataFrame:
    """Executes `query_sql` using the default BigQuery client and returns the result as a DataFrame

    :param query_sql: SQL query to execute
    :type query_sql: str

    :return: Query result as pandas DataFrame
    """
    return get_client().query(query_sql).to_dataframe(bqstorage_client=storage_client)
