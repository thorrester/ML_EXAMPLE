import re
from threading import Lock, Condition
from typing import List, Dict, Iterable, Callable
from datetime import datetime, timedelta
import time
from concurrent import futures

from google.cloud import bigquery

from . import bq_client


def parallel_execute_query(queries: object,
                           *,
                           max_workers: object = 64) -> object:
    """Executes query in parallel.

    Writes the output to `destination_table`. Overwrites `destination_table` if already exists.

    :type destination_table: str
    :param destination_table: Fully qualified BigQuery table id to write the query result.

    :type queries: Dict[str, str]
    :param queries: A dictionary containing partition_id as the key, and the query to execute as the value.

    :type destination_partition_field: str
    :param destination_partition_field: Optional. If supplied, the `destination_table` is
                                        partitioned (by DAY) using this field. This field must exists
                                        in the `destination_table` table.

    :type clustering_fields: List[str]
    :param clustering_fields: Optional. If supplied, the `destination_table` is clustered using this list.
                              Each item in the list must exist in the `destination_table` table.

    :type max_workers: int
    :param max_workers: Maximum number of concurrent job.

    :return: None
    :rtype: None
    """
    func_args = [
        {
            "table": table,
            "query": query,
        } for table, query in queries.items()
    ]

    with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        #execute_func = limit_execution_rate(execute_query, max_executions=45, time_period_seconds=15)
        jobs = {pool.submit(execute_query, **args): args for args in func_args}

        for job in futures.as_completed(jobs):
            try:
                job.result()
            except Exception as e:
                print(f"Error processing query: {e}")
                _cancel_jobs(jobs)
                pool.shutdown(wait=True)
                raise

    print(f"Finished stage01")


def _cancel_jobs(jobs: Iterable[futures.Future]):
    for job in jobs:
        job.cancel()


def execute_query(table: str,
                  query: str):
    """Executes a query and writes the result to `destination_table`.

    :type table: str
    :param partition_id: Identifier for this query job.

    :type query: str
    :param query: Query to execute.

    :return: None
    :rtype: None
    """

    start_time = time.perf_counter()
    job = bq_client.get_client().query(query)
    print(f"Executing query for table: '{table}'. Job ID: {job.job_id}")
    job.result()
    print(f"Job done for table '{table}'. Job ID: {job.job_id}. "
          f"Duration: {timedelta(seconds=time.perf_counter() - start_time)}")


def limit_execution_rate(func: Callable,
                         max_executions: int = 45,
                         time_period_seconds: float = 15.0,
                         delay_seconds: float = 1.0):
    """Limits the concurrent executions within a time period

    BigQuery has maximum rate limit of maximum 50 partition operations every 10 seconds.
    See https://cloud.google.com/bigquery/quotas

    This function ensures that there are no more than `max_executions` running concurrently
    within `time_period_seconds`.

    The execution of the function is delayed when the limit is reached.

    :type func: Callable
    :param func: Function to execute.

    :type max_executions: int
    :param max_executions: Maximum concurrent executions.

    :type time_period_seconds: float
    :param time_period_seconds: Time period for the rate limit, in seconds.

    :type delay_seconds: float
    :param delay_seconds: Time delay when rate limit is reached, in seconds.
    """

    lock = Lock()
    waitable_lock = Condition()
    executions = {
        'active_count': 0,
        'recent': []
    }

    # Discard executions that are older than `time_period_seconds`
    def _refresh_ops():
        with lock:
            time_now = time.perf_counter()
            executions['recent'] = [e for e in executions['recent'] if time_now - e <= time_period_seconds]
            recent_ops_count = len(executions['recent'])
        return recent_ops_count

    def _execute(*args, **kwargs):
        # Check if maximum concurrent jobs is reached
        with waitable_lock:
            while executions['active_count'] >= max_executions:
                print(f"Maximum {max_executions} active executions reached. Waiting for older jobs to finish.")
                waitable_lock.wait()
            executions['active_count'] = executions['active_count'] + 1

        # Check if maximum recent updates is reached
        while _refresh_ops() >= max_executions:
            print(f"Maximum {max_executions} executions reached in the last {time_period_seconds} seconds."
                  f" Sleeping for {delay_seconds} seconds.")
            time.sleep(delay_seconds)

        # Execute function
        error = None
        ret_val = None
        try:
            ret_val = func(*args, **kwargs)
        except Exception as e:
            error = e

        # Log recent job completion
        with lock:
            executions['recent'].append(time.perf_counter())

        # Free up active_count slot
        with waitable_lock:
            executions['active_count'] = executions['active_count'] - 1
            waitable_lock.notify()

        if error:
            raise error

        return ret_val

    return _execute
