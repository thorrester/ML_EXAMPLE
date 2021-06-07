import os
from threading import Lock

from google.cloud import storage

lock = Lock()  # Lock for instantiating Storage client

_client = {}  # holds singleton instance of Storage client


def get_client() -> storage.Client:
    """Return singleton instance of Google Storage client

    The project ID is read from environment variable `GOOGLE_PROJECT_ID`.
    If not set, falls back to the default inferred from the environment.

    :return: Singleton instance of Storage client
    """
    if not _client.get('instance', None):
        with lock:
            if not _client.get('instance', None):
                _client['instance'] = storage.Client(project=os.environ.get('GOOGLE_PROJECT_ID', None))

    return _client['instance']