"""Alpine Meadow API client."""

import requests


class APIClient:
    """
    The API client is a helper class for posting metadata to the API server.
    """

    def __init__(self, host='localhost', port=5000):
        self._end_point = f'http://{host}:{port}'

    def register_task(self, task):
        response = requests.post(f'{self._end_point}/task',
                                 json={'task': task.dumps()}, timeout=60)
        return response.status_code

    def register_pipelines(self, task_id, pipelines):
        json_pipelines = list(map(lambda pipeline: pipeline.dumps(), pipelines))
        response = requests.post(f'{self._end_point}/pipelines',
                                 json={'task_id': task_id,
                                       'pipelines': json_pipelines}, timeout=60)
        return response.status_code
