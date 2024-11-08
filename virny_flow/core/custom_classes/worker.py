import math
import time
import requests

from openbox.utils.history import Observation
from virny_flow.core.utils.custom_logger import get_logger
from virny_flow.core.utils.pipeline_utils import observation_to_dict


class Worker:
    def __init__(self, address: str):
        self.address = address.rstrip('/')
        self._logger = get_logger(logger_name="worker")

    def request_with_retries(self, url, params, header, request_type):
        """
        Make a request call with retries to avoid network errors
        """
        exponent = math.exp(1)
        sleep_time = exponent  # in seconds
        response = None
        n_retries = 3
        for n_retry in range(n_retries):
            try:
                response = requests.post(url=url, json=params, headers=header) if request_type == 'POST' \
                    else requests.get(url=url, params=params, headers=header)
                if response.status_code == 400:
                    self._logger.error(f'Function request_with_retries(), number of request {n_retry + 1}, '
                                       f'exception: 400 Bad client request;\nMessage: {response.json()["detail"]}\n\n')
                    break

                response.raise_for_status()
                break
            except Exception as err:
                self._logger.error(f'Function request_with_retries(), number of request {n_retry + 1}, exception: {err}')
                if n_retry != n_retries - 1:
                    time.sleep(sleep_time)
                    self._logger.info(f'n_retry -- {n_retry}, sleep_time -- {sleep_time}')
                    sleep_time *= exponent

        return response

    def get_task(self, exp_config_name: str):
        params = {
            "exp_config_name": exp_config_name
        }
        response = self.request_with_retries(url=f'{self.address}/get_worker_task',
                                             params=params,
                                             header=None,
                                             request_type='GET')

        # Check the status code of the response
        if response.status_code == 200:
            # Parse the response content (assuming it's JSON)
            task = response.json()
            self._logger.info(f"New task with UUID {task['task_uuid']} was taken")
            return task
        else:
            self._logger.info(f"Failed to retrieve data. Status code: {response.status_code}.")
            return None

    def complete_task(self, exp_config_name: str, task_uuid: str, physical_pipeline_uuid: str,
                      logical_pipeline_uuid: str, logical_pipeline_name: str, observation: Observation):
        params = {
            "exp_config_name": exp_config_name,
            "task_uuid": task_uuid,
            "physical_pipeline_uuid": physical_pipeline_uuid,
            "logical_pipeline_uuid": logical_pipeline_uuid,
            "logical_pipeline_name": logical_pipeline_name,
            "observation": observation_to_dict(observation),
        }
        response = self.request_with_retries(url=f'{self.address}/complete_worker_task',
                                             params=params,
                                             header=None,
                                             request_type='POST')

        # Check the status code of the response
        if response.status_code == 200:
            # Parse the response content (assuming it's JSON)
            data = response.json()
            self._logger.info(f"Response: {data['message']}")
        else:
            self._logger.info(f"Failed to retrieve data. Status code: {response.status_code}.")
