"""Meta-learning manager handles the read/write of meta-learning task traces and applies meta-learning."""
import os
import json
import pickle
import tempfile
import time
from operator import itemgetter

import numpy as np
from smac.configspace import Configuration
from smac.runhistory.runhistory import DataOrigin
from smac.tae import StatusType

from alpine_meadow.common import TaskKeyword
from alpine_meadow.utils import ignore_warnings
from alpine_meadow.utils.performance import time_calls


def get_run_history_from_similar_tasks(similar_tasks, key, external_run_history,
                                       configuration_space, logger=None):
    """
    Load run history from similar datasets as external run history
    """

    for task_trace in similar_tasks.values():
        run_history_json = task_trace.run_history.get(key, None)
        # logger.info('Json: {}'.format(run_history_json))
        if run_history_json is not None:
            try:
                ids_config = {int(id_): Configuration(configuration_space, values=values)
                              for id_, values in run_history_json["configs"].items()}

                # important to use add method to use all data structure correctly
                for k, v in run_history_json["data"]:
                    external_run_history.add(
                        config=ids_config[int(k[0])],
                        cost=float(v[0]),
                        time=float(v[1]),
                        status=StatusType(v[2]),
                        instance_id=k[1],
                        seed=int(k[2]),
                        additional_info=v[3],
                        origin=DataOrigin.EXTERNAL_SAME_INSTANCES)
            except BaseException as e:  # pylint: disable=broad-except
                exp_str = f'{e}'
                if 'Trying to set illegal value' in exp_str or 'Active hyperparameter' in exp_str:
                    logger.debug(msg='', exc_info=True)
                else:
                    logger.error(msg='', exc_info=True)

    return external_run_history


class RenameUnpickler(pickle.Unpickler):
    """Pickler implementation to support old SMAC packages."""

    def find_class(self, module, name):
        renamed_module = module
        if module == "smac.tae.execute_ta_run":
            renamed_module = "smac.tae"

        return super().find_class(renamed_module, name)


class MetaLearningManager:
    """
    The class for managing everything about the pipeline traces of datasets
    """

    def __init__(self, config):
        self._config = config
        self._base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "files")
        self._task_traces = {}
        self._loaded = False
        self._registered_tasks = {}

    @property
    def num_task_traces(self):
        return len(self._task_traces)

    @property
    def num_pipeline_runs(self):
        num_pipeline_runs = 0
        for task_trace in self._task_traces.values():
            num_pipeline_runs += task_trace.num_pipeline_runs
        return num_pipeline_runs

    def get_num_similar_tasks(self, task):
        return len(self._registered_tasks.get(task.id, []))

    def get_num_reference_pipelines(self, task):
        num_reference_pipelines = 0
        for similar_task_name in self._registered_tasks.get(task.id, []):
            similar_task = self._task_traces[similar_task_name]
            num_reference_pipelines += similar_task.num_pipeline_runs
        return num_reference_pipelines

    @ignore_warnings
    @time_calls
    def register_task(self, task, config):
        """
        Find its similar datasets and put them in the internal map
        """

        if task.id in self._registered_tasks:
            return self._registered_tasks[task.id]

        if not self._loaded:
            self._load()

        start = time.perf_counter()

        meta_features = task.meta_features
        if meta_features is None:
            return None

        # get regressor
        task_type = task.type
        regressor = self._regressors.get(task_type, None)
        if regressor is None:
            return None

        # find similar tasks within similarity threshold
        similar_tasks = []
        feature_vector = None
        applicable_task_names = []
        for task_name, task_trace in self._task_traces.items():
            if task_trace.task_type != task_type:
                continue

            if feature_vector is None:
                feature_vector = task_trace.get_feature_vector(meta_features)
            else:
                feature_vector = np.concatenate((feature_vector, task_trace.get_feature_vector(meta_features)))
            applicable_task_names.append(task_name)

        similarities = regressor.predict(feature_vector).flatten()
        for i, task_name in enumerate(applicable_task_names):
            similarity = similarities[i]
            if similarity >= config.meta_learning_similarity_threshold:
                config.logger.debug(f'Similar task: {task_name}, {similarity}')
                similar_tasks.append((similarity, task_name))

        # find most similar datasets
        similar_tasks = sorted(similar_tasks, key=itemgetter(0))[:config.meta_learning_similar_datasets_num]
        similar_tasks = list(map(lambda x: x[1], similar_tasks))

        # save
        self._registered_tasks[task.id] = similar_tasks

        task.meta_learning_info['num_similar_tasks'] = len(similar_tasks)
        config.logger.debug(f'Find {len(similar_tasks)} similar tasks for '
                            f'meta-learning in {time.perf_counter() - start} seconds')

        return similar_tasks

    @ignore_warnings
    @time_calls
    def update_pipeline_arms_from_history(self, task, pipeline_arms):
        """
        Find all run history of pipelines from similar datasets, we calculate the history quality of each pipeline arm
        based on the history, and also load them into the external run history for possible hyper-parameter tuning.
        """

        start = time.perf_counter()

        # import concurrent.futures

        # process_pool = concurrent.futures.ProcessPoolExecutor()
        # unique_pipeline_arms = {}
        # for pipeline_arm in pipeline_arms:
        #    key = frozenset(pipeline_arm.get_unique_tunable_primitives_strs())
        #    unique_pipeline_arms[key] = pipeline_arm

        similar_tasks = {}
        for similar_task in self._registered_tasks.get(task.id, []):
            similar_tasks[similar_task] = self._task_traces[similar_task]
        external_values = {}
        updated_runs_count = 0
        for pipeline_arm in pipeline_arms:
            key = frozenset(pipeline_arm.get_unique_tunable_primitives_strs())
            if key in external_values:
                pipeline_arm.external_run_history = external_values[key]
            else:
                get_run_history_from_similar_tasks(
                    similar_tasks, key, pipeline_arm.external_run_history,
                    pipeline_arm.configuration_space,
                    logger=self._config.logger)
                external_values[key] = pipeline_arm.external_run_history

            updated_runs_count += len(pipeline_arm.external_run_history.data)
            cost_per_config = pipeline_arm.external_run_history._cost_per_config  # pylint: disable=protected-access
            costs = list(cost_per_config.values())
            if costs:
                # self._config.logger.info('Costs: {}'.format(costs))
                # execution_time = max(map(lambda x: x.time, pipeline_arm.external_run_history.data.values()))
                pipeline_arm.external_quality_mean = -np.mean(costs)
                pipeline_arm.external_quality_std = np.std(costs)

        task.meta_learning_info['num_updated_runs'] = updated_runs_count
        self._config.logger.debug(f'Updated {updated_runs_count} runs for '
                                  f'meta-learning time: {time.perf_counter() - start}')

    @ignore_warnings
    @time_calls
    def find_starting_pipelines(self, pipeline_arms, pipelines_num):
        """
        From similar datasets, we find the overall best k pipelines
        """

        all_configurations = []
        unique_pipeline_arms = set()
        for pipeline_arm in pipeline_arms:
            key = frozenset(pipeline_arm.get_unique_tunable_primitives_strs())
            if key in unique_pipeline_arms:
                continue
            unique_pipeline_arms.add(key)

            for configuration in pipeline_arm.external_run_history.config_ids.keys():
                cost = pipeline_arm.external_run_history.get_cost(configuration)
                all_configurations.append((cost, configuration, pipeline_arm))
        good_configurations = sorted(all_configurations, key=lambda x: x[0])[:pipelines_num]

        starting_pipelines = []
        for cost, configuration, pipeline_arm in good_configurations:
            pipeline = pipeline_arm.get_pipeline_from_configuration(configuration)
            pipeline.tags['pipeline_arm'] = 'meta-learning'
            pipeline.tags['hyperparameters'] = 'meta-learning'
            self._config.logger.debug(f'Starting pipeline: {cost}, {pipeline.to_pipeline_desc(human_readable=True)}')
            starting_pipelines.append(pipeline)

        return starting_pipelines

    @time_calls
    def dump_run_history(self, pipeline_arms, output_path):
        """
        Dump all the run history of given pipeline arms
        """

        run_history_jsons = []
        dumped_runs_count = 0
        for pipeline_arm in pipeline_arms:
            run_history = pipeline_arm.run_history
            dumped_runs_count += len(run_history.data)
            with tempfile.NamedTemporaryFile(delete=False) as run_history_file:
                run_history.save_json(fn=run_history_file.name, save_external=True)

            with open(run_history_file.name, 'r') as file:
                run_history_json = json.load(file)
            run_history_json['pipeline_arm'] = pipeline_arm.get_unique_tunable_primitives_strs()
            run_history_jsons.append(run_history_json)

        with open(output_path, 'w') as file:
            json.dump(run_history_jsons, file)
        self._config.logger.debug(f'Dumped {dumped_runs_count} runs for meta-learning')

    @ignore_warnings
    def _load(self):
        """
        Load the traces of all processed tasks
        """

        load_start = time.perf_counter()

        if self._config.enable_meta_learning:
            self._config.logger.info('Meta-learning is enabled')

        # load datasets
        if self._config.enable_meta_learning:
            with open(os.path.join(self._base_path, 'datasets.pkl'), 'rb') as f:
                self._task_traces = RenameUnpickler(f).load()
        self._config.logger.debug(f'Loaded {len(self._task_traces)} task traces')

        # load regressors
        if self._config.enable_meta_learning:
            try:
                with open(os.path.join(self._base_path, 'regressors.pkl'), 'rb') as f:
                    self._regressors = pickle.load(f)
            except:  # pylint: disable=bare-except  # noqa: E722
                self._regressors = {}
                self._config.logger.error(msg='Cannot load mete-learning regressors!', exc_info=True)
        else:
            self._regressors = {}
        self._config.logger.debug(f'Loaded meta-learning regressors for: '
                                  f'{list(map(TaskKeyword.Name, self._regressors.keys()))}')
        self._config.logger.debug(f'Loading meta-learning traces time: {time.perf_counter() - load_start}')

        self._loaded = True
