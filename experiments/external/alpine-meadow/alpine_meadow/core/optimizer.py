# pylint: disable=consider-using-with
"""Alpine Meadow optimizer implementation."""

from alpine_meadow.api import APIClient
from alpine_meadow.common import Task, Config
from alpine_meadow.backend import BaseBackend
from alpine_meadow.utils.performance import time_calls, dump_metrics
from alpine_meadow.utils import AMException
from .rule import RuleExecutor
from .cost import CostModel
from .meta_learning import MetaLearningManager, PipelineHistory
from .base import OptimizerResult, OptimizerState


class Optimizer:
    """
    The optimizer takes into a task description (and data) and produces as much best pipeline as
    possible under the user-provide metric within the budget.
    It first uses the `RuleExecutor` to apply rules and create the search space.
    Then it creates some generator workers to create new pipelines out of the search space, and some execution
    workers to execute these candidate pipelines.
    The evaluated pipelines will be compared against the current best one and sent back to the user if it is better.
    """

    @time_calls
    def __init__(self, task: Task, config: Config = None, backend: BaseBackend = None):
        # initialize state
        self.state = OptimizerState(self)

        # initialize config if needed
        if config is None:
            config = Config()
        self.config = config
        self.logger = self.config.logger

        # initialize task
        self.task = task
        self.task.config = config

        # post api server
        if config.enable_api_client:
            self.api_client = APIClient()
            self.api_client.register_task(task)

        # initialize backend if needed
        if backend is None:
            from alpine_meadow.backend import DefaultBackend
            backend = DefaultBackend(self.config)
            self.config.logger.debug(f"Using default backend with {backend.get_num_workers()} workers")
        self.backend = backend

        # metrics and history
        from collections import defaultdict

        self.metrics = defaultdict(float)
        self.meta_learning_manager = MetaLearningManager(config)
        self.pipeline_history = PipelineHistory()

        # build search space
        self.cost_model = CostModel(self.config, self.state)
        self.rule_executor = RuleExecutor(self.config)
        self.search_space = self.rule_executor.execute(self.task, self.pipeline_history)
        if not self.search_space.pipeline_arms:
            raise NotImplementedError("Search space is empty!")

        # apply meta learning
        if self.config.enable_learn_from_history:
            # find similar datasets for meta learning
            self.meta_learning_manager.register_task(task, self.config)

            # update pipeline arms
            pipeline_arms = self.search_space.pipeline_arms
            self.meta_learning_manager.update_pipeline_arms_from_history(task, pipeline_arms)

        # initialize evaluation method
        if self.config.evaluation_method == 'adaptive' and task.dataset.num_instances <= 10:
            self.logger.warning(f"The dataset is too small with {task.dataset.num_instances} rows for "
                                f"running adaptive, falling back to naive")
            self.config.evaluation_method = 'naive'

        if self.config.evaluation_method == 'adaptive':
            from .evaluation.adaptive import AdaptiveEvaluation

            self.evaluation_method = AdaptiveEvaluation()
        elif self.config.evaluation_method == 'naive':
            from .evaluation.naive import NaiveEvaluation

            self.evaluation_method = NaiveEvaluation()
        else:
            raise AMException(f"Unknown evaluation method: {self.config.evaluation_method}")
        self.evaluation_method.init(self, self.task.dataset)

        self.logger.info(f'{self.task}')
        self.logger.debug(f'Initializing optimizer time: {self.state.get_elapsed_time()}')

    def update_task(self, dataset=None, train_dataset=None, validation_dataset=None):
        if dataset is not None:
            self.task.dataset = dataset
        if train_dataset is not None:
            self.task.train_dataset = train_dataset
            self.task.validation_dataset = validation_dataset

        self.evaluation_method.init(self, self.task.dataset)

    def optimize(self, time_limit: float = None, pipelines_num_limit: int = None):
        """
        Starts searching for pipelines and returns the pipelines as a generator
        :param time_limit: time out in seconds
        :param pipelines_num_limit: the maximum number of pipelines can be evaluated
        :return: a generator of pipelines
        """

        self.state.start()

        # budget
        if time_limit is None and pipelines_num_limit is None:
            time_limit = self.config.timeout_seconds
        if time_limit is not None:
            self.logger.info(f'Time limit: {time_limit}')
            self.state.time_limit = time_limit
        else:
            if pipelines_num_limit is None:
                raise AMException("No limit for time and pipeline nums!")
            self.logger.info(f'Pipelines num limit: {pipelines_num_limit}')
            self.state.pipelines_num_limit = pipelines_num_limit

        # apply automatic feature engineering
        if self.config.enable_feature_engineering:
            from .optimization import automatic_feature_engineering

            automatic_feature_engineering(self, self.task.dataset)

        # start optimizing
        best_pipeline = None
        for best_pipeline in self._run():
            if self.config.compute_all_metrics:
                self.evaluation_method.compute_all_metrics(best_pipeline)

            # compute elapsed time, progress and score
            elapsed_time = self.state.get_elapsed_time()
            progress = self.state.progress()
            score = best_pipeline.get_score()

            # return result
            self.logger.info(f"Validated Pipelines Num: {int(self.metrics['validated_pipelines_num'])}")
            self.logger.info(f'Progress {progress}, Time: {self.state.get_elapsed_time()}, Score: {score}, '
                             f'Pipeline: {best_pipeline.pipeline.pipeline_arm.get_unique_primitives_strs()[-1]}')
            yield OptimizerResult(pipeline=best_pipeline, progress=progress,
                                  elapsed_time=elapsed_time, score=score)

        # apply ensembling
        if self.config.enable_ensembling:
            try:
                from .optimization import ensembling

                ensemble_pipeline = ensembling(self)
                self.logger.info(f'Ensembling score: {ensemble_pipeline.get_score()}')
                best_pipeline = ensemble_pipeline
            except BaseException:  # pylint: disable=broad-except
                self.logger.debug(msg='', exc_info=True)

        # show best pipeline and metrics
        if not self.state.find_first_pipeline:
            raise AMException("Failed to find a pipeline with the limits!")
        self.show_metrics()

        # show total elapsed time
        self.logger.info(f'Optimizing time: {self.state.get_elapsed_time()}')

        # return final result
        if best_pipeline is not None:
            self.logger.debug(f'Best pipeline: {best_pipeline.pipeline.to_pipeline_desc(human_readable=True)}')
            yield OptimizerResult(pipeline=best_pipeline, progress=100.0,
                                  elapsed_time=self.state.get_elapsed_time(),
                                  score=best_pipeline.get_score())

    def stop(self):
        self.logger.info("Stopping AutoML...")
        self.state.done = True

    def show_metrics(self):
        import json

        self.logger.debug(f'Optimizer metrics: {json.dumps(self.metrics, sort_keys=True, indent=4)}')
        self.logger.debug(f'Profiling metrics: {json.dumps(dump_metrics(), sort_keys=True, indent=4)}')
        # self.cost_model.show(self.search_space.pipeline_arms)

    def dumps(self):
        """
        Dump all the traces as a dict
        :return:
        """

        task = self.task
        pipeline_history = self.pipeline_history
        dict_ = {
            'task': task.dumps(),
            'pipelines': list(map(lambda pipeline: pipeline.dumps(), pipeline_history.get_all_pipelines())),
            'pipeline_runs': list(map(lambda run: run.dumps(), pipeline_history.get_all_pipeline_runs()))
        }

        # dump run history
        if task.config.log_trace:
            import tempfile
            import json

            run_history_table = []
            dumped_runs_count = 0
            for pipeline_arm in self.search_space.pipeline_arms:
                run_history = pipeline_arm.run_history
                if not run_history.data:
                    continue
                dumped_runs_count += len(run_history.data)

                with tempfile.NamedTemporaryFile(delete=False) as run_history_file:
                    run_history.save_json(fn=run_history_file.name, save_external=True)
                with open(run_history_file.name, 'r') as file:
                    run_history_json = json.load(file)
                run_history_json['pipeline_arm'] = pipeline_arm.get_unique_tunable_primitives_strs()
                run_history_table.append(run_history_json)
            task.config.logger.debug(f'Dumped {dumped_runs_count} runs for meta-learning')

            dict_["run_history"] = run_history_table

        # dump cost moddel
        dict_['cost_model'] = self.cost_model.dumps(self)

        # dump metrics
        dict_['optimizer_metrics'] = self.metrics
        dict_['profiling_metrics'] = dump_metrics()

        return dict_

    def dump(self, path):
        import json

        with open(path, 'w') as f:
            json.dump(self.dumps(), f, indent=4)

    def _run(self):
        """
        Start running the search, and returns a generator of best pipelines.
        :return:
        """

        import time
        import queue
        from concurrent.futures import ThreadPoolExecutor

        # set up queues
        generation_threads_num = self.config.generation_threads_num
        evaluation_workers_num = self.backend.get_num_workers()
        evaluation_queue_size = max(
            self.config.starting_pipelines_num + 3,
            int(self.config.configurations_per_arm_num * evaluation_workers_num
                * self.config.evaluation_workers_reservation_rate))
        to_be_evaluated_pipelines_queue = queue.Queue(maxsize=evaluation_queue_size)
        to_be_returned_pipelines_queue = queue.Queue(maxsize=evaluation_queue_size)

        # put some good starting pipelines into the evaluation queue (with meta-learning)
        starting_pipelines = self.meta_learning_manager.find_starting_pipelines(
            self.search_space.pipeline_arms, self.config.starting_pipelines_num)
        self.logger.info(f'Meta-learning starting pipelines num: {len(starting_pipelines)}')
        for starting_pipeline in starting_pipelines:
            starting_pipeline.created_time = 0.0
            to_be_evaluated_pipelines_queue.put_nowait(starting_pipeline)

        # start generation and evaluation threads
        from .workers import generating_pipelines_worker, evaluating_pipelines_worker

        thread_pool = ThreadPoolExecutor(max_workers=generation_threads_num + evaluation_workers_num)
        for worker_id in range(generation_threads_num):
            thread_pool.submit(generating_pipelines_worker,
                               worker_id, self, to_be_evaluated_pipelines_queue)
        for worker_id in range(evaluation_workers_num):
            thread_pool.submit(evaluating_pipelines_worker,
                               worker_id, self, to_be_evaluated_pipelines_queue, to_be_returned_pipelines_queue)

        # main thread: returning best pipelines
        best_pipeline = None
        while True:
            try:
                pipeline = to_be_returned_pipelines_queue.get_nowait()

                if best_pipeline is None or pipeline.is_better(best_pipeline):
                    best_pipeline = pipeline
                    yield best_pipeline
                    self.state.find_first_pipeline = True
                else:
                    # pipeline_executor.release()
                    pass

            except queue.Empty:
                time.sleep(0.01)

            # if it is done (stopped by others), we will stop
            if self.state.done:
                self.logger.info("AutoML is stopped...")
                break

            # check if the budget limit is met, is so we exit
            remaining_time = self.state.get_remaining_time()
            if remaining_time and remaining_time <= 0:
                break
            if self.state.pipelines_num_limit:
                if int(self.metrics['validated_pipelines_num']) >= self.state.pipelines_num_limit:
                    break
        self.logger.debug("Optimization is done. Shutting down...")
        self.state.done = True

        # clean up
        while True:
            try:
                to_be_evaluated_pipelines_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                to_be_returned_pipelines_queue.get_nowait()
            except queue.Empty:
                break
        self.logger.debug('Queues have been cleaned up')
        thread_pool.shutdown()
        self.logger.debug('Thread pool have been shut down')
