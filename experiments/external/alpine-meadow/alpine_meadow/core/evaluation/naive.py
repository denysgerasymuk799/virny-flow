"""Naive evaluation method."""

from alpine_meadow.utils.performance import time_calls
from .base import Evaluation


class NaiveEvaluation(Evaluation):
    """
    The naive evaluation method which simply trains and test the pipeline
    """

    @time_calls
    def validate_pipeline(self, pipeline):
        import time

        task = self._optimizer.task

        if self.is_cross_validation_enabled():
            yield self.evaluate_pipeline_with_cross_validation(pipeline)
            return

        pipeline_executor = self._optimizer.backend.get_pipeline_executor(pipeline, task.metrics)
        pipeline_history = pipeline.pipeline_arm.pipeline_history

        # train
        train_start = time.perf_counter()
        pipeline_executor = self.train_pipeline(pipeline_executor, self._train_dataset)
        train_time = time.perf_counter() - train_start

        # test
        validation_start = time.perf_counter()
        pipeline_executor = self.test_pipeline(pipeline_executor, [(self._validation_dataset, self._validation_target)])
        validation_time = time.perf_counter() - validation_start
        validation_score = pipeline_executor.get_scores(to_utility=True)[-1]
        # validation_error = pipeline_executor.get_scores(to_error=True)[-1]

        # save metrics
        self._optimizer.state.add_score(validation_score)
        self._optimizer.state.add_time(train_time + validation_time)
        pipeline.metrics.score = validation_score
        pipeline.metrics.time = train_time + validation_time
        pipeline_history.update_run_history(pipeline.id)

        self._optimizer.metrics['validated_pipelines_num'] += 1

        # update validation method
        pipeline_executor.validation_method = self._validation_method

        yield pipeline_executor
