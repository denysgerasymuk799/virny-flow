"""Alpine Meadow workers."""


def generating_pipelines_worker(worker_id, optimizer, output_queue):
    """
    Generate new pipelines and put them into the queue.
    :param worker_id:
    :param optimizer:
    :param output_queue:
    :return:
    """

    import time
    import queue

    optimizer.logger.debug(f'Generating worker {worker_id} starts...')

    while True:
        if optimizer.state.done:
            break

        try:
            # get pipeline arm
            pipeline_arm, tags = optimizer.search_space.get_k_pipeline_arms(optimizer.cost_model, k=1)[0]
            pipelines = pipeline_arm.get_next_pipelines(
                use_bayesian_optimization=optimizer.config.enable_bayesian_optimization,
                pipelines_num=optimizer.config.configurations_per_arm_num)

            # add time/tags to pipelines
            for pipeline in pipelines:
                pipeline.created_time = optimizer.state.get_elapsed_time()
                pipeline.tags = {**pipeline.tags, **tags}

            # register pipelines
            if optimizer.config.enable_api_client:
                optimizer.api_client.register_pipelines(optimizer.task.id,
                                                        pipelines)

            # add pipelines to the evaluation queue
            for pipeline in pipelines:
                if optimizer.state.done:
                    break

                while True:
                    if optimizer.state.done:
                        break
                    try:
                        output_queue.put_nowait(pipeline)
                        # sleep more to encourage diversity
                        time.sleep(0.01)
                        break
                    except queue.Full:
                        optimizer.metrics['generation_worker_idle_time'] += 0.01
                        time.sleep(0.01)
        except BaseException:  # pylint: disable=broad-except
            optimizer.logger.debug(msg='', exc_info=True)
            optimizer.metrics['generation_worker_errors_num'] += 1

    optimizer.logger.debug(f'Generating worker {worker_id} finishes...')


def evaluating_pipelines_worker(worker_id, optimizer, input_queue, output_queue):
    """
    Evaluate pipelines from the input_queue and put them into the output_queue.
    :param worker_id:
    :param optimizer:
    :param input_queue:
    :param output_queue:
    :return:
    """

    import time
    import queue

    optimizer.logger.debug(f'Evaluating worker {worker_id} starts...')

    while True:
        if optimizer.state.done:
            break

        try:
            pipeline = input_queue.get_nowait()
            optimizer.logger.debug(f'Evaluating worker {worker_id} starts '
                                   f'evaluating pipeline {pipeline.id}: \n{pipeline.to_json()}')

            for pipeline_executor in optimizer.evaluation_method.validate_pipeline(pipeline):
                output_queue.put(pipeline_executor)

                if optimizer.state.done:
                    break
            optimizer.logger.debug(f'Evaluating worker {worker_id} is done '
                                   f'evaluating pipeline {pipeline.id}: \n{pipeline.to_json()}')

        except queue.Empty:
            optimizer.metrics['evaluation_worker_idle_time'] += 0.01
            time.sleep(0.01)

        except BaseException:  # pylint: disable=broad-except
            optimizer.logger.debug(msg='', exc_info=True)
            optimizer.metrics['evaluation_worker_errors_num'] += 1

    optimizer.logger.debug(f'Evaluating worker {worker_id} finishes...')
