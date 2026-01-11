import sys
import os

import json
import numpy as np
import pandas as pd
import time
import traceback

from alpine_meadow.common import Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.common.metric import get_score
from alpine_meadow.core import Optimizer


def get_df(dataset_path, problem_path, split_label):
    df = pd.read_csv(os.path.join(dataset_path, 'tables', 'learningData.csv'))
    df = df.set_index('d3mIndex')
    df.index = df.index.map(np.int64)
    splits = pd.read_csv(os.path.join(problem_path, 'dataSplits.csv'))
    df = df.reindex(splits[splits['type'] == split_label.upper()].index).dropna(how='all')

    return df


def get_problem_info(problem_path):
    problem_schema_path = os.path.join(problem_path, "problemDoc.json")
    with open(problem_schema_path, 'r', encoding='utf8') as problem_schema_file:
        problem_schema = json.load(problem_schema_file)

    if 'taskKeywords' in problem_schema['about']:
        keywords = problem_schema['about']['taskKeywords']
    else:
        keywords = [problem_schema['about']['taskType']]
    if 'classification' in keywords:
        task_type = TaskKeyword.Value('CLASSIFICATION')
        metric = PerformanceMetric.Value('F1_MACRO')
    else:
        assert 'regression' in keywords
        task_type = TaskKeyword.Value('REGRESSION')
        metric = PerformanceMetric.Value('MEAN_SQUARED_ERROR')

    target_columns = list(map(lambda x: x['colName'], problem_schema['inputs']['data'][0]['targets']))

    return {
        'task_type': task_type,
        'metric': metric,
        'target_columns': target_columns
    }


def run_dump_exp(dataset_dir, problem_dir, timeout_seconds, output_path, fe_enabled=False):
    # optimizer config
    config = Config(debug=True)
    config.timeout_seconds = timeout_seconds
    config.enable_cross_validation = True
    config.enable_cost_model = False
    config.enable_aps_pruning = False
    config.aps_sub_epochs_num = 10
    config.log_trace = True
    config.enable_feature_engineering = fe_enabled
    config.fe_hyperparams["verbose"] = True

    # tasks
    train_df = get_df(dataset_dir, problem_dir, 'TRAIN')
    test_df = get_df(dataset_dir, problem_dir, 'TEST')
    problem_info = get_problem_info(problem_dir)

    task_type = problem_info['task_type']
    metric = problem_info['metric']
    target_columns = problem_info['target_columns']
    task = Task([task_type], [metric], target_columns, dataset=train_df)
    test_dataset = task.dataset.from_data_frame(test_df)
    true_results = test_df[target_columns].values

    # optimizer
    optimizer = Optimizer(task, config=config)

    # run
    dataset_id = '.'.join(output_path.split('.')[:-1])
    result_df = pd.DataFrame(columns=('Time', 'Score'))
    start = time.time()
    for result in optimizer.optimize():
        pipeline = result.pipeline

        time_ = time.time() - start
        predicted_results = pipeline.test([test_dataset]).outputs
        predicted_results = predicted_results.astype(true_results.dtype)
        score = get_score(metric, true_results, predicted_results)
        result_df.loc[len(result_df)] = [time_, score]
        print('Time: {}, Score: {}'.format(time_, score), flush=True)

        with open(os.path.join(output_path), 'w') as results_file:
            results_file.write(result_df.to_csv(index=False))

        try:
            optimizer.dump('{}.json'.format(dataset_id))
        except:
            traceback.print_exc()

    # clean up
    os.system('rm -rf /tmp/tmp*')

    return result_df


if __name__ == "__main__":
    if len(sys.argv) == 6:
        fe_enabled = sys.argv[5].lower() == 'true'
        run_dump_exp(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], fe_enabled=fe_enabled)
    else:
        run_dump_exp(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
