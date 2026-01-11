"""Convert meta-learning traces to Alpine Meadow internal representations."""
import argparse
import json
import pickle
import os
import traceback

import numpy as np
from smac.tae import StatusType

from alpine_meadow.common import TaskKeyword
from alpine_meadow.core.meta_learning import TaskTrace
from alpine_meadow.core.meta_learning.meta_features.meta_feature import DatasetMetafeatures


def parse_args():
    parser = argparse.ArgumentParser(description="Converting traces")
    parser.add_argument("--input", type=str, help="input path", required=True)
    parser.add_argument("--output", type=str, help="output path", required=True)

    return parser.parse_args()


def process_run_history(task_type, run_history):
    # get all data
    all_data = []
    for run_history_item in run_history:
        for data_item in run_history_item['data']:
            all_data.append(data_item[1][0])

    # process all data
    all_data = np.array(all_data)
    if task_type == TaskKeyword.Value('REGRESSION'):
        all_data = all_data[~np.isnan(all_data)]
        all_data = all_data[~np.isinf(all_data)]
        median = np.median(all_data)
        all_data = all_data[all_data <= 10 * median]

    # normalize
    mean, std = np.mean(all_data), np.std(all_data)
    for run_history_item in run_history:
        data_items = []
        for data_item in run_history_item['data']:
            if task_type == TaskKeyword.Value('REGRESSION'):
                if data_item[1][0] <= 10 * median:
                    data_item[1][0] = (data_item[1][0] - mean) / std
                    data_items.append(data_item)
            else:
                assert task_type == TaskKeyword.Value('CLASSIFICATION')
                data_item[1][0] = (data_item[1][0] - mean) / std
                data_items.append(data_item)
        run_history_item['data'] = data_items

    # save into dict
    run_history_dict = {}
    for pipeline_arm_run_history_json in run_history:
        key = frozenset(pipeline_arm_run_history_json['pipeline_arm'])
        run_history_dict[key] = pipeline_arm_run_history_json

    return run_history_dict


def load_trace(trace_path):
    # parse json
    with open(trace_path, 'r') as f:
        trace_json = json.load(f, object_hook=StatusType.enum_hook)

    # get task type
    keywords = trace_json['task']['keywords']
    if 'CLASSIFICATION' in keywords:
        task_type = TaskKeyword.Value('CLASSIFICATION')
    else:
        assert 'REGRESSION' in keywords
        task_type = TaskKeyword.Value('REGRESSION')

    # get meta features
    meta_features = DatasetMetafeatures.load_from_dict(trace_json['task']['meta_features'])

    # get run history
    run_history = process_run_history(task_type, trace_json['run_history'])

    # save dataset metadata
    dataset_metadata = TaskTrace(task_type=task_type, meta_features=meta_features,
                                 run_history=run_history)

    return trace_json['task']['id'], dataset_metadata


def main():
    args = parse_args()

    # load traces
    datasets = {}
    for root, sub_dirs, files in os.walk(args.input):
        for file in files:
            if file.endswith('.json'):
                trace_path = os.path.join(root, file)
                try:
                    dataset_id, dataset_metadata = load_trace(trace_path)
                    datasets[dataset_id] = dataset_metadata
                    if len(datasets) % 10 == 0:
                        print('Loaded {} datasets'.format(len(datasets)))
                except:
                    traceback.print_exc()

    print('Loaded {} datasets'.format(len(datasets)))

    # dump
    with open(args.output, 'wb') as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    main()
