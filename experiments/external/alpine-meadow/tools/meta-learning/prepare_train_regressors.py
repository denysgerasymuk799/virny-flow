"""Prepare training data for regressor of meta-learning."""
import argparse
import pickle
import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from alpine_meadow.common import TaskKeyword
from alpine_meadow.utils import ignore_warnings


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "smac.tae.execute_ta_run":
            renamed_module = "smac.tae"

        return super().find_class(renamed_module, name)


def parse_args():
    parser = argparse.ArgumentParser(description="Converting traces")
    parser.add_argument("--input", type=str, help="input path", required=True)
    parser.add_argument("--output", type=str, help="output path", required=True)

    return parser.parse_args()


def get_pipelines_scores(task_trace, pipelines):
    # get scores
    scores = []
    for pipeline in pipelines:
        all_data = []
        for data_item in task_trace.run_history[pipeline]['data']:
            all_data.append(data_item[1][0])
        score = np.mean(all_data)
        scores.append(score)

    # fill nan with means
    scores = np.array(scores)
    mean = np.nanmean(scores)
    scores[np.where(np.isnan(scores))] = mean

    # make it 2D
    scores = scores.reshape(1, -1)

    return scores


@ignore_warnings
def main():
    args = parse_args()

    # load datasets
    with open(args.input, 'rb') as f:
        datasets = RenameUnpickler(f).load()

    # process
    train_data = {}
    for task_type in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]:
        start = time.perf_counter()

        task_datasets = []
        meta_features = {}
        for dataset, task_trace in datasets.items():
            if task_trace.task_type != task_type:
                continue

            task_datasets.append(dataset)
            meta_features[dataset] = task_trace.get_meta_features_values(task_trace.meta_features)

        task_train_X = []
        task_train_y = []
        for left_dataset in task_datasets:
            for right_dataset in task_datasets:
                # get common pipelines
                left_task_trace = datasets[left_dataset]
                right_task_trace = datasets[right_dataset]
                common_pipelines = set(left_task_trace.run_history).intersection(set(right_task_trace.run_history))
                if not common_pipelines:
                    continue

                # get similarity
                common_pipelines = list(common_pipelines)
                left_scores = get_pipelines_scores(left_task_trace, common_pipelines)
                right_scores = get_pipelines_scores(right_task_trace, common_pipelines)
                if np.isnan(np.sum(left_scores)) or np.isnan(np.sum(right_scores)):
                    continue

                similarity = cosine_similarity(left_scores, right_scores).flatten()[0]

                # prepare features and label
                features = np.array(meta_features[left_dataset] + meta_features[right_dataset],
                                    dtype=float)
                label = similarity
                task_train_X.append(features)
                task_train_y.append(label)
        task_train_X = np.array(task_train_X, dtype=float)
        task_train_y = np.array(task_train_y, dtype=float)

        train_data[task_type] = (task_train_X, task_train_y)
        print('Task type: {}, # of training data: {}, time: {}'.format(
            TaskKeyword.Name(task_type), len(task_train_X), time.perf_counter() - start
        ))

    # dump
    with open(args.output, 'wb') as f:
        pickle.dump(train_data, f)


if __name__ == "__main__":
    main()
