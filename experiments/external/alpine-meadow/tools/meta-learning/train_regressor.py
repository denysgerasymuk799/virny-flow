"""Train regressor for meta-learning."""
import argparse
import pickle
import time

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer as Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from alpine_meadow.common import TaskKeyword
from alpine_meadow.utils import ignore_warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Converting traces")
    parser.add_argument("--input", type=str, help="input path", required=True)
    parser.add_argument("--output", type=str, help="output path", required=True)

    return parser.parse_args()


@ignore_warnings
def main():
    args = parse_args()

    # load datasets
    with open(args.input, 'rb') as f:
        train_data = pickle.load(f)

    # process
    regressors = {}
    for task_type in [TaskKeyword.Value('CLASSIFICATION'), TaskKeyword.Value('REGRESSION')]:
        start = time.perf_counter()

        # input
        X, y = train_data[task_type]

        # train
        imputer = Imputer(strategy='most_frequent')
        regressor = RandomForestRegressor()
        pipeline = Pipeline([('Imputer', imputer), ('RandomForestRegressor', regressor)])
        pipeline.fit(X, y)

        # test
        y_pred = pipeline.predict(X)
        error = mean_squared_error(y, y_pred)

        regressors[task_type] = pipeline
        print('Task type: {}, # of training data: {}, error:{}, time: {}'.format(
            TaskKeyword.Name(task_type), len(X), error, time.perf_counter() - start
        ))

    # dump
    with open(args.output, 'wb') as f:
        pickle.dump(regressors, f)


if __name__ == "__main__":
    main()
