import sys

import pandas as pd
from alpine_meadow.common import Dataset, Task, TaskKeyword, PerformanceMetric, Config
from alpine_meadow.core import Optimizer


train_df = pd.read_csv(sys.argv[1])
test_df = pd.read_csv(sys.argv[1])
metrics = [PerformanceMetric.Value('F1_MACRO')]
target_column = "CASE_STATUS"
task = Task([TaskKeyword.Value('CLASSIFICATION')], metrics, [target_column],
            dataset=train_df)
print('Naive score', task.compute_naive_score())

# run optimizer
config = Config(timeout_seconds=30, debug=True)
optimizer = Optimizer(task, config=config)
test_dataset = Dataset(test_df)
for result in optimizer.optimize():
    print('Time: {}, Score: {}'.format(result.elapsed_time, result.score))
    print('Test score: {}'.format(result.pipeline.score([test_dataset], [test_df[target_column]]).scores[0]))
