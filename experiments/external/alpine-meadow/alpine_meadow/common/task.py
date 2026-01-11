# pylint: disable=no-else-return, inconsistent-return-statements, invalid-name, raise-missing-from
"""Task description class, including task keyword, optimization metrics, etc."""

from typing import List
import uuid
import datetime

import numpy as np
import pandas as pd
import pyarrow as pa

from .proto.task_pb2 import DatasetType, TaskKeyword, PerformanceMetric
from .proto.task_pb2 import ValidationMethod  # noqa: F401  # pylint: disable=unused-import
from alpine_meadow.utils import ignore_warnings, AMException


class Dataset:
    """Helper class (e.g., split data, acquire dataset schema) for multiple data formats, e.g., DataFrame."""

    def __init__(self, data, task=None):
        self._id = str(uuid.uuid4())
        self._data = data
        # TODO: support more dataset types
        if isinstance(self._data, pd.DataFrame):
            self._schema = pa.Schema.from_pandas(self._data, preserve_index=False)
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

        self._tags = []
        self._task = task
        if self._task is not None:
            self._task.add_dataset_split(self)

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> DatasetType:
        if isinstance(self._data, pd.DataFrame):
            return DatasetType.TABULAR
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    @property
    def num_instances(self) -> int:
        if isinstance(self._data, pd.DataFrame):
            return len(self._data)
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    @property
    def num_columns(self) -> int:
        if isinstance(self._data, pd.DataFrame):
            return len(self._data.columns)
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    @property
    def num_bytes(self) -> int:
        if isinstance(self._data, pd.DataFrame):
            return self._data.memory_usage(index=True).sum()
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tags):
        self._tags = tags

    def from_data_frame(self, df):
        if isinstance(self._data, pd.DataFrame):
            return Dataset(df, self._task)
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    def to_data_frame(self):
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    def to_raw_data(self):
        if isinstance(self._data, pd.DataFrame):
            return self._data
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    def clean_rows(self, target_columns):
        """
        Remove rows with null target columns
        :return:
        """

        if isinstance(self._data, pd.DataFrame):
            self._data = self._data.dropna(subset=target_columns)
        else:
            raise AMException(f"Currently {type(self._data)} is not supported as dataset")

    def dumps(self, full=False):
        dict_ = {
            'id': self.id,
            'type': self.type,
            'num_instances': self.num_instances,
            'tags': self.tags
        }
        if full:
            dict_['schema'] = self.schema.pandas_metadata
        return dict_


class Task:
    """Task description class."""

    def __init__(self, task_keywords, metrics, target_columns: List[str],
                 dataset=None, train_dataset=None, validation_dataset=None,
                 class_weights=None, pos_label=None):
        self._id = str(uuid.uuid4())
        self._keywords = task_keywords
        self._metrics = metrics
        self._target_columns = target_columns
        self._start_time = datetime.datetime.now()
        self._meta_features = None
        self._config = None
        self._dataset_splits = {}

        if dataset is not None:
            self.dataset = dataset
        else:
            if train_dataset is None:
                raise AMException("dataset and train_dataset cannot be None at the same time")
            self.dataset = train_dataset

        if train_dataset is not None:
            if validation_dataset is None:
                raise AMException("validation_dataset and validation_dataset must be None"
                                  " or not None at the same time")
            self.train_dataset = train_dataset
            self.validation_dataset = validation_dataset
        else:
            if validation_dataset is not None:
                raise AMException("validation_dataset and validation_dataset must be None"
                                  " or not None at the same time")
            self.train_dataset = None
            self.validation_dataset = None

        # check target type
        targets = self.dataset.to_data_frame()[self.target_columns]
        if self.type == TaskKeyword.CLASSIFICATION:
            from sklearn.utils.multiclass import type_of_target

            if 'continuous' in type_of_target(targets.values):
                raise AMException("Targets for classification cannot be continuous!")

        # class weights
        self._class_weights = class_weights

        # pos label for precision and recall
        self._pos_label = None
        if pos_label is not None:
            self._pos_label = type(targets.values.item(0))(pos_label)

        self._meta_learning_info = {}
        self._naive_score = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def type(self):
        if TaskKeyword.CLASSIFICATION in self.keywords:
            return TaskKeyword.CLASSIFICATION
        elif TaskKeyword.REGRESSION in self.keywords:
            return TaskKeyword.REGRESSION
        else:
            raise AMException("Currently we only support CLASSIFICATION and REGRESSION")

    @property
    def keywords(self):
        return self._keywords

    @property
    def metrics(self):
        return self._metrics

    @property
    def class_weights(self):
        return self._class_weights

    @property
    def pos_label(self):
        return self._pos_label

    @property
    def scoring_kwargs(self):
        return {
            'class_weights': self.class_weights,
            'pos_label': self.pos_label
        }

    @property
    def target_columns(self) -> List[str]:
        return self._target_columns

    @property
    def start_time(self):
        return self._start_time

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if dataset is None:
            self._dataset = None
            return

        if isinstance(dataset, Dataset):
            self._dataset = dataset
        else:
            self._dataset = Dataset(dataset, self)
        self._dataset.tags.append('input')

        # check datasets such that make sure target columns are in the dataset
        # and there are non-target columns
        columns = set()
        for field in self.dataset.schema:
            columns.add(field.name)
        target_columns_set = set(self.target_columns)
        if not columns.issuperset(target_columns_set):
            raise AMException(f"Some target columns are missing: {columns - target_columns_set}")
        if len(columns) <= len(target_columns_set):
            raise AMException("No non-target columns in the dataset")

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, train_dataset):
        if train_dataset is None:
            self._train_dataset = None
            return

        if isinstance(train_dataset, Dataset):
            self._train_dataset = train_dataset
        else:
            self._train_dataset = Dataset(train_dataset, self)
        self._train_dataset.tags.append('input')
        self._train_dataset.tags.append('train')

    @property
    def validation_dataset(self) -> Dataset:
        return self._validation_dataset

    @validation_dataset.setter
    def validation_dataset(self, validation_dataset):
        if validation_dataset is None:
            self._validation_dataset = None
            return

        if isinstance(validation_dataset, Dataset):
            self._validation_dataset = validation_dataset
        else:
            self._validation_dataset = Dataset(validation_dataset, self)
        self._validation_dataset.tags.append('input')
        self._validation_dataset.tags.append('validation')

    @property
    def meta_features(self):
        return self._meta_features

    @property
    def dataset_splits(self):
        return self._dataset_splits

    @property
    def meta_learning_info(self):
        return self._meta_learning_info

    @meta_learning_info.setter
    def meta_learing_info(self, meta_learning_info):
        self._meta_learning_info = meta_learning_info

    @property
    def naive_score(self):
        return self._naive_score

    @ignore_warnings
    def compute_meta_features(self):
        """
        Compute the meta features for this dataset/task
        """

        # only compute meta features for tabular classification/regression problems
        if self.dataset.type != DatasetType.TABULAR:
            return None

        keywords = self.keywords
        if TaskKeyword.CLASSIFICATION not in keywords and TaskKeyword.REGRESSION not in keywords:
            return None

        # sample from the dataframe
        df = self.dataset.to_data_frame()
        total_size = len(df)
        sample_size = 1000
        if total_size > sample_size:
            df = df.sample(n=sample_size)

        from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype

        # get y
        target_columns = self.target_columns
        target_column = target_columns[0]
        if target_column not in df.columns:
            y = np.ones(len(df))
        else:
            y = df[target_column].values

        # get non-target columns
        columns = []
        categorical_columns = []
        for column in df.columns:
            if column in target_columns:
                continue
            if is_string_dtype(df[column]) or is_bool_dtype(df[column]):
                columns.append(column)
                categorical_columns.append(column)
            elif is_numeric_dtype(df[column]):
                columns.append(column)
        df = df[columns].copy()

        # encode categorical columns
        from sklearn.preprocessing import LabelEncoder

        for categorical_column in categorical_columns:
            le = LabelEncoder()
            df[categorical_column] = le.fit_transform(df[categorical_column].values.astype(str))

        categorical = list(map(lambda column: column in categorical_columns, columns))
        X = df[columns].values

        import tempfile
        from alpine_meadow.core.meta_learning.meta_features import meta_features as meta_features_lib
        from alpine_meadow.core.meta_learning.meta_features.meta_feature import DatasetMetafeatures

        if TaskKeyword.CLASSIFICATION in self.keywords:
            exclude_meta_features = ['Landmark1NN', 'LandmarkDecisionNodeLearner', 'LandmarkDecisionTree',
                                     'LandmarkLDA', 'LandmarkNaiveBayes',
                                     'PCAFractionOfComponentsFor95PercentVariance', 'PCAKurtosisFirstPC',
                                     'PCASkewnessFirstPC', 'PCA']
        else:
            exclude_meta_features = ['Landmark1NN', 'LandmarkDecisionNodeLearner', 'LandmarkDecisionTree',
                                     'LandmarkLDA', 'LandmarkNaiveBayes',
                                     'PCAFractionOfComponentsFor95PercentVariance', 'PCAKurtosisFirstPC',
                                     'PCASkewnessFirstPC', 'NumberOfClasses', 'ClassOccurences',
                                     'ClassProbabilityMin', 'ClassProbabilityMax', 'ClassProbabilityMean',
                                     'ClassProbabilitySTD', 'ClassEntropy', 'LandmarkRandomNodeLearner', 'PCA']

        dataset_id = str(uuid.uuid4())
        with meta_features_lib.metafeatures_lock:  # only one thread can run this because of global variables
            meta_features = meta_features_lib.calculate_all_metafeatures(X, y, categorical, dataset_id,
                                                                         dont_calculate=exclude_meta_features)
        meta_features.metafeature_values['NumberOfInstances'].value = total_size
        with tempfile.NamedTemporaryFile(delete=False) as meta_features_file:
            meta_features.dump(meta_features_file.name)

        meta_features = DatasetMetafeatures.load(meta_features_file.name)
        self._meta_features = meta_features

        return meta_features

    def compute_naive_score(self, df=None):
        """
        Return the naive score for this task, e.g., for classification, we always predict the majority;
        for the regression, we always predict the mean.
        """

        from alpine_meadow.common.metric import get_score

        if df is None:
            df = self.dataset.to_data_frame()
        metric = self.metrics[0]
        target_column = self.target_columns[0]

        if metric in [PerformanceMetric.ROC_AUC, PerformanceMetric.ROC_AUC_MICRO,
                      PerformanceMetric.ROC_AUC_MACRO]:
            self._naive_score = 0
            return self._naive_score

        if self.type == TaskKeyword.CLASSIFICATION:
            if metric in [PerformanceMetric.PRECISION, PerformanceMetric.RECALL,
                          PerformanceMetric.F1]:
                pos_label = self.scoring_kwargs.get('pos_label', None)
                if pos_label is None:
                    raise AMException("Expect pos_label for precision/recall/f1")
                label = pos_label
            else:
                label = df[target_column].mode().values[0]
            new_df = pd.DataFrame(df[target_column])
            new_df[target_column + '_test'] = label
        else:
            if self.type != TaskKeyword.REGRESSION:
                raise AMException(f"Unknown task type: {self.type}")
            try:
                mean = df[target_column].mean()
            except TypeError as e:
                if 'numeric' in f'{e}':
                    raise AMException(f"Column {target_column} is not numeric!")
                raise e
            new_df = pd.DataFrame(df[target_column])
            new_df[target_column + '_test'] = mean

        self._naive_score = get_score(metric, new_df[target_column], new_df[target_column + '_test'],
                                      **self.scoring_kwargs)
        return self._naive_score

    def add_dataset_split(self, dataset_split):
        self._dataset_splits[dataset_split.id] = dataset_split

    def create_dataset_splits(self, cv, df=None):
        """
        Create dataset splits from cross-validation indexes.
        :param cv:
        :param df:
        :return:
        """

        if df is None:
            df = self.dataset.to_data_frame()
        if self.type in [TaskKeyword.CLASSIFICATION]:
            labels = df[self.target_columns[0]].values
        else:
            labels = np.ones(len(df))

        dataset_splits = []
        for train_index, test_index in cv.split(df, labels):
            train_data_frame = df.iloc[train_index]
            test_data_frame = df.iloc[test_index]

            # create train dataset
            train_dataset = self.dataset.from_data_frame(train_data_frame)
            train_dataset.tags.append('train')
            train_dataset.tags.append('cross-validation')

            # create validation dataset
            test_dataset = self.dataset.from_data_frame(test_data_frame)
            test_dataset.tags.append('validation')
            test_dataset.tags.append('cross-validation')

            if self.target_columns is None:
                dataset_splits.append((train_dataset, test_dataset))
            else:
                test_target = test_dataset.to_data_frame()[self.target_columns]
                dataset_splits.append((train_dataset, (test_dataset, test_target)))
        return dataset_splits

    def dumps(self):
        return {
            'id': self.id,
            'keywords': ','.join(map(TaskKeyword.Name, self.keywords)),
            'target_columns': self.target_columns,
            'metrics': ','.join(map(PerformanceMetric.Name, self.metrics)),
            'start_time': f'{self.start_time}',
            'dataset': self.dataset.dumps(True),
            'dataset_splits': list(map(lambda split: split.dumps(), self.dataset_splits.values())),
            'meta_features': self.meta_features.dumps() if self.meta_features is not None else None,
            'class_weights': self.class_weights,
            'pos_label': self.pos_label
        }

    def __str__(self):
        repr_ = '--- Alpine Meadow Task ---'
        repr_ += f'Id: {self.id}\n'
        repr_ += f"Task keywords: {','.join(map(TaskKeyword.Name, self.keywords))}\n"
        repr_ += f"Target columns: {','.join(self.target_columns)}\n"
        repr_ += f"Metrics: {','.join(map(PerformanceMetric.Name, self.metrics))}\n"
        repr_ += f'Train dataset size: {self.train_dataset.num_instances}\n'
        repr_ += f'Validation dataset size: {self.validation_dataset.num_instances}\n'
        repr_ += f'Class weights: {self.class_weights}\n'
        if self.pos_label is not None:
            repr_ += f'Pos label: {self.pos_label}\n'
        return repr_
