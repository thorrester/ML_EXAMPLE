import os
import re

from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Dict, Iterable, Tuple, Union, Optional
import json
from google.cloud import bigquery
import joblib
import glob

import sklearn
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csc_matrix, coo_matrix
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import time
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from . import bq_client
from . import gcs_client

DATE_PATTERN = re.compile(r"\d{6}")
ABSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def get_features(table: str, filter_list: List[str] = None):
    filter_list = filter_list or []

    client = bq_client.get_client()
    table: bigquery.table.Table = client.get_table(table)

    return {field.name: field.field_type for field in table.schema if field.name not in filter_list}


def split_column_type(features: Dict[str, str], exclude_cols: Iterable[str] = None):
    exclude_cols = set(col.lower() for col in (exclude_cols or []))
    categorical_cols = [k for k, v in features.items() if v == 'STRING' and k.lower() not in exclude_cols]
    numerical_cols = [k for k, v in features.items() if v != 'STRING' and k.lower() not in exclude_cols]
    return categorical_cols, numerical_cols


def query_data(table: str, reduce_mem=True, predict_data=True, id_column: str = None, splits: int = None,
               split: int = None, limit: int = None, where: str = None):
    """ Pulls data

        :param table: Table where prediction data exists
        :param reduce_mem: Boolean indicate whether to use reduced memory function
        :param predict_data: Boolean indicating if prediction or training data is being queried
        :param id_column: ID to split on
        :param splits: Total number of splits to create
        :param split: Unique split to pull data on
        :param limit: Number to limit query results by
        :param where: Where clause to inject into statement
        """
    limit = limit or None
    query = f"SELECT * FROM `{table}` WHERE 1=1"

    if predict_data:
        query = query + f' AND MOD(ABS(FARM_FINGERPRINT({id_column})), {splits}) = {split}'

    if where is not None:
        query = query + f' AND {where}'

    if limit is not None and not predict_data:
        query = query + f' AND RAND() <= {limit}/(SELECT COUNT(1) FROM `{table}`)'

    if limit is not None and predict_data:
        query = query + f' LIMIT {limit}'

    result = bq_client.query(query)
    if reduce_mem:
        result, NA_list = reduce_mem_usage(result)
        print(f'NAs: {NA_list}')

    return result


def preprocess(data, num_cols, cat_cols, embedding_cols, transformer, label=None, train=True, dense=False):
    if type(label) == list:
        if len(label) > 1:
            pass
        else:
            label = label[0]
    if train:
        transformer.fit(data[[*num_cols, *cat_cols, *embedding_cols]])

    if not dense:
        X = coo_matrix(transformer.transform(data[[*num_cols, *cat_cols, *embedding_cols]]), dtype=np.float16).tocsc()
    else:
        X = transformer.transform(data[[*num_cols, *cat_cols, *embedding_cols]])

    if label is not None:
        y = data[label].reset_index(drop=True)
        return X, y

    else:
        return X


def optimize_optuna(objective, direction='minimize', n_trials=3):
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))
    best_params = study.best_trial.params

    return best_params


class Multi_Class_Objective(object):
    def __init__(self, X, y, sample_size):
        # Hold this implementation specific arguments as the fields of the class.
        self.X = X
        self.y = y
        self.sample_size = sample_size

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param = {
            "objective": "multiclass",
            "metric": "multiclass",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_class": 3,
            "max_depth": trial.suggest_int('max_depth', 2, 50),
            "min_data_in_leaf": trial.suggest_int('min_data_in_leaf', 1, 256),
            "min_data_in_bin": trial.suggest_int('min_data_in_bin', 1, 256),
            "min_gain_to_split": trial.suggest_discrete_uniform('min_gain_to_split', 0.1, 5, 0.01),
            "learning_rate": trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 11),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

        }
        sample = np.random.choice(self.X.shape[0], size=int(self.sample_size), replace=False)
        X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(self.X[sample], self.y[sample])

        dtrain = lgb.Dataset(X_train_opt, label=y_train_opt)
        gbm = lgb.train(param, dtrain)
        preds = np.argmax(gbm.predict(X_test_opt), axis=1)
        score = f1_score(y_test_opt, preds, average='micro')

        return score


class Classification_Objective(object):

    def __init__(self, X, y, sample_size):
        # Hold this implementation specific arguments as the fields of the class.
        self.X = X
        self.y = y
        self.sample_size = sample_size

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "max_depth": trial.suggest_int('max_depth', 2, 50),
            "min_child_samples": trial.suggest_int('min_child_samples', 1, 256),
            "min_data_in_bin": trial.suggest_int('min_data_in_bin', 1, 256),
            "min_split_gain": trial.suggest_discrete_uniform('min_split_gain', 0.1, 5, 0.01),
            "learning_rate": trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 11),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 20),

        }

        sample = np.random.choice(self.X.shape[0], size=int(self.sample_size), replace=False)
        X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(self.X[sample], self.y[sample])

        gbm = lgb.LGBMClassifier(**param, n_jobs=-1)
        gbm.fit(X_train_opt, y_train_opt)

        preds = gbm.predict(X_test_opt)
        score = f1_score(y_test_opt, preds, average='binary')
        return score


class Regression_Objective(object):

    def __init__(self, X, y, sample_size):
        # Hold this implementation specific arguments as the fields of the class.
        self.X = X
        self.y = y
        self.sample_size = sample_size

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        param = {
            "metric": "quantile",
            "verbose": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int('n_estimators', 100, 500),
            "max_depth": trial.suggest_int('max_depth', 2, 25),
            "min_child_samples": trial.suggest_int('min_child_samples', 1, 256),
            "min_data_in_bin": trial.suggest_int('min_data_in_bin', 1, 256),
            "min_split_gain": trial.suggest_discrete_uniform('min_split_gain', 0.1, 5, 0.01),
            "learning_rate": trial.suggest_loguniform('learning_rate', 0.005, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 2 ** 11),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        }

        sample = np.random.choice(self.X.shape[0], size=int(self.sample_size), replace=False)
        X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(self.X[sample], self.y[sample])

        gbm = lgb.LGBMRegressor(objective='quantile', alpha=0.5, **param)
        gbm.fit(X_train_opt, y_train_opt)

        preds = gbm.predict(X_test_opt)
        score = mean_absolute_error(y_test_opt, preds)
        return score


def write_to_gcs(bucket, destination_path, file):
    client = gcs_client.get_client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(destination_path)
    blob.upload_from_filename(file)

    print("File {} uploaded to {}.".format(file, destination_path))


def save_model(model, name, bucket, exp_name, model_type: str = 'lgb'):
    model_path = f'{ABSPATH}/runs/artifacts/model'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if model_type == 'net':
        gcs_path = f'gs://{bucket}/artifacts/{exp_name}'
        save_path = f'{gcs_path}/{name}'
        tf.saved_model.save(model, save_path)
        print("File uploaded to {}.".format(save_path))

    else:
        # Save local
        local_model_path = model_path + f'/{name}.pkl'
        joblib.dump(model, filename=local_model_path)
        save_path = f'artifacts/{exp_name}/{name}.pkl'
        write_to_gcs(bucket, save_path, local_model_path)
        gcs_path = f'{bucket}/{save_path}'

    return gcs_path


def load_model(bucket, model_path):
    client = gcs_client.get_client()
    bucket = client.bucket(bucket)
    blob = bucket.blob(model_path)
    blob.download_to_filename(f'model.pkl')
    model = joblib.load('model.pkl')

    return model


def model_selection(table: str):
    # Separate out in case of using different metric magnitudes
    query = f"SELECT MODEL_LOCATION FROM `{table}` WHERE MODEL_TYPE = 'quantile_regression' ORDER BY DATE DESC LIMIT 1"
    df = bq_client.query(query)
    nn_path = str(df['MODEL_LOCATION'][0])

    query = f"SELECT CONFIG_PATH FROM `{table}` WHERE MODEL_TYPE = 'quantile_regression' ORDER BY DATE DESC LIMIT 1"
    df = bq_client.query(query)
    nn_config_path = '/'.join(i for i in df['CONFIG_PATH'][0].split('/')[1:-1])

    return nn_path, nn_config_path

def load_tf_model(config_path: str, weight_path: str, model_name: str, bucket: str):
    config = load_model(bucket, config_path + f'/{model_name}_config.pkl')
    model = tf.keras.Model.from_config(config)
    weight_path = f'{weight_path}/{model_name}/checkpoint'
    model.load_weights(weight_path)

    return model


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def prediction_chunker(id_field: str, predict_table: str, chunk_size: int):
    """ Generates chunks of specified size for inference

    :param id_field: Field to use for ID
    :param predict_table: Table where prediction data exists
    :param chunk_size: Size of chunks
    """
    query = f"SELECT {id_field} FROM `{predict_table}`"
    result = bq_client.query(query)[f'{id_field}'].to_list()

    id_chunks = list(divide_chunks(result, chunk_size))

    return id_chunks


def query_prediction_data(predict_table: str, id_column: str, splits: int, split: int):
    """ Pulls splits of prediction data

    :param predict_table: Table where prediction data exists
    :param id_column: ID to split on
    :param splits: Total number of splits to create
    :param split: Unique split to pull data on
    """
    query = f"SELECT * FROM `{predict_table}` WHERE MOD(ABS(FARM_FINGERPRINT({id_column})), {splits}) = {split}"
    result = bq_client.query(query)
    result, NA_list = reduce_mem_usage(result)
    print(f'NAs: {NA_list}')
    return result


def determine_order(graph: Dict[str, List[str]]) -> List[str]:
    reverse = {}
    for key, nodes in graph.items():
        for node in nodes:
            reverse.setdefault(node, []).append(key)

    graph = reverse
    result: List[str] = []
    seen: Set[str] = set()

    def recursive_helper(node: str) -> None:
        for neighbor in graph.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                recursive_helper(neighbor)
        if node not in result:
            result.append(node)

    for key in graph.keys():
        recursive_helper(key)
    return result


def query_to_table(bq_project: str,
                   bq_dataset: str,
                   table_id: str,
                   query: str,
                   write_type: str = 'truncate',
                   clustering_fields: List[str] = None,
                   destination_partition_field: str = None):
    destination_table = f"{bq_project}.{bq_dataset}.{table_id}"
    if write_type == 'insert':
        write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    job_config = bigquery.QueryJobConfig(
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=write_disposition,
        destination=destination_table,
        clustering_fields=clustering_fields,
        priority=bigquery.QueryPriority.BATCH
    )

    if destination_partition_field:
        job_config.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=destination_partition_field
        )

    start_time = time.perf_counter()
    job = bq_client.get_client().query(query, job_config=job_config)
    print(f"Executing query for '{destination_table}'. Job ID: {job.job_id}")
    job.result()
    print(f"Job done for'{destination_table}'. Job ID: {job.job_id}. "
          f"Duration: {timedelta(seconds=time.perf_counter() - start_time)}")


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            # print("******************************")
            # print("Column: ", col)
            # print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            # print("dtype after: ", props[col].dtype)
            # print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def lgb_f1(y_true, y_pred):
    y_pred = np.where(y_pred < 0.5, 0, 1)
    score = f1_score(y_true, y_pred)
    return 'f1 score', score, True


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def get_tokenizer(table):
    dict_ = query_data(table, reduce_mem=False, predict_data=False, limit=None)
    dict_ = dict(zip(dict_.WORD, dict_.INDEX))
    num_words = len(dict_.keys())
    tokenizer = Tokenizer(num_words=num_words, oov_token='<unk>', filters='{')
    tokenizer.word_index = dict_
    return tokenizer, num_words


def preprocess_text(attributes: List, data: pd.DataFrame, train: True, tokenizer=None):
    tokenized_data = {}
    if train:
        tokenizers = {}
        for name in attributes:
            table = f'analytics-store-ops-thd.OH_ACCURACY_ML.{name}_DICT'
            tokenizer, num_words = get_tokenizer(table)
            tokenized_text = tokenizer.texts_to_sequences(data.loc[data['DATA_LABEL'] == 'TRAIN'][name])
            tokenizers[name] = {}
            tokenizers[name]['tokenizer'] = tokenizer
            tokenizers[name]['num_words'] = num_words

            # Pad
            tokenizers[name]['maxlen'] = max([len(i) for i in tokenized_text])
            tokenized_data[name] = {}
            tokenized_data[name]['TRAIN'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text,
                                                                                          maxlen=tokenizers[name][
                                                                                              'maxlen'],
                                                                                          padding='post')
            for label in ['TEST', 'EVAL']:
                tokenized_text = tokenizer.texts_to_sequences(data.loc[data['DATA_LABEL'] == label][name])
                tokenized_data[name][label] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text,
                                                                                            maxlen=tokenizers[name]['maxlen'],
                                                                                            padding='post')
        return tokenizers, tokenized_data

    else:
        for name in attributes:
            tokenized_text = tokenizer[name]['tokenizer'].texts_to_sequences(data[name])
            tokenized_data[name] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text,
                                                                                 maxlen=tokenizer[name]['maxlen'],
                                                                                 padding='post')
        return tokenized_data
