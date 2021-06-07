from typing import List

import os
import logging

from commons import utils, bq_client
from inference.nn_model_training import net_training_fn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
import datetime
import numpy as np
import pandas as pd
import math

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

date_ = datetime.datetime.now()


def training_fn(bq_project: str,
                bq_dataset: str,
                train_table_id: str,
                model_name: str,
                embedding_features: List[str],
                metadata_table_id: str,
                filter_list: List[str],
                git_branch: str,
                batch_size: int,
                epochs: int,
                data_limit: int = None,
                # regression_labels: List[str] = None,
                regression_labels: List[str] = None,
                hyperparameter_tuning=True):
    """Runs training on a model against the given data from BigQuery

    :param bq_project: BigQuery project ID where the dataset is residing
    :param bq_dataset: BigQuery Dataset ID where the the table is residing
    :param embedding_features: List of features for entity embeddings
    :param metadata_table_id: Table ID that holds training metadata
    :param filter_list: List of columns to filter from data
    :param git_branch: git branch to associated with code
    :param data_limit: number of records to run training on
    :param regression_labels: List of labels used for regression targets
    :param classification_labels: List of labels used for classification targets
    :param hyperparameter_tuning: Boolean that indicates if hyperparameter tuning will be done

    """
    labels = regression_labels

    print(f"Starting training: "
          f"| project: {bq_project} "
          f"| dataset: {bq_dataset} "
          f"| table: {train_table_id}")

    exp_name = str(math.floor(datetime.datetime.now().timestamp())) + f'_{git_branch}'
    print(f'Experiment: {exp_name}')

    train_table = f'{bq_project}.{bq_dataset}.{train_table_id}'
    metadata_table = f'{bq_project}.{bq_dataset}.{metadata_table_id}_{git_branch}'

    # Pull data
    data = utils.query_data(train_table, reduce_mem=True, predict_data=False, limit=data_limit)
    #data = pd.read_csv('train_data.csv', nrows=100000)

    # Get features and type
    features = utils.get_features(train_table, filter_list)
    categorical_cols, numeric_cols = utils.split_column_type(features, exclude_cols=labels)
    categorical_cols = list(set(categorical_cols) - set(embedding_features))

    print(f'categorical features: {categorical_cols}')
    print(f'numerical features: {numeric_cols}')
    print(f'embedding features: {embedding_features}')

    # Preprocess
    #data.to_csv('train_data.csv')
    unknown_val = len(np.unique(data.loc[data['DATA_LABEL'] == 'TRAIN'][embedding_features])) + 1
    print(unknown_val)

    # column transformer
    num_pipe = Pipeline([('imputer', SimpleImputer()), ('normalize', StandardScaler())])
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numeric_cols),
                                                  ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False,
                                                                        dtype=np.float32), categorical_cols),
                                                  ('embed', OrdinalEncoder(handle_unknown='use_encoded_value',
                                                                           unknown_value=unknown_val, dtype=np.float32),
                                                   embedding_features)
                                                  ],
                                    # one hot encode all categoricals
                                    sparse_threshold=0,
                                    remainder='passthrough'
                                    )

    print(f"Labels: {labels}")

    x_train, y_train = utils.preprocess(data.loc[data['DATA_LABEL'] == 'TRAIN'],
                                        numeric_cols,
                                        categorical_cols,
                                        embedding_features,
                                        label=labels,
                                        dense=True,
                                        transformer=transformer,
                                        train=True)

    x_test, y_test = utils.preprocess(data.loc[data['DATA_LABEL'] == 'TEST'],
                                      numeric_cols,
                                      categorical_cols,
                                      embedding_features,
                                      label=labels,
                                      dense=True,
                                      transformer=transformer,
                                      train=False)

    x_eval, y_eval = utils.preprocess(data.loc[data['DATA_LABEL'] == 'EVAL'],
                                      numeric_cols,
                                      categorical_cols,
                                      embedding_features,
                                      label=labels,
                                      dense=True,
                                      transformer=transformer,
                                      train=False)

    ohe_path = utils.save_model(transformer, f'{model_name}_transformer', 'store-ops-ml', exp_name)

    print(f'Train shape: {x_train.shape}, {y_train.shape}')
    print(f'Test shape: {x_test.shape}, {y_test.shape}')
    print(f'Eval shape: {x_eval.shape}, {y_eval.shape}')
    feature_names = utils.get_feature_names(transformer)

    # find embedding features and exclude them from the lgbm modeling
    embed_bay_idx = feature_names.index('embed__BAY_LOC')
    print(f'Embedding col index: {embed_bay_idx}')

    bay_embed_train = x_train[:, embed_bay_idx]
    bay_embed_test = x_test[:, embed_bay_idx]
    bay_embed_eval = x_eval[:, embed_bay_idx]
    num_bay_tokens = len(np.unique(bay_embed_train))
    print(num_bay_tokens)

    x_train = x_train[:, :embed_bay_idx]
    x_test = x_test[:, :embed_bay_idx]
    x_eval = x_eval[:, :embed_bay_idx]

    print(x_train.shape, bay_embed_train.shape)
    print(x_test.shape, bay_embed_test.shape)
    print(x_eval.shape, bay_embed_eval.shape)

    net_training_fn(train_data=(x_train, bay_embed_train, y_train),
                    test_data=(x_test, bay_embed_test, y_test),
                    eval_data=(x_eval, bay_embed_eval, y_eval),
                    data_features=[*categorical_cols, *numeric_cols, *embedding_features],
                    model_name=model_name,
                    num_tokens=num_bay_tokens,
                    batch_size=batch_size,
                    epochs=epochs,
                    metadata_table=metadata_table,
                    exp_name=exp_name
                    )

