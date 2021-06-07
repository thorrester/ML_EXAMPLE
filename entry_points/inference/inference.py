from commons import utils, bq_client
from typing import List
import joblib
import numpy as np
import os
import datetime
from concurrent import futures
from inference.nn_inference import inference_reg_fn
# from inference.inference_reg import inference_reg_fn
# from inference.inference_class import inference_class_fn
import tensorflow as tf


def inference_fn(bq_project: str,
                 bq_dataset: str,
                 train_table_id: str,
                 predict_table_id: str,
                 model_name: str,
                 embedding_features: List[str],
                 # regression_labels: List[str],
                 metadata_table_id: str,
                 prediction_output_table: str,
                 splits: int,
                 id_field: str,
                 bucket: str,
                 filter_list: List[str],
                 git_branch: str,
                 data_limit: int = None):
    train_table = f'{bq_project}.{bq_dataset}.{train_table_id}'
    predict_table = f'{bq_project}.{bq_dataset}.{predict_table_id}'
    metadata_table = f'{bq_project}.{bq_dataset}.{metadata_table_id}_{git_branch}'
    prediction_output_table = f'{bq_project}.{bq_dataset}.{prediction_output_table}_{git_branch}_staged'

    # Get features and type
    features = utils.get_features(train_table, filter_list)
    categorical_cols, numeric_cols = utils.split_column_type(features)
    categorical_cols = list(set(categorical_cols) - set(embedding_features))

    # Pull model paths
    nn_weight_path, nn_config_path = utils.model_selection(metadata_table)

    # Load transformer
    transformer = utils.load_model(bucket, nn_config_path + f'/{model_name}_transformer.pkl')

    # Load model
    model = utils.load_tf_model(nn_config_path, nn_weight_path, model_name, bucket)

    feature_names = utils.get_feature_names(transformer)
    # find embedding features and exclude them from the lgbm modeling
    embed_bay_idx = feature_names.index('embed__BAY_LOC')
    print(f'Embedding col index: {embed_bay_idx}')

    # Delete prediction table
    client = bq_client.get_client()
    client.delete_table(prediction_output_table, not_found_ok=True)
    print("Deleted table '{}'.".format(prediction_output_table))

    print('Starting predictions')
    for split in range(0, splits):
        # call data
        predict_data = utils.query_data(predict_table, reduce_mem=False, predict_data=True, id_column=id_field,
                                        splits=splits, split=split, limit=data_limit)

        x_pred = utils.preprocess(data=predict_data, num_cols=numeric_cols, cat_cols=categorical_cols,
                                  embedding_cols=embedding_features,
                                  transformer=transformer, train=False, dense=True)

        bay_embed_pred = x_pred[:, embed_bay_idx]
        x_pred = x_pred[:, :embed_bay_idx]

        bay_str_data = predict_data.copy()[['BAY_LOC', 'STR_NBR']]

        inference_reg_fn(num_data=x_pred,
                            bay_data=bay_embed_pred,
                            bay_str_data=bay_str_data,
                            prediction_output_table=prediction_output_table,
                            model=model)

        print(f'Results saved to: {prediction_output_table} for split {split}')
