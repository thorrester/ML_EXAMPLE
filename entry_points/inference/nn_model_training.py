from typing import List, Tuple
import os
import logging

from commons import utils, bq_client
from google.cloud import bigquery
import datetime
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import time
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
plt.style.use(f'{PARENT_DIR}/commons/custom_style.mplstyle')

log = logging.getLogger(__name__)
log.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
date_ = datetime.datetime.now()

#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)

def tilted_loss(q, y, f):
    e = (y - f)
    max_ = tf.maximum(q * e, (q - 1) * e)
    if len(max_.shape) > 1:
        max_ = tf.squeeze(max_)

    return tf.reduce_mean(tf.squeeze(max_), axis=-1)

class tilted_metric():
    def __init__(self, fn):
        self.y = np.asanyarray([]).reshape(-1,1)
        self.preds = np.asanyarray([]).reshape(-1,1)
        self.fn = fn

    def update_state(self, y, preds):
        self.y = np.append(self.y, np.array(y), axis=0)
        self.preds = np.append(self.preds, np.array(preds), axis=0)

    def result(self):
        result = self.fn(self.y, self.preds)
        return result

    def reset_states(self):
        self.y = np.asanyarray([]).reshape(-1,1)
        self.preds = np.asanyarray([]).reshape(-1,1)

def nn_model(num_tokens, num_data_shape):
    # sku desc
    embed_dim = 100

    attribute = 'BAY_EMBED'
    bay_inputs = tf.keras.layers.Input(shape=(1,), name=attribute)
    embedding_layer = tf.keras.layers.Embedding(num_tokens+2, output_dim=embed_dim)(bay_inputs)
    embedding_layer = tf.keras.layers.Flatten()(embedding_layer)

    num_data_inputs = tf.keras.layers.Input(shape=(num_data_shape,), name='NUM_DATA')
    num_data_x = tf.keras.layers.concatenate([num_data_inputs, embedding_layer])

    x = tf.keras.layers.Dense(256, activation='relu')(num_data_x)
    stage01_x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(stage01_x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    oos = tf.keras.layers.Dense(1, activation='linear', name='OOS_PERC', dtype='float32')(x)
    out = tf.keras.layers.Dense(1, activation='linear', name='SHELF_OUT_PERC', dtype='float32')(x)
    low = tf.keras.layers.Dense(1, activation='linear', name='SHELF_LOW_PERC', dtype='float32')(x)

    x = tf.keras.layers.concatenate([stage01_x, oos, out, low])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    opp_val = tf.keras.layers.Dense(1, activation='linear', name='BAY_RETL_OPP', dtype='float32')(x)

    model = tf.keras.Model(
        inputs={'BAY_EMBED': bay_inputs,
                'NUM_DATA': num_data_inputs},
        outputs=[oos, out, low, opp_val]
    )
    return model



def net_training_fn(train_data: Tuple,
                    test_data: Tuple,
                    eval_data: Tuple,
                    model_name: str,
                    num_tokens: int,
                    batch_size: int,
                    epochs: int,
                    metadata_table: str,
                    data_features: List[str],
                    exp_name: str,
                    #label_columns: List[str],
                    #indices: tuple,
                    #indice_label: str,
                    #storage_bucket: str
                    ):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    x_train, bay_embed_train, y_train = train_data
    x_test, bay_embed_test, y_test = test_data
    x_eval, bay_embed_eval, y_eval = eval_data

    # create tf. dataset
    train_data = tf.data.Dataset.from_tensor_slices(({'BAY_EMBED': bay_embed_train,'NUM_DATA': x_train},
                                                     {'OOS_PERC': y_train['OOS_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_OUT_PERC': y_train['SHELF_OUT_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_LOW_PERC': y_train['SHELF_LOW_PERC'].to_numpy().reshape(-1,1),
                                                      'BAY_RETL_OPP': y_train['BAY_RETL_OPP'].to_numpy().reshape(-1,1)
                                                      }))

    test_data = tf.data.Dataset.from_tensor_slices(({'BAY_EMBED': bay_embed_test,'NUM_DATA': x_test},
                                                     {'OOS_PERC': y_test['OOS_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_OUT_PERC': y_test['SHELF_OUT_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_LOW_PERC': y_test['SHELF_LOW_PERC'].to_numpy().reshape(-1,1),
                                                      'BAY_RETL_OPP': y_test['BAY_RETL_OPP'].to_numpy().reshape(-1, 1)
                                                      }))

    eval_data = tf.data.Dataset.from_tensor_slices(({'BAY_EMBED': bay_embed_eval,'NUM_DATA': x_eval},
                                                     {'OOS_PERC': y_eval['OOS_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_OUT_PERC': y_eval['SHELF_OUT_PERC'].to_numpy().reshape(-1,1),
                                                      'SHELF_LOW_PERC': y_eval['SHELF_LOW_PERC'].to_numpy().reshape(-1,1),
                                                      'BAY_RETL_OPP': y_eval['BAY_RETL_OPP'].to_numpy().reshape(-1, 1)
                                                      }))

    steps_per_epoch = x_train.shape[0] // batch_size
    train_ds = train_data.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_data.batch(batch_size, drop_remainder=True).prefetch(1)
    eval_ds = eval_data.batch(batch_size, drop_remainder=True).prefetch(1)

    model = nn_model(num_tokens, x_train.shape[1])
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, decay=0.01, nesterov=True)
    loss_50th_fn = lambda y, f: tilted_loss(0.50, y, f)
    train_oos_metric_fn = tilted_metric(loss_50th_fn)
    train_out_metric_fn = tilted_metric(loss_50th_fn)
    train_low_metric_fn = tilted_metric(loss_50th_fn)
    train_opp_val_metric_fn = tilted_metric(loss_50th_fn)
    test_out_metric_fn = tilted_metric(loss_50th_fn)
    test_opp_val_metric_fn = tilted_metric(loss_50th_fn)
   # test_metric_fn = tilted_metric(loss_50th_fn)


    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            q50_oos, q50_out, q50_low, q50_opp_val = model(x, training=True)
            q50_oos_loss = loss_50th_fn(y['OOS_PERC'], q50_oos)
            q50_out_loss = loss_50th_fn(y['SHELF_OUT_PERC'], q50_out)
            q50_low_loss = loss_50th_fn(y['SHELF_LOW_PERC'], q50_low)
            q50_opp_val_loss = loss_50th_fn(y['BAY_RETL_OPP'], q50_opp_val)
        grads = tape.gradient([q50_oos_loss, q50_out_loss, q50_low_loss, q50_opp_val_loss], model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return q50_oos, q50_out, q50_low, q50_opp_val

    @tf.function
    def test_step(x,y):
        q50_oos_pred, q50_out_pred, q50_low_pred, q50_opp_val = model(x, training=False)
        return q50_oos_pred, q50_out_pred, q50_low_pred, q50_opp_val

    print(model.summary())

    #Train
    history={'epoch': [],
             'train_loss': [],
             'test_loss': [],
             }
    for epoch in range(1, epochs + 1):
        print(f'\nStart of epoch: {epoch}')
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in zip(tqdm(range(steps_per_epoch), position=0, leave=True), train_ds):
            q50_oos_pred, q50_out_pred, q50_low_pred, q50_opp_val_pred = train_step(x_batch_train, y_batch_train)
            train_oos_metric_fn.update_state(y_batch_train['OOS_PERC'], q50_oos_pred)
            train_out_metric_fn.update_state(y_batch_train['SHELF_OUT_PERC'], q50_out_pred)
            train_low_metric_fn.update_state(y_batch_train['SHELF_LOW_PERC'], q50_low_pred)
            train_opp_val_metric_fn.update_state(y_batch_train['BAY_RETL_OPP'], q50_opp_val_pred)

        metrics = []
        for label_, metric_fn in zip(['oos', 'out', 'low', 'opp_val'], [train_oos_metric_fn, train_out_metric_fn, train_low_metric_fn, train_opp_val_metric_fn]):
            metric = round(float(metric_fn.result()),4)
            metrics.append(f'{label_}: {metric}')
            metric_fn.reset_states()
            if label_ == 'out':
                history['epoch'].append(epoch)
                history['train_loss'].append(metric)
        print(f'Training metrics over epoch:', sep='')
        print(*metrics, sep=' ')

        for x_batch_test, y_batch_test in test_ds:
            _, q50_out_pred, _, q50_opp_val_pred = test_step(x_batch_test, y_batch_test)
            test_out_metric_fn.update_state(y_batch_test['SHELF_OUT_PERC'], q50_out_pred)
            test_opp_val_metric_fn.update_state(y_batch_test['BAY_RETL_OPP'], q50_opp_val_pred)

        test_out_metric = round(float(test_out_metric_fn.result()),4)
        test_opp_val_metric = round(float(test_opp_val_metric_fn.result()), 4)
        test_opp_val_metric_fn.reset_states()
        test_out_metric_fn.reset_states()
        history['test_loss'].append(test_out_metric)
        print(f'Test out q50: {float(test_out_metric)}')
        print(f'Test opp val q50: {float(test_opp_val_metric)}')
        print(f'Time taken: {time.time() - start_time}')

    #Evaluation
    for x_batch_test, y_batch_test in eval_ds:
        _, q50_out_pred, _, q50_opp_val_pred = test_step(x_batch_test, y_batch_test)
        test_out_metric_fn.update_state(y_batch_test['SHELF_OUT_PERC'], q50_out_pred)
        test_opp_val_metric_fn.update_state(y_batch_test['BAY_RETL_OPP'], q50_opp_val_pred)

    eval_out_metric = round(float(test_out_metric_fn.result()), 4)
    eval_opp_val_metric = round(float(test_opp_val_metric_fn.result()), 4)
    print(f'Eval out q50: {float(eval_out_metric)}')
    print(f'Eval opp val q50: {float(eval_opp_val_metric)}')

    #Save model
    gcs_path = f'gs://store-ops-ml/artifacts/{exp_name}'
    weight_path = f'{gcs_path}/{model_name}/checkpoint'
    model.save_weights(weight_path, save_format='tf')

    model_config = model.get_config()
    config_path = utils.save_model(model_config, f'{model_name}_config', 'store-ops-ml', exp_name)

    print(weight_path, gcs_path)

    results = pd.DataFrame.from_dict(
        dict(DATE=date_,
             MODEL_TYPE=['quantile_regression'],
             MODEL_LABEL=f"multi-output",
             VERSION=f"{model_name}_{exp_name}",
             DATA_FEATURES=[f"{data_features}" or None],
             EVAL_OUT=eval_out_metric,
             EVAL_OPP=eval_opp_val_metric,
             MODEL_LOCATION=[gcs_path],
             CONFIG_PATH=[config_path]
             )
    )

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("MODEL_TYPE", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("MODEL_LABEL", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("VERSION", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("DATA_FEATURES", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("EVAL_OUT", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("EVAL_OPP", bigquery.enums.SqlTypeNames.FLOAT64),
            bigquery.SchemaField("MODEL_LOCATION", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("CONFIG_PATH", bigquery.enums.SqlTypeNames.STRING),
        ],
        # write_disposition="WRITE_TRUNCATE",
    )

    job = bq_client.get_client().load_table_from_dataframe(results, metadata_table, job_config=job_config)
    job.result()
    print(f'Uploaded results to: `{metadata_table}`')

