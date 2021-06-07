from commons import utils, bq_client
from typing import List, Callable, Dict
import joblib
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf


def inference_reg_fn(num_data: np.array,
                     bay_data: np.array,
                     bay_str_data: pd.DataFrame,
                     prediction_output_table: str,
                     model: Callable,
                     batch_size: int = 1024,
                     ):
    """Runs inference on a model against the given data from BigQuery

     :param batch_size: Batch size for inference
     :param predict_data: Data for prediction
     :param bay_str_data: Name of gcs bucket
     :param prediction_output_table: Table to save results to
     :param model: model for prediction
     """

    pred_data = tf.data.Dataset.from_tensor_slices({'BAY_EMBED': bay_data,
                                                    'NUM_DATA': num_data,
                                                    })

    pred_data = pred_data.batch(batch_size, drop_remainder=False).prefetch(1)
    pred_keys = ['oos', 'out', 'low', 'opp_val']
    preds_dict = {**dict.fromkeys(pred_keys, np.array([]))}

    for x_batch in pred_data:
        q50_oos, q50_out, q50_low, q50_opp_val = model(x_batch, training=False)

        preds_dict['oos'] = np.append(preds_dict['oos'], q50_oos.numpy().flatten(), axis=0)
        preds_dict['out'] = np.append(preds_dict['out'], q50_out.numpy().flatten(), axis=0)
        preds_dict['low'] = np.append(preds_dict['low'], q50_low.numpy().flatten(), axis=0)
        preds_dict['opp_val'] = np.append(preds_dict['opp_val'], q50_opp_val.numpy().flatten(), axis=0)

    bay_str_data = bay_str_data.copy().reset_index()
    bay_str_data['OOS_PREDICTION'] = preds_dict['oos']
    bay_str_data['OUT_PREDICTION'] = preds_dict['out']
    bay_str_data['LOW_PREDICTION'] = preds_dict['low']
    bay_str_data['OPP_VAL_PREDICTION'] = np.exp(preds_dict['opp_val'])
    bay_str_data['DATE'] = datetime.datetime.now()

    # Create results data
    job = bq_client.get_client().load_table_from_dataframe(bay_str_data, prediction_output_table)
    job.result()


