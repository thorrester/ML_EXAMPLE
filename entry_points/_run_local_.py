import os
import yaml
from inference.inference import inference_fn
#from inference.inference_rollup import prediction_rollup_fn
#from inference.classification_training import class_training_fn
#from inference.regression_training import reg_training_fn
from inference.training import training_fn
from commons import utils
from datetime import datetime
from pytz import timezone
import time
import git

git_branch = str(git.Repo(path=f'{os.getcwd()}', search_parent_directories=True).active_branch)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = f'{os.getcwd()}/roots.pem'

def run(train=True, inference=True):
    with open(os.path.join(PARENT_DIR, 'pipeline_config.local.yaml'), 'r') as f:
        pipeline_config = yaml.safe_load(f)
    print(pipeline_config)


    if train:
        training_fn(bq_project=pipeline_config['training']['bq_project'],
                    bq_dataset=pipeline_config['training']['bq_dataset'],
                    train_table_id=pipeline_config['training']['train_table_id'],
                    regression_labels=pipeline_config['training']['regression_labels'],
                    model_name=pipeline_config['training']['model_name'],
                    embedding_features= pipeline_config['training']['embedding_features'],
                    metadata_table_id=pipeline_config['training']['metadata_table_id'],
                    filter_list=pipeline_config['training']['filter_list'],
                    git_branch=git_branch,
                    batch_size=pipeline_config['training']['batch_size'],
                    epochs=pipeline_config['training']['epochs'],
                    data_limit=pipeline_config['training']['data_limit'],
                    hyperparameter_tuning=pipeline_config['training']['hyperparameter_tuning'])

    if inference:
        inference_fn(bq_project=pipeline_config['training']['bq_project'],
                     bq_dataset=pipeline_config['training']['bq_dataset'],
                     train_table_id=pipeline_config['training']['train_table_id'],
                     model_name=pipeline_config['training']['model_name'],
                     embedding_features=pipeline_config['training']['embedding_features'],
                     predict_table_id=pipeline_config['inference']['predict_table_id'],
                     metadata_table_id=pipeline_config['training']['metadata_table_id'],
                     prediction_output_table=pipeline_config['inference']['prediction_output_table'],
                     id_field=pipeline_config['inference']['id_field'],
                     bucket=pipeline_config['inference']['bucket'],
                     splits=pipeline_config['inference']['splits'],
                     filter_list=pipeline_config['training']['filter_list'],
                     git_branch=git_branch,
                     data_limit=pipeline_config['inference']['data_limit'],
                     )


if __name__ == "__main__":
    run()
