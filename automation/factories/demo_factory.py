from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.models import Variable
from src.train_models import transfer_learning
from automation.manifest.manifest_prod import config as cfg
from automation.bash_scripts import 


class TransferLearningTask():
    def __init__(self,
                 transfer_learning,
                 validation):
        self.task_transfer_learning = transfer_learning
        self.Task_validation = validation

model_name = Variable.get('model_name')
train_path = Variable.get('tl_train_path')
test_path = Variable.get('tl_test_path')
val_path = Variable.get('tl_val_path')

def createTransferLearningTasks(params, default_args):
    train_model_tl = PyrhonOperator(
        task_id = 'train_model',
        python_callable = transfer_learning,
        provide_context=True,
        email_on_success=True,
        email_on_failure=True,
        email=cfg['email'],
        params=params,
        default_args=default_args
    )

    validation = BashOperator(
        task_id = 'validation',
        bash_command = 'to_follow',
        params=params,
        default_args = default_args
    )

    return TransferLearningTask(transfer_learning, validation)