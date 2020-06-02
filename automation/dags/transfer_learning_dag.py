from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from automation.manifest.manifest_prod import config as cfg
from automation.manifest.manifest_prod import env as master_env
from src.shared.util import dict_merge
from automation.factories.demo_factory import createTransferLearningTasks

def createTransferLearningDAG(params):
    today = datetime,now().strftime("%Y%m%d")
    print("Creating Transfer Learning DAG for ", today)

    default_cfg = master_env.get('default', {})
    env_cfg = master_env.get(params_get('env', 'default'), {})
    env = dict_merge(default_cfg, env_cfg)

    default_args = {
        'owner': 'Dr Cadx',
        'depends_on_past':  False,
        'start_date': datetime(2020,1,1),
        'email': cfg.get('email', ''),
        'email_on_failure': True,
        'email_on_retry': True,
        'retries': cfg.get('airflow_retries', 1),
        'retry_delay': timedelta(minutes=5),
        'provide_context':True
    }

    dag_name = 'Transfer_Learning_Modeling'
    print('Initialize DAG: {}'.format(dag_name))

    dag = DAG(dag_name,
              default_args=default_args)

    dag.catchup = False

    print('Retrieve The Transfer Learning Tasks')
    dag_tasks =  createTransferLearningTasks(params, default_args)

    print('Sequencing the Transfer Learning Tasks')
    dag >> dag_tasks.transfer_learning >> dag_tasks.validation

    return dag         
