from src.train_models.transfer_learning import main
import subprocess
subprocess.run('source activate tensorflow2_p36' && 'python main.py {{params.model_name}} {{params.train_path}} {{params.test_path}} {{params.val_path}}' && 'source deactivate', shell=True)