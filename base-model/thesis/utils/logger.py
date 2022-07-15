from pytorch_lightning.loggers import WandbLogger
from datetime import datetime

def get_logger(config):
    kwargs = {'entity': 'cgat'}
    project = "CGAT Convolution Based Geometry Aware Transformer for Point Cloud Segmentation"
    id = get_run_id() 
    run_name = config.training.run_name
    wandb_logger = WandbLogger(
        project=project, name=run_name, mode=config.training.wandb_mode, id=id, dir='thesis/runs', **kwargs)
    return wandb_logger


def get_run_id():
    return datetime.now().strftime('%y%m%d%H%M%S%f')[:-4]
