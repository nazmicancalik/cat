import wandb
from datetime import datetime


def log(log):
    kwargs = {'entity': 'cgat'}
    project = "CGAT Convolution Based Geometry Aware Transformer for Point Cloud Segmentation"
    id = get_run_id()   
    wandb_logger = WandbLogger(
        project=project, name=id, id=id, dir='thesis/runs', **kwargs)
    return wandb_logger


def get_run_id():
    return datetime.now().strftime('%y%m%d%H%M%S%f')[:-4]
