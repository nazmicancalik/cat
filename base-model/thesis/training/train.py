import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from modules.classification_module import LightningClassificationModule
from modules.segmentation_module import LightningSegmentationModule
from datasets.data_modules import ScanNetSegmentationDataModule, ShapeNetDataModule, ModelNetDataModule
from datasets.modelnet import ModelNetDataset
from utils.config import get_config, get_project_root
from utils.logger import get_logger

ROOT = get_project_root()

def train_tune(config):
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(save_last=False, save_top_k=1, monitor='val_avg_instance_acc', mode='max')
    
    # Added for hyperparameter tuning
    callbacks = [lr_monitor, model_checkpoint]
    
    if config.training.tuning:
        tune_callback = TuneReportCallback(
            {
                "loss":"val_final_loss",
                "mean_accuracy":"val_avg_instance_acc"
            },
            on="validation_end"
        )
        callbacks.append(tune_callback)


    # dm = ScanNetSegmentationDataModule(config)
    # net = LightningSegmentationModule(config)

    #dm = ShapeNetDataModule(config)
    dm = ModelNetDataModule(config)
    net = LightningClassificationModule(config)

    trainer = pl.Trainer(
        logger=get_logger(config),
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        gpus=config.training.gpus,
        callbacks=callbacks,    
        fast_dev_run=config.training.fast_dev_run,
        log_every_n_steps=config.training.log_every_n_steps
    )

    # Train the net
    trainer.fit(net, datamodule=dm)
    
    # Test the net
    trainer.test(model=net, datamodule=dm, ckpt_path="best")

def train_asha_scheduler(config_):
    # Here change the parameters, that needs to be optimized on
    config = config_

    """
    Here are the hyperparameters decided.
    Just update the existing config file to include the fields as hyperparameter.
    Current Hyperparameter:
        - learning rate
        - convolution block
        - sequence length
        - k: number of neighbours
        - transformer feature embedding dimension
        - transformer encoder fc hidden ratio
        - classifier hidden dimension
    """
    
    #config.training.batch_size = tune.choice([32])
    
    config.training.learning_rate= tune.loguniform(1e-4,1e-1)
    config.modules.transformer.convolution.enabled = tune.choice([True, False])
    config.modules.transformer.seq_len = tune.choice([128,256,512])
    config.modules.DGCNN.k = tune.choice([10,20,25])
    config.modules.transformer.embed_dim = tune.choice([128,256,512,768])
    config.modules.transformer.encoder.fc_hidden_ratio = tune.choice([2,4])
    config.modules.transformer.classifier_hidden_dim = tune.choice([512,1024])

    # Change the run name according to the config, needed for wandb
    """
    doesnt really work because the name is decided before the parameters are decided
    config.training.run_name = f"lr_{str(config.training.learning_rate)}\
                                _conv_{str(config.modules.transformer.convolution.enabled)}\
                                _seq_len_{str(config.modules.transformer.seq_len)}\
                                _k_{str(config.modules.DGCNN.k)}\
                                _embed_dim_{str(config.modules.transformer.embed_dim)}\
                                _fc_ratio_{str(config.modules.transformer.encoder.fc_hidden_ratio)}\
                                _cls_hidden_{str(config.modules.transformer.classifier_hidden_dim)}\
                                "
    """
    scheduler = ASHAScheduler(
        max_t=config.training.asha_max_epochs,
        grace_period=config.training.asha_grace_period,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "mean_accuracy", "training_iteration"]
    )

    train_fn_with_parameters = tune.with_parameters(train_tune)
    resources_per_trial = {"cpu": 8, "gpu": 1}

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        scheduler=scheduler,
        local_dir=os.path.join(ROOT,"hyperparameter_optimization_results"),
        num_samples=config.training.hyperparameter_optimization_num_samples,
        progress_reporter=reporter,
        max_failures=2,
        name="experiment_01_04_22")
    print("Best hyperparameters found were: ", analysis.best_config)


def train(config):
    if config.training.tuning:
        train_asha_scheduler(config)
    else:
        train_tune(config)
