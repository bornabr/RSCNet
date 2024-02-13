import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import argparse

from dataset import data_loader
from model import RSCNet

def get_cfg():
	parser = argparse.ArgumentParser(description='RSCNet')
	parser.add_argument('--root_dir', type=str, help='Root directory containing UT_HAR dataset folder')
	parser.add_argument('--debug', action='store_true', help='Enable debug mode')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for data loading')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
	parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
	parser.add_argument('--compression_rate', type=int, default=512, help='Compression rate for RSCNet')
	parser.add_argument('--num_frames', type=int, default=50, help='Number of frames for each sample')
	parser.add_argument('--recurrent_block', type=int, default=256, help='Number of hidden units for recurrent block')

	args = parser.parse_args()

	cfg = {
		'dataset': {
			'root_dir': args.root_dir,
			'batch_size': args.batch_size,
			'type': 'UT_HAR',
			'name': 'UT_HAR',
			'num_classes': 7,
			'input_shape': (-1, 1, 250, 90) 
		},
		'model': {
			'lr': 1e-2,
			'weight_decay': 1.5e-6,
			'momentum': 0.9,
			'epochs': args.epochs,
			'compression_rate': 500, # 1/500
			'expansion':1,
			'frames': args.num_frames,
			'RecurrentBlock': args.recurrent_block,
			'lambda1': 50
		},
		'seed': args.seed,
		'num_workers': args.num_workers,
		'validation_split': 0.2,
		'debug': False,
	}

	return cfg

def main(cfg):

	seed_everything(cfg['seed'], workers=True)

	train_loader, validation_loader, test_loader = data_loader(cfg['dataset'], validation_split=cfg['validation_split'])	

	if not cfg['debug']:
		wandb_logger = WandbLogger(project='RSCNet', log_model='all')
		checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max", save_last=True, save_top_k=3)

		wandb_logger.experiment.config.update(cfg)

	cfg['model']['batch_size'] = cfg['dataset']['batch_size']

	model = RSCNet(cfg['model'], dataset=cfg['dataset'])

	if cfg['debug']:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			fast_dev_run=2,
			# overfit_batches=1,
			detect_anomaly=True,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			# val_check_interval=0,
		)
	else:
		trainer = Trainer(
			devices="auto",
			accelerator="auto",
			detect_anomaly=True,
			max_epochs=cfg['model']['epochs'],
			log_every_n_steps=1,
			# val_check_interval=0,
			logger=wandb_logger,
			callbacks=[checkpoint_callback],
		)
	trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

	if not cfg['debug']:
		trainer.test(ckpt_path="best",dataloaders=test_loader)
	


if __name__ == '__main__':
	cfg = get_cfg()

	main(cfg)