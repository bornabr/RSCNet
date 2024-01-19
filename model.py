import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import wandb

from modules import Encoder
from modules import Decoder
from modules import Classifier
from modules import RecurrentBlock

class RSCNet(pl.LightningModule):
	def __init__(self, hparams, dataset):
		super(RSCNet, self).__init__()
		hparams['dataset'] = dataset
		self.save_hyperparameters(hparams)
		num_frames=self.hparams.frames

		# UT-HAR Dataset Configurations
		self.input_shape = (1,num_frames, 90)
		self.sequence_length = 250//num_frames
		self.embedding_size = 1*num_frames*90//self.hparams.compression_rate
		
		self.input_size = np.prod(self.input_shape)

		self.encoder = Encoder(self.input_shape)
		self.encoder_fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(int(self.input_size), int(self.input_size/self.hparams.compression_rate)),
		)

		if self.hparams.RecurrentBlock:
			self.recurrent_block = RecurrentBlock(self.embedding_size, self.hparams.RecurrentBlock)
			self.decoder_fc = nn.Sequential(
				nn.Linear(self.hparams.RecurrentBlock, int(self.input_size)),
				nn.Unflatten(1, (self.input_shape)),
			)
		else:
			self.decoder_fc = nn.Sequential(
				nn.Linear(int(self.input_size/self.hparams.compression_rate), int(self.input_size)),
				nn.Unflatten(1, (self.input_shape)),
			)

		self.decoder = Decoder(self.input_shape, self.hparams.expansion)

		if self.hparams.RecurrentBlock:
			self.classifier = Classifier(self.sequence_length*self.hparams.RecurrentBlock, self.hparams.dataset['num_classes'])
		else:
			self.classifier = Classifier(self.sequence_length*self.embedding_size, self.hparams.dataset['num_classes'])

		self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
		self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.hparams.dataset['num_classes'])
	
	def forward(self, x):
		batch_size = x.shape[0]
		num_frames=self.hparams.frames

		# UT-HAR specific data manipulations
		# batch x 1 x 250 x 90
		new_x = x.permute(0,2,1,3).contiguous()
		# batch x 250 x 1 x 90
		new_x = new_x.view(batch_size*self.sequence_length,num_frames,1,90)
		# (batch x t) x num_frames x 1 x 90
		new_x=new_x.permute(0,2,1,3)
		# (batch x t)x 1 x num_frames x 90

		z_e = self.encoder(new_x)

		c = self.encoder_fc(z_e)

		seq_c = c.view(batch_size, self.sequence_length, -1)
		
		if self.hparams.RecurrentBlock:
			seq_c_r_d, _ = self.recurrent_block(seq_c)
			seq_c_r_d = seq_c_r_d.contiguous()
			c_r_d = seq_c_r_d.view(batch_size*self.sequence_length, -1)
			z_d = self.decoder_fc(c_r_d)
		else:
			z_d = self.decoder_fc(c)

		# Recunstruction
		x_hat = self.decoder(z_d)

		# UT-HAR specific data manipulations
		new_x_hat = x_hat.permute(0,2,1,3).contiguous()
		# (batch * t) x num_frames x 1 x 90
		new_x_hat = new_x_hat.view(batch_size, 250, 1, 90)
		# batch x 250 x 1 x 90
		new_x_hat =new_x_hat.permute(0,2,1,3).contiguous()
		# batch x 1 x 250 x 90
	
		if self.hparams.RecurrentBlock:
			y_hat = self.classifier(seq_c_r_d)
		else:
			y_hat = self.classifier(seq_c)

		return new_x_hat, y_hat
	
	def training_step(self, batch, batch_idx):
		x, y = batch
	
		x_hat, y_hat = self(x)
	
		# Calculate loss
		loss, recon_loss, class_loss = self.loss(x, x_hat, y, y_hat)
		self.log('train_loss', loss)
		self.log('train_recon_loss', recon_loss)
		self.log('train_class_loss', class_loss)

		# Calculate accuracy
		self.train_accuracy.update(y_hat, y)

		# Calculate NMSE loss
		nmse = self.nmse(x, x_hat)
		self.log('train_nmse', nmse, on_step=False, on_epoch=True)

		return loss

	def on_train_epoch_end(self):
		train_acc = self.train_accuracy.compute()
		self.log('train_accuracy', train_acc)
		self.train_accuracy.reset()

	def validation_step(self, batch, batch_idx):
		if self.trainer.global_step == 0: 
			wandb.define_metric('val_accuracy', summary='max')
			wandb.define_metric('val_nmse', summary='min')
			wandb.define_metric('val_loss', summary='min')

		x, y = batch
	
		x_hat, y_hat = self(x)
	
		# Calculate loss
		loss, recon_loss, class_loss = self.loss(x, x_hat, y, y_hat)
		self.log('val_loss', loss)
		self.log('val_recon_loss', recon_loss)
		self.log('val_class_loss', class_loss)
		
		# Calculate accuracy
		self.val_accuracy.update(y_hat, y)

		# Calculate NMSE loss
		nmse = self.nmse(x, x_hat)
		self.log('val_nmse', nmse, on_step=False, on_epoch=True)

		return loss
	
	def test_step(self, batch, batch_idx):
			x, y = batch
		
			x_hat, y_hat = self(x)
		
			# Calculate loss
			loss, recon_loss, class_loss = self.loss(x, x_hat, y, y_hat)
			self.log('test_loss', loss)
			self.log('test_recon_loss', recon_loss)
			self.log('test_class_loss', class_loss)
			
			# Calculate accuracy
			self.test_accuracy.update(y_hat, y)

			# Calculate NMSE loss
			nmse = self.nmse(x, x_hat)
			self.log('test_nmse', nmse, on_step=False, on_epoch=True)

			return loss

	def on_test_epoch_end(self):
		test_acc = self.test_accuracy.compute()
		self.log('test_accuracy', test_acc)
		self.test_accuracy.reset()

	def on_validation_epoch_end(self):
		val_acc = self.val_accuracy.compute()
		self.log('val_accuracy', val_acc)
		self.val_accuracy.reset()

	def loss(self, x, x_hat, y, y_hat):
		# Reconstruction loss
		recon_loss = F.mse_loss(x_hat, x)
		# Classification loss
		class_loss = F.cross_entropy(y_hat, y)
		# Total loss
		loss = self.hparams.lambda1 * recon_loss + class_loss
		return loss, recon_loss, class_loss
	
	def nmse(self, x, x_hat):
		return 10 * torch.log10(torch.mean(torch.mean(torch.square(x-x_hat), dim=(1,2,3))/torch.mean(torch.square(x), dim=(1,2,3))))
	
	def configure_optimizers(self):
		optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
								momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.trainer.max_epochs)
		return [optim], [scheduler]
	