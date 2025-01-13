from pathlib import Path
from utils import io
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision('medium')
NUM_WORKERS = 0

from torch.utils.data import DataLoader
from utils.datasets import NWPDataset
from models import UNet

import pytorch_lightning as pl

class _SegmentationModel(pl.LightningModule):
	def __init__(self, **hparams):
		super().__init__()
		self.save_hyperparameters()
		self.load_data()
		self.cnn = self.get_model()

		def M_MSE(y_hat, y, reduction='mean'):
			return F.mse_loss(y_hat*self.mask, y*self.mask, reduction=reduction)
		self.loss = M_MSE
		self.denorm_rmse = lambda loss: (loss*(self.case_study_max**2)).sqrt().item()
		self.metrics = []
		self.test_predictions = []

		self.train_losses = []
		self.val_losses = []
		self.save_preds = False
	
	def get_model(self):
		return UNet(self.in_features, self.out_features, dropout=self.hparams.mcdropout)

	def forward(self, x, idx=0):
		return self.cnn(x)

	def load_data(self):
		case_study_max, available_models, train_dates, val_dates, test_dates, indices_one, indices_zero, mask, nx, ny = io.get_casestudy_stuff(
			self.hparams.input_path, n_split=self.hparams.n_split, case_study=self.hparams.case_study, ispadded=True,
			seed=self.hparams.seed
		)

		# Removing intense events from training dates
		all_dates = pd.read_csv(self.hparams.input_path/'allevents_dates.csv', sep=';')
		intense_events = all_dates[all_dates['INTENSE']]['DATA'].values
		train_dates = np.array([date for date in train_dates if date not in intense_events])

		self.x_train, self.y_train, in_features, out_features = io.load_data(self.hparams.input_path, train_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_val, self.y_val, in_features, out_features = io.load_data(self.hparams.input_path, val_dates, case_study_max, indices_one, indices_zero, available_models)
		self.x_test, self.y_test, in_features, out_features = io.load_data(self.hparams.input_path, test_dates, case_study_max, indices_one, indices_zero, available_models)

		self.train_dates, self.val_dates, self.test_dates = train_dates, val_dates, test_dates

		self.case_study_max = case_study_max
		mask = torch.from_numpy(mask).float().to(self.device)
		self.register_buffer('mask', mask) # This makes sure self.mask is on the same device as the model

		self.in_features = in_features
		self.out_features = out_features
	
	def train_dataloader(self):
		train_dataset = NWPDataset((torch.from_numpy(self.x_train), 
							  		torch.from_numpy(self.y_train).unsqueeze(1),
							  		torch.from_numpy(self.train_dates)))
		return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_WORKERS)
		
	def val_dataloader(self):
		val_dataset = NWPDataset((torch.from_numpy(self.x_val), 
								  torch.from_numpy(self.y_val).unsqueeze(1),
								  torch.from_numpy(self.val_dates)))
		return DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=NUM_WORKERS)
	
	def test_dataloader(self):
		test_dataset = NWPDataset((torch.from_numpy(self.x_test),
									torch.from_numpy(self.y_test).unsqueeze(1),
									torch.from_numpy(self.test_dates)))
		return DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=NUM_WORKERS)
	
	def training_step(self, batch, idx):
		x, y = batch['x'], batch['y']
		y_hat = self.forward(x, idx)
		loss = self.loss(y_hat, y)
		self.train_losses.append([self.current_epoch, loss.item()])
		self.log("train/loss", loss)
		self.log("train/rmse", self.denorm_rmse(loss), prog_bar=True)
		return loss

	def validation_step(self, batch, idx):
		x, y = batch['x'], batch['y']
		y_hat = self.forward(x, idx)
		loss = self.loss(y_hat, y)
		self.val_losses.append([self.current_epoch, loss.item()])
		self.log("val/loss", loss)
		self.log("val/rmse", self.denorm_rmse(loss), prog_bar=True)

		for metric in self.metrics:
			self.log(f"val_{metric.__name__}", metric(y_hat, y))
	
	def save_prediction(self, tensor, name):
		"""Save a tensor as a CSV file in self.output_dir"""
		pd.DataFrame(tensor.squeeze().cpu().numpy()).to_csv(self.output_dir/f"{name}.csv", index=False, header=False)

	def test_step(self, batch, idx):
		x, y = batch['x'], batch['y']
		if self.hparams.mcdropout or self.hparams.network_model == 'sde_unet':
			y_hat = torch.zeros_like(y)
			for _ in range(10):
				out_i = self.forward(x, idx)
				if self.save_preds:
					for batch_idx in range(x.shape[0]):
						if isinstance(batch, dict):
							bdate = batch['ev_date'][batch_idx]
						else:
							bdate = batch[batch_idx]['ev_date']
						self.save_prediction(out_i[batch_idx], f"pred_{bdate}_{idx}.csv")
				y_hat += out_i
			y_hat /= 10
		else:
			y_hat = self.forward(x, idx)
			if self.save_preds:
				for batch_idx in range(x.shape[0]):
					self.save_prediction(y_hat[batch_idx], f"pred_{batch['ev_date'][batch_idx]}_0.csv")
					
		loss = self.loss(y_hat, y)
		self.log("test/rmse", self.denorm_rmse(loss))
		self.log_images(x, y, y_hat, idx)
		# self.test_predictions.append(y_hat)

		for channel in range(x.shape[1]):
			loss_ch = self.loss(x[:, channel:channel+1, :, :], y)
			self.log(f"test/rmse NWP {channel}", self.denorm_rmse(loss_ch))

		for metric in self.metrics:
			self.log(f"test/test_{metric.__name__}", metric(y_hat, y))
	
	# def on_train_end(self):
	# 	import seaborn as sns
	# 	import pandas as pd
	# 	import matplotlib.pyplot as plt
	# 	fig, ax = plt.subplots()
	# 	train_losses = pd.DataFrame(self.train_losses, columns=['epoch', 'loss'])
	# 	val_losses = pd.DataFrame(self.val_losses, columns=['epoch', 'loss'])
	# 	sns.lineplot(data=train_losses, x='epoch', y='loss', label='train')
	# 	sns.lineplot(data=val_losses, x='epoch', y='loss', label='valid')
	# 	plt.legend()
	# 	plt.yscale('log')
	# 	fig.savefig(Path(self.logger.log_dir)/"losses.png")

	def eval(self):
		return self.cnn.eval()
	
	@torch.no_grad()
	def multiple_eval(self, x, num_forward_passes=10):
		""" Function to get multiple evaluations of tensor x and returning the mean and variance of the predictions"""
		predictions = []
		for _ in range(num_forward_passes):
			predictions.append(self.forward(x))
		predictions = torch.stack(predictions, dim=0)
		return predictions.mean(dim=0), predictions.var(dim=0)

	def log_images(self, features, masks, logits_, idx):
		if not hasattr(self, 'log_dir') and self.logger is not None:
			self.log_dir = Path(self.logger.log_dir)
		respath = self.log_dir / "test_images"
		respath.mkdir(exist_ok=True)
		tensor_to_img = lambda t: (t.detach().cpu().numpy()[0] * 255)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(features, masks, logits_)):
			Image.fromarray(tensor_to_img(y_pred)).convert('L').save(respath/f"{idx}_{img_idx}_pred.png")
			Image.fromarray(tensor_to_img(y_true)).convert('L').save(respath/f"{idx}_{img_idx}_gt.png")
			for i in range(image.shape[0]):
				Image.fromarray(tensor_to_img(image[i:i+1])).convert('L').save(respath/f"{idx}_{img_idx}_m{i}.png")

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
	
	def get_monte_carlo_predictions(self, forward_passes=10):
		""" Function to get the monte-carlo samples and uncertainty estimates
		through multiple forward passes

		Parameters
		----------
		data_loader : object
			data loader object from the data loader module
		forward_passes : int
			number of monte-carlo samples/forward passes
		"""
		assert hasattr(self.cnn, 'dropout') and self.cnn.dropout > 0, 'Model does not have dropout layers'

		dropout_predictions = []
		for i in range(forward_passes):
			predictions = []
			self.cnn.eval()
			with torch.no_grad():
				for batch in self.test_dataloader():
					x, y, ev_date = batch['x'], batch['y'], batch.get('ev_date')
					x = x.to('cuda')
					output = self.cnn(x)
					predictions.append(output)
				predictions = torch.cat(predictions, dim=0) # shape (n_samples, C, H, W)
			dropout_predictions.append(predictions)
		dropout_predictions = torch.stack(dropout_predictions, dim=0) # shape (n_forward_passes, n_samples, C, H, W)
		

		# Calculating stats across multiple MCD forward passes 
		mean = dropout_predictions.mean(dim=0)
		variance = dropout_predictions.var(dim=0)

		# Calculating entropy across multiple MCD forward passes 
		# entropy = -torch.sum(mean * torch.log(mean + 1e-6), axis=-1)

		# # Calculating mutual information across multiple MCD forward passes 
		# mutual_info = entropy - torch.mean(torch.sum(-dropout_predictions * torch.log(dropout_predictions + 1e-6),
		# 									dim=-1), dim=0)
		
		y_all = torch.cat([batch['y'] for batch in self.test_dataloader()], dim=0)
		loss = self.loss(mean, y_all.cuda())

		self.log("test/rmse", self.denorm_rmse(loss))
		self.log("test/variance", variance.mean().item())
		print(f"MCD RMSE", self.denorm_rmse(loss))
		print(f"MCD variance", variance.mean().item())
		# print(f"MCD entropy", entropy.mean().item())
		# print(f"MCD mutual info", mutual_info.mean().item())

		for metric in self.metrics:
			print(f"MCD {metric.__name__}", metric(mean, y))

	def eval_proba(self, lv_thresholds=[1, 5, 10, 20, 50, 100, 150], forward_passes=20, save_dir=None):
		"""
		Perform an evaluation loop for the probabilistic forecasts. The function have to store, for each threshold and each image,
		the probability for the models (the MCD samples) to be above the threshold. The function will then compute the Brier score
		"""

		assert hasattr(self.cnn, 'dropout') and self.cnn.dropout > 0, 'Model does not have dropout layers'

		if save_dir is None:
			save_dir = Path(self.logger.log_dir)
		save_dir.mkdir(exist_ok=True)

		probabilities = {lv: [] for lv in lv_thresholds}
		y_all = torch.cat([batch['y'] for batch in self.test_dataloader()], dim=0) * self.case_study_max
		enames_all = torch.cat([batch['ev_date'] for batch in self.test_dataloader()], dim=0)
		
		self.cnn.eval()
		with torch.no_grad():
			for i in range(forward_passes):
				predictions = []
				for batch in self.test_dataloader():
					x = batch['x'].cuda()
					output = self.cnn(x)
					predictions.append(output)
				predictions = torch.cat(predictions, dim=0)
				predictions = predictions * self.case_study_max
				for lv in lv_thresholds:
					probabilities[lv].append((predictions > lv).float())
			for lv in lv_thresholds:
				probabilities[lv] = torch.stack(probabilities[lv], dim=0).mean(dim=0)

		# Calculating Brier score
		brier_scores = {}
		input_models_brier_score = {}
		x_all = torch.cat([batch['x'] for batch in self.test_dataloader()], dim=0) * self.case_study_max
		for lv in lv_thresholds:
			(save_dir/str(lv)).mkdir(exist_ok=True)
			# save the probability results for each threshold under the logger directory / <threshold> / pred{ev_date}.csv
   
			for i, ename in enumerate(enames_all):
				# probabilities[lv] has shape (n_samples, C, H, W) to get the current image, we need to index the first dimension
				# img = probabilities[lv][i].cpu().numpy()
				pd.DataFrame(probabilities[lv][i].squeeze().cpu().numpy()).to_csv(save_dir/f"{lv}/pred{ename}.csv", index=False, header=False)


			brier_scores[lv] = ((probabilities[lv] - y_all.cuda().gt(lv).float())**2).mean().item()
			prob_input_models = (x_all > lv).float()
			input_models_brier_score[lv] = ((prob_input_models - y_all.gt(lv).float())**2).mean().item()
			print(f"Brier score for threshold {lv} mm: {brier_scores[lv]:.4f}")
			print(f">Brier score for input models: {input_models_brier_score[lv]:.4f}\n")
		
		import matplotlib.pyplot as plt
		import seaborn as sns
		sns.set_style("whitegrid")

		plt.plot(lv_thresholds, [brier_scores[lv] for lv in lv_thresholds], label='Brier score')
		plt.plot(lv_thresholds, [input_models_brier_score[lv] for lv in lv_thresholds], label='Input models Brier score')
		plt.xlabel('Threshold (mm)')
		plt.ylabel('Brier score')
		plt.legend()
		plt.savefig("brier_scores.png")
		# plt.savefig(Path(self.logger.log_dir)/"brier_scores.png")
		