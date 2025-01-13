import torch
from torch import nn
from models import SDEUNet
from base_segmodel import _SegmentationModel


class SegmentationModel(_SegmentationModel):
	def __init__(self, **hparams):
		super().__init__(**hparams)
		self.diff_loss = nn.BCELoss()
	
	def get_model(self):
		self.automatic_optimization = False
		return SDEUNet(self.in_features, self.out_features)
			
	def configure_optimizers(self):
		optimizer_F = torch.optim.Adam([
			{'params': self.cnn.encs.parameters()},
			{'params': self.cnn.decs.parameters()},
			{'params': self.cnn.outconv.parameters()}
		], lr=self.hparams.lr)
		optimizer_G = torch.optim.Adam(self.cnn.diffusion.parameters(), lr=self.hparams.lr)
		return optimizer_F, optimizer_G
	
	
	def forward(self, x, train_diffusion=False):
		return self.cnn(x, train_diffusion)
	
	def prepare_diff_labels(self, y, val=None):
		labels = [torch.full(
			(y.shape[0], self.cnn.diffusion.diffusion_blocks[i].out_channels, y.shape[2]//2**i, y.shape[3]//2**i), 
			val, device=y.device) for i in range(len(self.cnn.diffusion.diffusion_blocks))]
		return labels

	def training_step(self, batch, batch_idx):
		x, y = batch['x'], batch['y']

		optimizer_F, optimizer_G = self.optimizers()
		# optimize F
		optimizer_F.zero_grad()
		y_hat = self(x)
		loss = self.loss(y_hat, y)
		self.manual_backward(loss)
		optimizer_F.step()
		self.train_losses.append([self.current_epoch, loss.item()])

		# optimize G
		labels = self.prepare_diff_labels(y, 0.)
		optimizer_G.zero_grad()
		predict_in = self(x, train_diffusion=True)
		loss_diff_in = sum([self.diff_loss(predict_in[i], labels[i]) for i in range(len(labels))])
		self.manual_backward(loss_diff_in)
		
		labels = self.prepare_diff_labels(y, 1.)
		inputs_out = torch.randn_like(x) + x
		predict_out = self(inputs_out, train_diffusion=True)
		loss_diff_out = sum([self.diff_loss(predict_out[i], labels[i]) for i in range(len(labels))])
		self.manual_backward(loss_diff_out)
		optimizer_G.step()

		self.log("train/loss", loss)
		self.log("train/loss_diff_in", loss_diff_in)
		self.log("train/loss_diff_out", loss_diff_out)
		self.log("train/rmse", self.denorm_rmse(loss), prog_bar=True)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch['x'], batch['y']
		y_hat = self(x)
		loss = self.loss(y_hat, y)
		self.val_losses.append([self.current_epoch, loss.item()])
		self.log("val/loss", loss)
		self.log("val/rmse", self.denorm_rmse(loss), prog_bar=True)

		labels = self.prepare_diff_labels(y, 0.)
		predict_in = self(x, train_diffusion=True)
		loss_diff_in = sum([self.diff_loss(predict_in[i], labels[i]) for i in range(len(labels))])
		
		labels = self.prepare_diff_labels(y, 1.)
		inputs_out = 2*torch.randn_like(x) + x
		predict_out = self(inputs_out, train_diffusion=True)
		loss_diff_out = sum([self.diff_loss(predict_out[i], labels[i]) for i in range(len(labels))])
		self.log("val/loss_diff_in", loss_diff_in)
		self.log("val/loss_diff_out", loss_diff_out)

		for metric in self.metrics:
			self.log(f"val/{metric.__name__}", metric(y_hat, y))


# class SmallSegmentationModel(SegmentationModel):
# 	def get_model(self):
# 		self.automatic_optimization = False
# 		return SDEUNet(self.in_features, self.out_features, channels=[64, 128, 256, 512])