from typing import Any, Dict, Tuple
import copy
import torch
import torch.nn as nn
import pickle
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from src.utils.darcy_loss import LpLoss
from src.utils.weighted_acc_rmse import weighted_rmse_torch, weighted_acc_torch

class FDVarNetLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        img_grad: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        obs_vars: list,
        mean_path: str,
        std_path: str,
        clim_paths: list,
        var_idx_dir: str,
        loss: object,
        pred_ckpt: str,
        iter_update: list,
        nb_grad_update: list,
        lr_update: list,
        alpha: list,
        device: str,
    ) -> None:
        """Initialize a `FDVarNetLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        weights_dict = torch.load(self.hparams.pred_ckpt)['state_dict']
        load_weights_dict = {k[4:]: v for k, v in weights_dict.items()
                            if self.net.phi_r.state_dict()[k[4:]].numel() == v.numel()}
        self.net.phi_r.load_state_dict(load_weights_dict, strict=True)
        for param in self.net.phi_r.named_parameters():
            param[1].requires_grad = False

        self.run_device = device
        self.net.to(self.run_device)

        self.img_grad = img_grad

        # loss function
        self.criterion = self.hparams.loss
        # for averaging loss across batches
        self.train_loss = MeanMetric()

        self.val_loss = MeanMetric()

        self.test_loss = MeanMetric()

        self.var_idx = [k for k in np.load(self.hparams.var_idx_dir)]
        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
        self.clims = [np.load(clim_paths[i]) for i in range(len(clim_paths))]

        mult = np.ones(self.net.phi_r.out_chans)
        for i in range(self.net.phi_r.out_chans):
            mult[i] = self.std[self.var_idx[i]] * mult[i]
        self.mult = torch.tensor(mult, dtype=torch.float32, requires_grad=False)

        clims = np.ones([3, self.net.phi_r.out_chans, self.net.phi_r.img_size[0], self.net.phi_r.img_size[1]])
        for j in range(len(clim_paths)):
            for i in range(mult.shape[0]):
                clims[j, i] = ((self.clims[j][self.var_idx[i]] - self.mean[self.var_idx[i]]) / self.std[
                    self.var_idx[i]])[0, -1, :-1] * clims[j, i]
        self.clims = torch.tensor(clims, dtype=torch.float32, requires_grad=False)

        self.obs_vars = obs_vars
        mask = np.zeros([1, 4, self.net.phi_r.out_chans, self.net.phi_r.img_size[0], self.net.phi_r.img_size[1]])
        for i in range(self.net.phi_r.out_chans):
            if i in self.obs_vars:
                mask[:, :, i] = 1
        self.mask = torch.tensor(mask, requires_grad=False)

        self.weight = torch.tensor(np.zeros([1, 24, 1, 1]), dtype=torch.float32, requires_grad=False)

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse_u10_best = MinMetric()
        self.val_rmse_v10_best = MinMetric()
        self.val_rmse_z500_best = MinMetric()
        self.val_rmse_t850_best = MinMetric()
        self.val_acc_u10_best = MaxMetric()
        self.val_acc_v10_best = MaxMetric()
        self.val_acc_z500_best = MaxMetric()
        self.val_acc_t850_best = MaxMetric()

    def set_weight(self, val_loader):
        self.weight = torch.tensor(np.ones([1, 24, 1, 1]), dtype=torch.float32, requires_grad=False)

    def reset_mask(self, obs_vars):
        self.obs_vars = obs_vars
        mask = np.zeros([1, 4, self.net.phi_r.out_chans, self.net.phi_r.img_size[0], self.net.phi_r.img_size[1]])
        for i in range(self.net.phi_r.out_chans):
            if i in self.obs_vars:
                mask[0, :, i] = 1
        self.mask = torch.tensor(mask, requires_grad=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, y, mask)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], phase: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        xt, xb, obs_wind, obs_r, obs_t = batch
        xb = xb.to(xt.device, dtype=xt.dtype)
        new_obs = torch.zeros_like(xb).to(xt.device, dtype=xt.dtype)
        new_obs = torch.stack([new_obs, new_obs, new_obs, new_obs], dim=1)
        for i in range(self.net.phi_r.out_chans):
            if self.var_idx[i] == "u10":
                new_obs[:, :, i] = obs_wind[:, :, 0]
            elif self.var_idx[i] == "v10":
                new_obs[:, :, i] = obs_wind[:, :, 1]
            elif self.var_idx[i] == "r@50":
                new_obs[:, :, i] = obs_r[:, :, 0]
            elif self.var_idx[i] == "r@500":
                new_obs[:, :, i] = obs_r[:, :, 1]
            elif self.var_idx[i] == "r@850":
                new_obs[:, :, i] = obs_r[:, :, 2]
            elif self.var_idx[i] == "r@1000":
                new_obs[:, :, i] = obs_r[:, :, 3]
            elif self.var_idx[i] == "t@50":
                new_obs[:, :, i] = obs_t[:, :, 0]
            elif self.var_idx[i] == "t@500":
                new_obs[:, :, i] = obs_t[:, :, 1]
            elif self.var_idx[i] == "t@850":
                new_obs[:, :, i] = obs_t[:, :, 2]
            elif self.var_idx[i] == "t@1000":
                new_obs[:, :, i] = obs_t[:, :, 3]

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(xb, requires_grad=True)
            xa, hidden_new, cell_new, normgrad = self.forward(state,
                                                              new_obs.detach() * self.mask.to(xt.device, dtype=xt.dtype),
                                                              self.mask.to(xt.device, dtype=xt.dtype))
            if (phase == 'val') or (phase == 'test'):
                xa = xa.detach()
            g_xt = self.img_grad(xt[:, 0])
            g_xa = self.img_grad(xa)
            loss = self.hparams.alpha[0] * self.criterion(xa, xt[:,0]) + self.hparams.alpha[1] * self.criterion(g_xa, g_xt)

        torch.cuda.empty_cache()
        return loss, xa, xt[:,0]

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch, "train")

        # update and log metrics
        self.train_loss(loss.to("cpu"))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch, "val")

        val_rmse = self.mult.to(preds.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
        val_acc = weighted_acc_torch(preds - self.clims[1].to(preds.device, dtype=preds.dtype),
                                     targets - self.clims[1].to(preds.device, dtype=preds.dtype))

        # update and log metrics
        self.val_loss(loss.to("cpu"))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_u10", val_rmse[self.var_idx.index('u10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_v10", val_rmse[self.var_idx.index('v10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_z500", val_rmse[self.var_idx.index('z@500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse_t850", val_rmse[self.var_idx.index('t@850')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_u10", val_acc[self.var_idx.index('u10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_v10", val_acc[self.var_idx.index('v10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_z500", val_acc[self.var_idx.index('z@500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_t850", val_acc[self.var_idx.index('t@850')], on_step=False, on_epoch=True, prog_bar=True)
        return {'rmse': val_rmse, 'acc': val_acc}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        "Lightning hook that is called when a validation epoch ends."
        val_rmse, val_acc = 0, 0
        for out in validation_step_outputs:
            val_rmse += out['rmse'] / len(validation_step_outputs)
            val_acc += out['acc'] / len(validation_step_outputs)

        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss.to("cpu"))  # update best so far val loss
        val_rmse_u10 = val_rmse[self.var_idx.index('u10')]  # get current val rmse of u10
        self.val_rmse_u10_best(val_rmse_u10)  # update best so far val rmse of u10
        val_rmse_v10 = val_rmse[self.var_idx.index('v10')]  # get current val rmse of v10
        self.val_rmse_v10_best(val_rmse_v10)  # update best so far val rmse of v10
        val_rmse_z500 = val_rmse[self.var_idx.index('z@500')]  # get current val rmse of z500
        self.val_rmse_z500_best(val_rmse_z500)  # update best so far val rmse of z500
        val_rmse_t850 = val_rmse[self.var_idx.index('t@850')]  # get current val rmse of t850
        self.val_rmse_t850_best(val_rmse_t850)  # update best so far val rmse of t850
        val_acc_u10 = val_acc[self.var_idx.index('u10')]  # get current val acc of u10
        self.val_acc_u10_best(val_acc_u10)  # update best so far val acc of u10
        val_acc_v10 = val_acc[self.var_idx.index('v10')]  # get current val acc of v10
        self.val_acc_v10_best(val_acc_v10)  # update best so far val acc of v10
        val_acc_z500 = val_acc[self.var_idx.index('z@500')]  # get current val acc of z500
        self.val_acc_z500_best(val_acc_z500)  # update best so far val acc of z500
        val_acc_t850 = val_acc[self.var_idx.index('t@850')]  # get current val acc of t850
        self.val_acc_t850_best(val_acc_t850)  # update best so far val acc of t850

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/rmse_u10_best", self.val_rmse_u10_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/rmse_v10_best", self.val_rmse_v10_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/rmse_z500_best", self.val_rmse_z500_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/rmse_t850_best", self.val_rmse_t850_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_u10_best", self.val_acc_u10_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_v10_best", self.val_acc_v10_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_z500_best", self.val_acc_z500_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/acc_t850_best", self.val_acc_t850_best.compute(), sync_dist=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch, "test")

        # update and log metrics
        self.test_loss(loss.to("cpu"))
        test_rmse = self.mult.to(preds.device, dtype=preds.dtype) * weighted_rmse_torch(preds, targets)
        test_acc = weighted_acc_torch(preds - self.clims[2].to(preds.device, dtype=preds.dtype),
                                      targets - self.clims[2].to(preds.device, dtype=preds.dtype))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse_u10", test_rmse[self.var_idx.index('u10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse_v10", test_rmse[self.var_idx.index('v10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse_z500", test_rmse[self.var_idx.index('z@500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse_t850", test_rmse[self.var_idx.index('t@850')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_u10", test_acc[self.var_idx.index('u10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_v10", test_acc[self.var_idx.index('v10')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_z500", test_acc[self.var_idx.index('z@500')], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_t850", test_acc[self.var_idx.index('t@850')], on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer([{'params': self.net.model_Grad.parameters()},
                                {'params': self.net.model_VarCost.parameters()}])
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    _ = FourCastNetLitModule(None, None, None, None)
