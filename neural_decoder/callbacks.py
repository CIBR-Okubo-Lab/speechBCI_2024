from pytorch_lightning.callbacks import Callback
import time

class TimerCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.train_batch_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        train_time = time.time() - self.train_batch_time
        self.log("train/batchComputationTime", train_time, sync_dist=True, rank_zero_only=True, prog_bar=False)
        self.train_batch_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_time = time.time() - self.train_epoch_time
        self.log("train/epochComputationTime", train_time, sync_dist=True, rank_zero_only=True, prog_bar=False)
        self.train_epoch_time = 0
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.val_batch_time = time.time()
    
    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        val_time = time.time() - self.val_batch_time
        self.log("val/batchComputationTime", val_time, sync_dist=True, rank_zero_only=True, prog_bar=False)
        self.val_batch_time = 0