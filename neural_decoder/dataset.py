import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
    
class SpeechDataModule(pl.LightningDataModule):
    def __init__(self, loaded_data, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loaded_data = loaded_data
    
    def setup(self, stage):
        self.train_ds = SpeechDataset(self.loaded_data["train"])
        self.test_ds = SpeechDataset(self.loaded_data["test"])

    def _padding(self, batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._padding,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._padding,
        )