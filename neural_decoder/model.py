from .augmentations import GaussianSmoothing
from edit_distance import SequenceMatcher
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim


class GRUDecoder(pl.LightningModule):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays,
        dropout,
        strideLen,
        kernelLen,
        gaussianSmoothWidth,
        whiteNoiseSD,
        constantOffsetSD,
        bidirectional,
        l2_decay,
        lrStart,
        lrEnd,
        momentum,
        nesterov,
        gamma,
        stepSize,
        nBatch,
        output_dir,
    ):
        super().__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(
            torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.constantOffsetSD = constantOffsetSD
        self.whiteNoiseSD = whiteNoiseSD
        self.loss_ctc = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True)
        self.l2_decay = l2_decay
        self.lrStart = lrStart
        self.lrEnd = lrEnd
        self.momentum = momentum
        self.nesterov = nesterov
        self.gamma = gamma
        self.stepSize = stepSize
        self.nBatch = nBatch
        self.output_dir = output_dir
        self.testLoss = []
        self.testCER = []

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(
                self, "inpLayer" + str(x),
                nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(
                hidden_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out

    def training_step(self, batch, batch_idx):
        X, y, X_len, y_len, dayIdx = batch

        # Noise augmentation is faster on GPU
        if self.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=self.device) * self.whiteNoiseSD

        if self.constantOffsetSD > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=self.device)
                * self.constantOffsetSD
            )

        # Compute prediction error
        pred = self.forward(X, dayIdx)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)
        schedulers = self.lr_schedulers()
        cur_lr = schedulers.optimizer.param_groups[-1]["lr"]
        self.log_dict(
            {"train/predictionLoss": loss, "train/learning_rate": cur_lr},
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True,
            rank_zero_only=True)
        return {"loss": loss, "pred": pred, "y": y}

    def validation_step(self, batch, batch_idx):
        X, y, X_len, y_len, testDayIdx = batch

        total_edit_distance = 0
        total_seq_length = 0

        pred = self.forward(X, testDayIdx)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        adjustedLens = ((X_len - self.kernelLen) / self.strideLen).to(
            torch.int32
        )
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                pred[iterIdx, 0: adjustedLens[iterIdx], :].clone().detach(),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(
                y[iterIdx][0: y_len[iterIdx]].cpu().detach()
            )

            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)

        avgDayLoss = loss
        cer = total_edit_distance / total_seq_length

        self.log_dict({"val/predictionLoss": avgDayLoss, "val/ser": cer},
                      sync_dist=True, prog_bar=True, rank_zero_only=True)

        return {"loss": loss, "pred": pred, "y": y}

    def configure_optimizers(self):

        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lrStart,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.l2_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.stepSize,
            gamma=self.gamma,
        )

        freq = self.trainer.accumulate_grad_batches or 1

        return {
            "optimizer": optimizer,
            "lr_scheduler":
            {
                "scheduler": scheduler,
                "interval": "step",
                "freqeuncy": freq
            }
        }
