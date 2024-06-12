from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
import torch
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# local modules
from neural_decoder.callbacks import TimerCallback
from neural_decoder.dataset import SpeechDataModule
from neural_decoder.model import GRUDecoder

torch.set_float32_matmul_precision("medium")

def trainModel(args):

    # set seed
    pl.seed_everything(args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    
    # output directory
    if 'wandb' in args and args.wandb.enabled:
        args["outputDir"] = os.path.join(args["outputDir"], wandb.run.name)
    else:
        args["outputDir"] = os.path.join(args["outputDir"],f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    if os.getenv("LOCAL_RANK", '0') == '0': 
        os.makedirs(args["outputDir"], exist_ok=True)
        with open(os.path.join(args["outputDir"], "args"), "wb") as file:
            pickle.dump(args, file)
    
    # load data
    with open(args["datasetPath"], "rb") as handle:
        loadedData = pickle.load(handle)

    # data module
    dm = SpeechDataModule(loadedData, args["batchSize"], args["numWorkers"])

    # tensorboard logger
    logger = TensorBoardLogger(args["outputDir"], name="torch_dist_v0")
    
    # model
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"], 
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        l2_decay=args["l2_decay"], 
        lrStart=args["lrStart"], 
        lrEnd=args["lrEnd"], 
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        nBatch=args["nBatch"], 
        output_dir=args["outputDir"]
    )
    
    # checkpoint callback
    checkpointCallback = ModelCheckpoint(filename=args["outputDir"]+"/modelWeights", monitor="val/ser", mode="min", save_top_k=1, every_n_train_steps=None)
    checkpointCallback.FILE_EXTENSION = ""


    # trainer
    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        min_steps=1,
        max_steps=10000,
        accelerator=args["accelerator"],
        devices=args["devices"],
        precision=args["precision"],
        num_nodes=1,
        log_every_n_steps=1, 
        val_check_interval=100,
        check_val_every_n_epoch=None, 
        callbacks=[checkpointCallback, TimerCallback()]
    )

    # train
    trainer.fit(model, dm)

def loadModel(modelWeightPath, nInputLayers=24, device="cuda"):

    # load pl model
    pl_model = torch.load(modelWeightPath, map_location=device)
    
    # load hyperparameters
    args = pl_model["hyper_parameters"]
    state_dict = pl_model["state_dict"]

    model = GRUDecoder(
        neural_dim=args["neural_dim"],
        n_classes=args["n_classes"],
        hidden_dim=args["hidden_dim"],
        layer_dim=args["layer_dim"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"], 
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        l2_decay=args["l2_decay"], 
        lrStart=args["lrStart"], 
        lrEnd=args["lrEnd"], 
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        nBatch=args["nBatch"], 
        output_dir=args["output_dir"]
    ).to(device)

    model.load_state_dict(state_dict)
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config_1")
def main(cfg):
    
    conf_name = HydraConfig.get().job.config_name
    if 'wandb' in cfg and cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        print(f"config: {conf_name[conf_name.index('_')+1:]}")
        wandb_config['hyperparam_setting'] = conf_name[conf_name.index("_")+1:]
        run = wandb.init(**cfg.wandb.setup,
                            config=wandb_config,
                            name=f"run_{datetime.now().strftime('%Y%m%d-%H%M%S-')}" + wandb_config['hyperparam_setting'], 
                            sync_tensorboard=True)

    trainModel(cfg)
    if 'wandb' in cfg and cfg.wandb.enabled:
        run.finish()

if __name__ == "__main__":
    main()