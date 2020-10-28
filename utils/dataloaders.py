import zarr
#level5 toolkit
from l5kit.dataset import AgentDataset
from l5kit.data import ChunkedDataset
from l5kit.rasterization import build_rasterizer

import torch
from torch.utils.data import DataLoader

def get_train_dl(cfg, dm):
    # training cfg
    train_cfg = cfg["train_data_loader"]
    # rasterizer
    rasterizer = build_rasterizer(cfg, dm)

    # dataloader
    train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                num_workers=train_cfg["num_workers"])
    return train_dataset, train_dataloader