from tqdm import tqdm
import numpy as np
from utils.dataloaders import get_train_dl
from utils.environment import set_environ
from config.data_config import data_config
from config.train_config import training_cfg
from vis.visutils import get_sample_img
from models.lyftnet import LyftModel

import torch
from torch import nn,optim

def train(device, data_root=data_config['DATA_ROOT'], config=training_cfg):
    dm = set_environ(config,data_root)
    train_ds, train_dl = get_train_dl(config, dm)
    img = get_sample_img(train_dl)
    model = LyftModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction="none")

    tr_it = iter(train_dl)
    progress_bar = tqdm(range(config["train_params"]["max_num_steps"]))

    losses_train = []

    for _ in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dl)
            data = next(tr_it)
        model.train()
        torch.set_grad_enabled(True)
        
        # forward pass
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)
        
        outputs = model(inputs).reshape(targets.shape)
        loss = criterion(outputs, targets)

        # not all the output steps are valid, but we can filter them out from the loss using availabilities
        loss = loss * target_availabilities
        loss = loss.mean()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
            
        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Training model on GPU')
    else:
        print('Training model on CPU')
    #print(f'Training model on device: {device}')
    train(device)


    
