#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
import os
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from itertools import repeat
from tqdm import tqdm

sys.path.append('..')

from utils.nfcalib_utils import parse_args
from utils import data_utils

from src.mrartemev_nflib.flows import NormalizingFlowModel, InvertiblePermutation, Invertible1x1Conv, ActNorm, NSF_AR
from src.mrartemev_nflib.flows import MAF, AffineHalfFlow
from src.mrartemev_nflib.nn import ARMLP, MLP, DenseNet


def to_device(model, device, dims):
    model.to(device)
    model.prior = MultivariateNormal(torch.zeros(dims).to(device),
                                     torch.eye(dims).to(device))


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()

    # build vars
    model_name = f"{args.model}_{args.num_layers}_ind{args.index}"
    model_save_dir = f'../models/{args.data}/{model_name}/'
    os.makedirs(f'{model_save_dir}/checkpoints', exist_ok=True)

    # build data
    data = data_utils.data_mapping[args.data]()
    X_train_tensor = torch.from_numpy(data.trn.x).to(device)
    X_test_tensor = torch.from_numpy(data.tst.x).to(device)

    # build models
    prior = MultivariateNormal(torch.zeros(data.n_dims).to(device),
                               torch.eye(data.n_dims).to(device))

    flows = []
    for _ in range(args.num_layers):
        if args.model == 'MAF':
            flows.append(MAF(dim=data.n_dims, hidden_features=32, depth=5, base_network=ARMLP))
            flows.append(InvertiblePermutation(dim=data.n_dims))
        if args.model == 'SPLINE-AR':
            flows.append(ActNorm(dim=data.n_dims))
            flows.append(Invertible1x1Conv(dim=data.n_dims))
            flows.append(NSF_AR(dim=data.n_dims, K=8, B=3, hidden_features=32, depth=5, base_network=ARMLP))
        if args.model == 'GLOW':
            flows.append(ActNorm(dim=data.n_dims))
            flows.append(Invertible1x1Conv(dim=data.n_dims))
            flows.append(AffineHalfFlow(dim=data.n_dims, hidden_features=32, depth=5, base_network=DenseNet))
            flows.append(InvertiblePermutation(dim=data.n_dims))
        if args.model == 'RealNVP':
            flows.append(AffineHalfFlow(dim=data.n_dims, hidden_features=32, depth=5, base_network=DenseNet))
            flows.append(InvertiblePermutation(dim=data.n_dims))

    model = NormalizingFlowModel(prior, flows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    to_device(model, device, data.n_dims)

    # build dataloaders

    dloader = DataLoader(TensorDataset(X_train_tensor), batch_size=14000, shuffle=True, drop_last=True)
    test_dloader = DataLoader(TensorDataset(X_test_tensor), batch_size=2 ** 8, shuffle=True, drop_last=True)

    dloader = repeater(dloader)
    epoch_logpx = []

    for iteration in tqdm(range(args.num_epoch), position=0):
        model.train()

        # fit
        optimizer.zero_grad()
        logp_x = model.log_prob(next(dloader)[0])
        loss = -torch.mean(logp_x)
        loss.backward()

        # clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # step
        optimizer.step()

        # log
        epoch_logpx.append(-loss.item())

        # eval & save
        if (iteration + 1) % args.checkpoint_frequency == 0 or (iteration + 1) == args.num_epoch:
            with torch.no_grad():
                model.eval()
                eval_logp_x = np.mean(
                    [torch.mean(model.log_prob(batch[0].to(device))).item() for batch in test_dloader])
                train_logp_x = np.mean(epoch_logpx[-args.checkpoint_frequency:])
                print(f" {iteration}/{args.num_epoch}, val logpx: {eval_logp_x}, train logpx: {train_logp_x}")
                torch.save({'iteration': iteration,
                            'model.state_dict()': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_logp_x': eval_logp_x,
                            'train_logp_x': eval_logp_x
                            }, os.path.join(model_save_dir, 'checkpoints', f'{iteration}.checkpoint'))

    torch.save({'model.state_dict()': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(model_save_dir, 'final_model.checkpoint'))


if __name__ == '__main__':
    main()
