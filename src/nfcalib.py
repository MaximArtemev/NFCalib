#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Working with: ", device)


### Setup

parser = argparse.ArgumentParser(description='Easy args NFlib')

parser.add_argument('--data', action="store", type=str)
parser.add_argument('--model', action="store", type=str)
parser.add_argument('--index', action='store', type=int, default='1')
parser.add_argument('--num_layers', action="store", type=int, default=5)
parser.add_argument('--num_epoch', action="store", type=int, default=int(3 * 1e6))
parser.add_argument('--lr', action="store", type=float, default=0.0001)
parser.add_argument('--checkpoint_frequency', action="store", type=int, default=200000)
args = parser.parse_args()

data_name = args.data # 'POWER'
model_type = args.model # 'MAF'

num_layers = args.num_layers # 5
num_iters = args.num_epoch # 20000

model_name = f"{model_type}_{num_layers}_ind{args.index}"

model_save_dir = f'../models/{data_name}/{model_name}/'
os.makedirs(f'{model_save_dir}/checkpoints', exist_ok=True)


### Data

from utils import data_utils

data_mapping = {'BSDS300': data_utils.BSDS300,
                'GAS': data_utils.GAS,
                'MINIBOONE': data_utils.MINIBOONE,
                'POWER': data_utils.POWER,
                'HEPMASS': data_utils.HEPMASS}
data = data_mapping[data_name]()
dim = data.n_dims


X_train_tensor = torch.from_numpy(data.trn.x).to(device)
X_test_tensor = torch.from_numpy(data.tst.x).to(device)


from torch.distributions import MultivariateNormal

prior = MultivariateNormal(torch.zeros(data.n_dims).to(device),
                           torch.eye(data.n_dims).to(device))


from src.mrartemev_nflib.flows import NormalizingFlowModel, InvertiblePermutation, Invertible1x1Conv, ActNorm, NSF_AR
from src.mrartemev_nflib.flows import MAF, AffineHalfFlow
from src.mrartemev_nflib.nn import ARMLP, MLP


flows = []
for _ in range(num_layers):
    if model_type == 'MAF':
        flows.append(MAF(dim=data.n_dims, base_network=ARMLP))
        flows.append(InvertiblePermutation(dim=data.n_dims))
    if model_type == 'SPLINE-AR':
        flows.append(ActNorm(dim=data.n_dims))
        flows.append(Invertible1x1Conv(dim=data.n_dims))
        flows.append(NSF_AR(dim=data.n_dims, K=8, B=3, hidden_features=32, depth=1, base_network=MLP))
    if model_type == 'GLOW':
        flows.append(ActNorm(dim=data.n_dims))
        flows.append(Invertible1x1Conv(dim=data.n_dims))
        flows.append(AffineHalfFlow(dim=data.n_dims, hidden_features=32, base_network=MLP))
        flows.append(InvertiblePermutation(dim=data.n_dims))
    if model_type == 'RealNVP':
        flows.append(AffineHalfFlow(dim=data.n_dims, hidden_features=32, base_network=MLP))
        flows.append(InvertiblePermutation(dim=data.n_dims))

lr = args.lr # 0.0001
        
model = NormalizingFlowModel(prior, flows).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def to_device(model, device):
    model.to(device)
    model.prior = MultivariateNormal(torch.zeros(data.n_dims).to(device),
                                     torch.eye(data.n_dims).to(device))

to_device(model, device)


from torch.utils.data import Dataset, DataLoader, TensorDataset
from itertools import repeat

dloader = DataLoader(TensorDataset(X_train_tensor), batch_size=2**12,
                     shuffle=True, drop_last=True)
test_dloader = DataLoader(TensorDataset(X_test_tensor), batch_size=2**8,
                          shuffle=True, drop_last=True)

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

dloader = repeater(dloader)


from tqdm import tqdm


checkpoint_frequency = args.checkpoint_frequency

epoch_logpx = []


for iteration in tqdm(range(len(epoch_logpx), num_iters), position=0):
    batch = next(dloader)[0]
    model.train()

    # fit
    optimizer.zero_grad()
    logp_x = model.log_prob(batch)
    loss = -torch.mean(logp_x)
    loss.backward()

    # clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    # step
    optimizer.step()

    # log
    epoch_logpx.append(-loss.item())
    

        
    # eval & save
    if iteration % checkpoint_frequency == 0 and iteration != 0:     
        with torch.no_grad():
            model.eval()
            eval_logp_x = np.mean([torch.mean(model.log_prob(batch[0].to(device))).item() for batch in test_dloader])
            train_logp_x = np.mean(epoch_logpx[-checkpoint_frequency:])
            print(f" {iteration}/{num_iters}, val logpx: {eval_logp_x}, train logpx: {train_logp_x}")
            torch.save({'iteration': iteration,
                        'model.state_dict()': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_logp_x': eval_logp_x,
                        'train_logp_x': eval_logp_x
                       }, os.path.join(model_save_dir, 'checkpoints', f'{iteration}.checkpoint'))

eval_logp_x = np.mean([torch.mean(model.log_prob(batch[0].to(device))).item() for batch in test_dloader])
train_logp_x = np.mean(epoch_logpx[-checkpoint_frequency:])

torch.save({'iteration': iteration,
            'model.state_dict()': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_logp_x': eval_logp_x,
            'train_logp_x': eval_logp_x
           }, os.path.join(model_save_dir, 'final_model.checkpoint'))


### Model eval

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.nf import CalibratedModel, neg_log_likelihood
from src.nf.classifiers import train_catboost_clf
from scipy.special import logsumexp


def batched_sample(model, n, batch_size=256):
    generated = []
    for _ in range(n // batch_size):
        generated_batch = model.sample(batch_size)
        generated.append(generated_batch.cpu().detach())
    if n % batch_size != 0:
        generated_batch = model.sample(n % batch_size)
        generated.append(generated_batch.cpu().detach())
    generated = torch.cat(generated, dim=0)
    assert n == len(generated)
    return generated


model.eval()
model.sample_n = lambda n: batched_sample(model, n)

to_device(model, 'cpu')

n = min(100000, X_test_tensor.shape[0])

print('Model test LL ', torch.mean(model.log_prob(X_test_tensor[:n].cpu())).item())

clf_ds_train = np.row_stack([
    np.column_stack([X_train_tensor[:n].cpu().detach().numpy(), np.ones(n).reshape(-1, 1)]),
    np.column_stack([model.sample_n(n).cpu().detach().numpy(), np.zeros(n).reshape(-1, 1)])
]).astype(np.float32)

clf_ds_test = np.row_stack([
    np.column_stack([X_test_tensor[:n].cpu().detach().numpy(), np.ones(n).reshape(-1, 1)]),
    np.column_stack([model.sample_n(n).cpu().detach().numpy(), np.zeros(n).reshape(-1, 1)])
]).astype(np.float32)


clf = CatBoostClassifier(
    5000, eval_metric='AUC',
    metric_period=1000,
).fit(
    clf_ds_train[:, :-1], clf_ds_train[:, -1],
    eval_set=(clf_ds_test[:, :-1], clf_ds_test[:, -1])
)


calibrated_model = CalibratedModel(clf, model, logit=True)

samples = model.sample_n(n).cpu().detach().cpu().numpy()
clf_preds = clf.predict(samples, prediction_type='RawFormulaVal')
calibration_constant = logsumexp(clf_preds) - np.log(len(clf_preds))

print("Was", -neg_log_likelihood(model, X_test_tensor.cpu().detach()))
print("Calibrated", -neg_log_likelihood(calibrated_model, X_test_tensor.cpu().detach()) - calibration_constant)


