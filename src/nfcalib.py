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
parser.add_argument('--num_layers', action="store", type=int, default=5)
parser.add_argument('--num_epoch', action="store", type=int, default=20000)
parser.add_argument('--lr', action="store", type=int, default=0.0001)
parser.add_argument('--index', action='store', type=int, default='0')
args = parser.parse_args()

data_name = args.data # 'POWER'
model = args.model # 'MAF'

num_layers = args.num_layers # 5
num_epoch = args.num_epoch # 20000
if model == 'SPLINE-AR':
    num_epoch = num_epoch / 2

model_name = f"{model}_{num_layers}_ind{args.index}"

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


X_train_tensor = torch.from_numpy(data.trn.x)
X_test_tensor = torch.from_numpy(data.tst.x)


from torch.distributions import MultivariateNormal

prior = MultivariateNormal(torch.zeros(data.n_dims).to(device),
                           torch.eye(data.n_dims).to(device))


from src.mrartemev_nflib.flows import NormalizingFlowModel, InvertiblePermutation, Invertible1x1Conv, ActNorm, NSF_AR
from src.mrartemev_nflib.flows import MAF, AffineHalfFlow
from src.mrartemev_nflib.nn import ARMLP, MLP


flows = []
for _ in range(num_layers):
    if model == 'MAF':
        flows.append(MAF(dim=data.n_dims, base_network=ARMLP))
        flows.append(InvertiblePermutation(dim=data.n_dims))
    if model == 'SPLINE-AR':
        flows.append(ActNorm(dim=data.n_dims))
        flows.append(Invertible1x1Conv(dim=data.n_dims))
        flows.append(NSF_AR(dim=data.n_dims, K=8, B=3, hidden_features=32, depth=1, base_network=MLP))
    if model == 'GLOW':
        flows.append(ActNorm(dim=data.n_dims))
        flows.append(Invertible1x1Conv(dim=data.n_dims))
        flows.append(AffineHalfFlow(dim=data.n_dims, hidden_features=32, base_network=MLP))
        flows.append(InvertiblePermutation(dim=data.n_dims))
    if model == 'RealNVP':
        flows.append(AffineHalfFlow(dim=data.ndims))
        flows.append(InvertiblePermutation(dim=data.n_dims))

lr = args.lr # 0.0001
if model == 'SPLINE-AR':
    lr *= 10
        
model = NormalizingFlowModel(prior, flows).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def to_device(model, device):
    model.to(device)
    model.prior = MultivariateNormal(torch.zeros(data.n_dims).to(device),
                                     torch.eye(data.n_dims).to(device))

to_device(model, device)


from torch.utils.data import Dataset, DataLoader, TensorDataset

dloader = DataLoader(TensorDataset(X_train_tensor), batch_size=2**12,
                     shuffle=True, drop_last=True, num_workers=6)
test_dloader = DataLoader(TensorDataset(X_test_tensor), batch_size=2**12,
                          shuffle=True, drop_last=True, num_workers=6)


from tqdm import tqdm


plot_freq = 1500

epoch_losses = []
train_losses = []

for epoch in tqdm(range(len(epoch_losses), num_epoch), position=0):
    for ind, batch in enumerate(dloader):
        model.train()
        
        # fit
        optimizer.zero_grad()
        logp_x = model.log_prob(batch[0].to(device))
        loss = -torch.mean(logp_x)
        loss.backward()
        
        # clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
        # step
        optimizer.step()

        # log
        train_losses.append(-loss.item())

    # eval
    with torch.no_grad():
        model.eval()
        logp_x = np.mean([torch.mean(model.log_prob(batch[0].to(device))).item() for batch in test_dloader])
        epoch_losses.append([np.mean(train_losses), logp_x])
    
    # show
    if epoch % plot_freq == 0 and epoch != 0:     
   	    print(f"{epoch}/{num_epoch} val loss: {np.mean(np.array(epoch_losses)[-plot_freq:, 1])}")
        
    # checkpoint
    if epoch % plot_freq == 0 and epoch != 0:
        torch.save(model.state_dict(),
                   os.path.join(model_save_dir, 'checkpoints', f'{epoch}_model.state_dict'))
        torch.save(optimizer.state_dict(),
                   os.path.join(model_save_dir, 'checkpoints', f'{epoch}_optimizer.state_dict'))

torch.save(model.state_dict(), os.path.join(model_save_dir, 'model.state_dict'))
torch.save(optimizer.state_dict(), os.path.join(model_save_dir, 'optimizer.state_dict'))


### Model eval

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.nf import CalibratedModel, neg_log_likelihood
from src.nf.classifiers import train_catboost_clf
from scipy.special import logsumexp


model.eval()
model.sample_n = lambda x: model.sample(x)

to_device(model, 'cpu')


n = min(100000, X_test_tensor.shape[0])

print('Model test LL ', torch.mean(model.log_prob(X_test_tensor[:n])).item())

clf_ds_train = np.row_stack([
    np.column_stack([X_train_tensor[:n].detach().numpy(), np.ones(n).reshape(-1, 1)]),
    np.column_stack([model.sample(n).detach().numpy(), np.zeros(n).reshape(-1, 1)])
]).astype(np.float32)

clf_ds_test = np.row_stack([
    np.column_stack([X_test_tensor[:n].detach().numpy(), np.ones(n).reshape(-1, 1)]),
    np.column_stack([model.sample(n).detach().numpy(), np.zeros(n).reshape(-1, 1)])
]).astype(np.float32)


clf = CatBoostClassifier(
    5000, eval_metric='AUC',
    metric_period=1000,
).fit(
    clf_ds_train[:, :-1], clf_ds_train[:, -1],
    eval_set=(clf_ds_test[:, :-1], clf_ds_test[:, -1])
)


calibrated_model = CalibratedModel(clf, model, logit=True)

samples = model.sample(n).detach().cpu().numpy()
clf_preds = clf.predict(samples, prediction_type='RawFormulaVal')
calibration_constant = logsumexp(clf_preds) - np.log(len(clf_preds))

print("Was", -neg_log_likelihood(model, X_test_tensor))
print("Calibrated", -neg_log_likelihood(calibrated_model, X_test_tensor) - calibration_constant)






