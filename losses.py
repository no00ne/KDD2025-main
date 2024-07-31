import torch

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import pickle as pk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def validation_loss(y_preds, y_trues):
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    
    mse = np.mean((y_preds - y_trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_preds - y_trues))
    mape = np.mean(np.abs((y_trues - y_preds) / y_trues)) * 100

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def calculate_loss(args, x, y, y_pre, scaler, w = None, treat =None, treat_base = None):
    if args.causal:         
        y_pre = y_pre.reshape(y.shape)
        w_reshape = w.reshape(y.shape)
        loss = rwt_regression_loss(w_reshape, y, y_pre, scaler)

        labels = treat_label(treat, treat_base)
        mmd = IPM_loss(x, torch.mean(w, dim = -1, keepdim = True), labels)
        return mmd + loss
    else:
        y_pre = y_pre.reshape(y.shape)
        return F.mse_loss(y.squeeze(), y_pre.squeeze())

def validate_metric(y, y_pre, scaler):
    return mape_loss(scaler.inverse_transform(y.squeeze().cpu()), scaler.inverse_transform(y_pre.squeeze().cpu()))
    
def args_to_dict(args):
    args_to_dict = dict(filter(lambda x: x[0] in list_of_args,
                              args.__dict__.items()))
    return args_to_dict

def mape_loss(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

def rwt_regression_loss(w, y, y_pre, scaler):
    losses = ((y_pre.squeeze() - y.squeeze())**2) * w.squeeze()
    return losses.mean()

def pdist2sq(A, B):
    # Compute the squared Euclidean distance between each pair of vectors in A and B
    A_square = torch.sum(A ** 2, dim=1, keepdim=True)
    B_square = torch.sum(B ** 2, dim=1, keepdim=True)
    
    # Use einsum to compute pairwise distances
    AB_product = torch.einsum('ik,jk->ij', A, B)
    
    # Calculate the squared Euclidean distance
    D = A_square + B_square.T - 2 * AB_product
    D = torch.clamp(D, min=0.0)  # Ensure there are no negative distances due to numerical errors
    
    return D

def rbf_kernel(A, B, rbf_sigma=1):
    rbf_sigma = torch.tensor(rbf_sigma)
    return torch.exp(-pdist2sq(A, B) / torch.square(rbf_sigma) *.5)

def calculate_mmd(A, B, rbf_sigma=1):
    Kaa = rbf_kernel(A, A, rbf_sigma)
    Kab = rbf_kernel(A, B, rbf_sigma)
    Kbb = rbf_kernel(B, B, rbf_sigma)
    mmd = Kaa.mean() - 2 * Kab.mean() + Kbb.mean()
    return mmd


def treat_label(data, base):
    data = data.cpu().detach().numpy()
    base = base.cpu().detach().numpy()
    similarities = cosine_similarity(data, base)
    similarities = np.array([s[0] for s in similarities])
    bins = np.arange(-1, 1.1, 0.1)
    labels = np.digitize(similarities, bins, right=True)
    unique_labels = np.unique(labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    new_labels = np.array([label_mapping[label] for label in labels])

    return new_labels

def IPM_loss(x, w, labels, rbf_sigma=8):
    labels = np.array(labels)
    k = len(set(labels))
    if k == 1:
            return 0.0
    else:
        xw = x * w
        split_x = [x[labels == i] for i in set(labels)]
        split_xw = [xw[labels == i] for i in set(labels)]

        loss = torch.zeros(k)
    
        for i in range(k):
            A = split_xw[i]
            tmp_loss = torch.zeros(k - 1)
            idx = 0
            for j in range(k):
                if i == j:
                    continue
                B = split_x[j]
                partial_loss = calculate_mmd(A, B, rbf_sigma)
                tmp_loss[idx] = partial_loss
                idx += 1
            loss[i] = tmp_loss.max()

        return loss.mean()
