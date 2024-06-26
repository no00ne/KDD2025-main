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
    """
    计算预测模型的各种误差指标，包括 MSE, RMSE, MAE, 和 MAPE。
    
    参数:
    y_preds (list or np.array): 预测值
    y_trues (list or np.array): 真实值
    
    返回:
    dict: 包含 MSE, RMSE, MAE, 和 MAPE 的字典
    """
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)
    
    mse = np.mean((y_preds - y_trues) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_preds - y_trues))
    mape = np.mean(np.abs((y_trues - y_preds) / y_trues)) * 100  # 乘以100表示为百分比

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def calculate_loss(args, x, y, y_pre, scaler, w = None, treat =None, treat_base = None):
    if args.causal:
        batch_size = y.shape[0]
#         x = x.reshape(batch_size, args.reg_num, args.hidden_dim)
#         w = w.reshape(batch_size, args.reg_num, args.output_window)
#         
        y_pre = y_pre.reshape(y.shape)
        w_reshape = w.reshape(y.shape)
        loss = rwt_regression_loss(w_reshape, y, y_pre, scaler)  # 计算加权回归损失
        #mmd = IPM_loss(x, w, treat, args.k)
        
        #mmd = IPM_loss(x, torch.mean(w, dim = -1, keepdim = True), labels)

        #treat = treat.reshape(batch_size, args.reg_num, args.treat_hidden)
#         mmd = 0.0
#         labels = treat_label(treat, treat_base)
#         for i in range(args.output_window):
#             #print(x[i, :, :].shape, w[i, :, j:j+1].shape)
#             mmd += IPM_loss(x, w[:, i:i+1], labels) # 计算最大均值差异损失
#         mmd = 0.0
#         labels = treat_label(treat, treat_base, batch_size)
#         for i in range(batch_size):
#             for j in range(args.output_window):
#                 #print(x[i, :, :].shape, w[i, :, j:j+1].shape)
#                 mmd += IPM_loss(x[i, :, :], w[i, :, j:j+1], labels[i]) # 计算最大均值差异损失
        labels = treat_label(treat, treat_base)
        mmd = IPM_loss(x, torch.mean(w, dim = -1, keepdim = True), labels) # 计算最大均值差异损失
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
    #mae might be better than mse here <- hard to say...
    losses = ((y_pre.squeeze() - y.squeeze())**2) * w.squeeze()
    return losses.mean()
    #return losses.mean(dim = 0).sum()

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


# def treat_label(t):
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(t.detach().cpu().numpy())
#     data = X_scaled

# #     # 选择DBSCAN的参数
# #     # 使用K距离图（k-distance plot）来选择eps
# #     neighbors = NearestNeighbors(n_neighbors=10)
# #     neighbors_fit = neighbors.fit(data)
# #     distances, indices = neighbors_fit.kneighbors(data)
# #     distances = np.sort(distances, axis=0)
# #     distances = distances[:, 1]

# #     # 自动选择拐点
# #     # 计算一阶导数
# #     derivatives = np.diff(distances)
# #     # 计算二阶导数
# #     second_derivatives = np.diff(derivatives)
# #     # 找到二阶导数的最大值对应的点
# #     eps_index = np.argmax(second_derivatives) + 1
# #     eps = distances[eps_index]
    

#     # 使用DBSCAN进行聚类
#     min_samples = 10
#     eps = 0.05
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
#     labels = db.labels_
#     return labels

# def treat_label(data, base, batch_size):
#     data = data.cpu().detach().numpy()
#     base = base.cpu().detach().numpy()
#     similarities = cosine_similarity(data, base)
#     similarities = similarities.reshape(batch_size, 490)
#     batch_labels = []
#     for i in range(batch_size):
#         similarity = similarities[i]
#         bins = np.arange(0, 1.1, 0.1)
#         labels = np.digitize(similarity, bins, right=True)
#         unique_labels = np.unique(labels)
#         label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
#         new_labels = np.array([label_mapping[label] for label in labels])
#         batch_labels.append(new_labels)
#     return batch_labels

def treat_label(data, base):
    data = data.cpu().detach().numpy()
    base = base.cpu().detach().numpy()
    similarities = cosine_similarity(data, base)
    similarities = np.array([s[0] for s in similarities])
    bins = np.arange(0, 1.1, 0.1)
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

# def treat_order(treat):
#     data = treat.detach().cpu().numpy()
#     boundary_index = np.argmax(np.linalg.norm(data - np.mean(data, axis=0), axis=1))
#     similarity_matrix = cosine_similarity(data)
#     index = boundary_index  # 第一个数据点
#     similarities = similarity_matrix[index]
#     sorted_indices = np.argsort(-similarities)  # 从高到低排序
#     sorted_similarities = similarities[sorted_indices]
#     return sorted_indices

# def IPM_loss(x, w, idx, k = 20, rbf_sigma=1):
#     #x = x.squeeze(1)
# #     idx = treat_order(t)
#     splits = np.array_split(idx, k)
    
#     xw = x * w
#     sorted_x = x[idx]
#     sorted_xw = xw[idx]
    
#     split_x = torch.tensor_split(sorted_x, k)
#     split_xw = torch.tensor_split(sorted_xw, k)
    
#     loss = torch.zeros(k)
    
#     for i  in range(k):
#         A = split_xw[i]
#         tmp_loss = torch.zeros(k - 1)
#         idx = 0
#         for j in range(k):
#             if i == j:
#                 continue
#             B = split_x[j]
#             partial_loss = calculate_mmd(A, B, 1)
#             tmp_loss[idx] = partial_loss
#             idx += 1
#         loss[i] = tmp_loss.max()

#         return loss.mean()

# def IPM_loss(x, w, t, k = 20, rbf_sigma=1):
#     #x = x.squeeze(1)
#     idx = treat_order(t)
#     splits = np.array_split(idx, k)
    
#     xw = x * w
#     sorted_x = x[idx]
#     sorted_xw = xw[idx]
    
#     split_x = torch.tensor_split(sorted_x, k)
#     split_xw = torch.tensor_split(sorted_xw, k)
    
#     loss = torch.zeros(k)
    
#     for i  in range(k):
#         A = split_xw[i]
#         tmp_loss = torch.zeros(k - 1)
#         idx = 0
#         for j in range(k):
#             if i == j:
#                 continue
#             B = split_x[j]
#             partial_loss = calculate_mmd(A, B, 1)
#             tmp_loss[idx] = partial_loss
#             idx += 1
#         loss[i] = tmp_loss.max()

#         return loss.mean()

# def IPM_loss(x, w, l, rbf_sigma=1):
#     #x = x.squeeze(1)
#     l = l.long()
#     _, idx = torch.sort(l.long())
#     k = l.unique(return_counts=True)[0].shape[0]
#     if k == 1:
#         return 0.0
#     else:
#         xw = x * w
#         sorted_x = x[idx]
#         sorted_xw = xw[idx]
#         unique_label = l.unique(return_counts=True)[0].cpu().tolist()
#         loss = torch.zeros(k)  
#         for i in range(k):
#             A = sorted_xw[torch.where(l == unique_label[i])[0]]
#             tmp_loss = torch.zeros(k - 1)
#             idx = 0
#             for j in range(k):
#                 if i == j:
#                     continue
#                 B = sorted_x[torch.where(l == unique_label[j])[0]]
#                 partial_loss = calculate_mmd(A, B, 1)
#                 tmp_loss[idx] = partial_loss
#                 idx += 1
#             loss[i] = tmp_loss.max()

#             return loss.mean()