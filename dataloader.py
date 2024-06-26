import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle as pk
from normalization import *
from datetime import datetime, timedelta

def calculate_hour(hours_passed, start_date = '2023-04-01'):
    # 将起始日期转换为 datetime 对象
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    # 计算目标日期时间
    target_datetime = start_datetime + timedelta(hours=hours_passed)
    
    # 检查是否为周末
    is_weekend = target_datetime.weekday() >= 5  # 5 是周六，6 是周日
    
    # 计算目标小时（0-23 或 24-47）
    if is_weekend:
        result_hour = target_datetime.hour + 24
    else:
        result_hour = target_datetime.hour
    
    return result_hour

def calculate_week_hour(hours_passed, start_date = '2023-04-01'):
    # 将起始日期转换为 datetime 对象
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    # 计算目标日期时间
    target_datetime = start_datetime + timedelta(hours=hours_passed)
    
    # 计算从周开始的小时数（0-167）
    days_since_week_start = target_datetime.weekday()  # 星期一为0，星期日为6
    result_hour = days_since_week_start * 24 + target_datetime.hour
    
    return result_hour

def mean_along_dim0(sparse_tensor):
    # 获取稀疏张量的索引和值
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    # 获取张量的形状
    shape = sparse_tensor.shape
    assert shape[0] > 0, "The size of dim 0 must be greater than 0"

    # 初始化累积张量和计数张量
    accumulated_values = torch.zeros((shape[1], shape[2]), dtype=values.dtype)
    count_values = torch.zeros((shape[1], shape[2]), dtype=values.dtype)

    # 累积所有非零元素并计数
    for idx, val in zip(indices.t(), values):
        accumulated_values[idx[1], idx[2]] += val
        count_values[idx[1], idx[2]] += 1

    # 计算均值，避免除以零
    mean_values = accumulated_values / torch.clamp(count_values, min=1)
    mean_values[count_values == 0] = 0  # 将计数为0的位置均值设为0

    # 构建新的稀疏张量
    mean_indices = mean_values.nonzero(as_tuple=False).t()
    mean_values_sparse = mean_values[mean_values != 0]
    mean_sparse_tensor = torch.sparse_coo_tensor(mean_indices, mean_values_sparse, (shape[1], shape[2]))

    return mean_sparse_tensor

def combine_sparse_tensors(tensor_list):
    batch_size = len(tensor_list)
    V = tensor_list[0].size(0)
    
    combined_indices = []
    combined_values = []

    # 遍历列表中的每个稀疏张量
    for i, tensor in enumerate(tensor_list):
        indices = tensor._indices()
        values = tensor._values()
        
        # 添加批次维度信息
        batch_indices = torch.cat([torch.full((1, indices.size(1)), i, dtype=torch.long), indices], dim=0)
        
        # 将当前张量的 indices 和 values 添加到组合列表中
        if len(combined_indices) == 0:
            combined_indices = batch_indices
            combined_values = values
        else:
            combined_indices = torch.cat((combined_indices, batch_indices), dim=1)
            combined_values = torch.cat((combined_values, values))

    # 新的张量大小
    size = torch.Size([batch_size, V, V])

    # 创建合并后的稀疏张量
    combined_tensor = torch.sparse_coo_tensor(combined_indices, combined_values, size=size)
    
    return combined_tensor

class CausalDataset(Dataset):
    def __init__(self, args, data, mode = 'test'):
        self.args = args
        self.data = data
        self.mode = mode
        self.device = self.args.device
#         if self.mode == 'train' and self.args.causal:
#             self.labels = self.get_label()

    def get_label(self):
        self.args.logger.info('Conducting KMeas for treatments in training dataset...')
        alltreat = []
        for d in self.data:
            treat = d[4]
            for i in treat:
                if len(treat[i]) != 0:
                    alltreat.append(treat[i])
        alltreat = np.array(alltreat)
        
        self.args.logger.info('Starting news text embedding clustering...')
        kmeans = KMeans(n_clusters=self.args.n_clusters, random_state=0)
        kmeans.fit(alltreat)
        labels = kmeans.labels_
        self.args.logger.info('Clustering done!')
        
        n = 0
        all_labels = []
        for d in self.data:
            label = []
            treat = d[4]
            for i in treat:
                if len(treat[i]) != 0:
                    label.append(labels[n] + 1)
                    n += 1
                else:
                    label.append(0)
            all_labels.append(label)
            
        return all_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        indice, x, y, t, treat, adj = self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3], self.data[idx][4], self.data[idx][5]
        x = x.unsqueeze(2)
        if self.args.causal:
            treat = torch.FloatTensor(treat) / 100
            mask = (treat.sum(dim=1) == 0)
        else:
            treat = torch.zeros((x.shape))
            mask = (treat.sum(dim=1) == 0)
        
#         if self.args.causal:
#             treat_dict = self.data[idx][4]
#             for i in treat_dict:
#                 if len(treat_dict[i]) != 0:
#                     treat[i, :] = torch.FloatTensor(treat_dict[i])
        
        
        
        #x, y, t, adj, treat, mask = x.to(self.device), y.to(self.device), t.to(self.device), adj.to(self.device), treat.to(self.device), mask.to(self.device)
        
        return x, y, t, adj, treat, mask, indice
#         if self.mode == 'train':
#             if self.args.causal:
#                 label = torch.LongTensor(self.labels[idx])
#             else:
#                 label = torch.zeros((self.args.reg_num))
#             return x, y, t, adj, treat, label, mask, indice
#         else:
#             return x, y, t, adj, treat, mask, indice
        
class CausalDatasetPreloader():
    def __init__(self, args):
        self.args = args
        self.args.logger.info('Starting data preprocessing...')
        self.prepareAll()

    def prepareAll(self):
        self.get_adjacency()
        self.get_flows()
        if self.args.causal:
            self.get_treatment()
        self.get_indices()
        self.check_data()
        
    def check_data(self):
        assert len(self.adjacency_matrix) == len(self.flows)
    
    def data_scaler(self, data):
        self.args.logger.info(f'Data shape for scaler creation is ({data.shape[0]}, {data.shape[1]})')
        data = torch.FloatTensor(data)
        scaler = StandardScaler(mean = data.mean(dim = 0), std = data.std(dim = 0))
        return scaler
        
    def data_split(self):
        self.args.logger.info('Splitting datasets...')
        length = len(self.indices)
        train_len = int(length * self.args.training_rate)
        valid_len = int(length * self.args.validate_rate)
        train_uplimit = self.indices[train_len][-1]
        self.scaler = self.data_scaler(self.flows[:train_uplimit, :])
        #self.args.logger.info('Data scaler mean: {}, std: {})'.format(self.scaler.mean, self.scaler.std))
        data = []
        for index, indice in enumerate(tqdm(self.indices, desc="Processing")):
            x = self.scaler.transform(torch.FloatTensor(self.flows[indice[:self.args.input_window], :]))
            
            y = self.scaler.transform(torch.FloatTensor(self.flows[indice[self.args.input_window:], :]))
            #y = self.scaler.transform(torch.FloatTensor(self.flows[indice[self.args.input_window:], :]).to(self.args.device))
            
#             d = indice[self.args.input_window-1] // self.args.tim_num
#             last_t = indice[self.args.input_window-1] % self.args.tim_num
#             if self.args.causal:
#                 treat = self.treat_dict[d * self.args.tim_num + last_t]
#             else:
#                 treat = None
            if self.args.causal:
#                 treats = []
#                 for k in range(self.args.input_window):
#                     idx = indice[k]
#                     treat = self.treat_dict[k]
#                     treats.append(treat)
                treats = self.alltreat[indice[0] : indice[self.args.input_window], :, :]
            else:
                treats = None
            
            #adj = self.adjacency_matrix[d * self.args.tim_num + last_t]
            adj = combine_sparse_tensors(self.adjacency_matrix[indice[0] : indice[self.args.input_window]])
            indice = torch.LongTensor(indice)
            t = torch.LongTensor([calculate_week_hour(i.numpy().tolist()) for i in indice[:self.args.input_window]])
            #t = indice[:self.args.input_window] % self.args.tim_num  # 获取时间步
            t = t.unsqueeze(1).expand(-1, x.shape[1])  # 扩展时间步的维度
            
            data.append([indice, x, y, t, treats, adj])
         
        train_data, valid_data, test_data = data[:train_len], data[train_len:train_len+valid_len], data[train_len+valid_len:]
        return train_data, valid_data, test_data
    
    def get_indices(self):
        self.indices = []
        for i in range(self.flows.shape[0]- self.args.input_window - self.args.output_window):
            self.indices.append(list(range(i, i + self.args.input_window + self.args.output_window)))
    
    def get_flows(self):
        self.args.logger.info('Reading Regional flows data...')
        flows = np.load('/home/yangxiaojie/KDD2025/model/data/flows.npy')
        flows = flows[:, ::self.args.interval, :]
        self.flows = flows.reshape(len(self.adjacency_matrix), self.args.reg_num)
        self.args.logger.info('Regional flows data loaded!')
        self.args.logger.info('Regional flows datashape: ({}, {})'.format(self.flows.shape[0], self.flows.shape[1]))
    
    def get_adjacency(self):
        with open('/home/yangxiaojie/KDD2025/model/data/odmetrics_sparse_tensors.pk', 'rb') as f:
            self.adjacency_matrix = pk.load(f)
        self.args.logger.info('Regional adjacency matrix loaded!')
        self.args.logger.info('Regional adjacency matrix length: ({})'.format(len(self.adjacency_matrix)))
        self.args.logger.info('Adjacency matrix shape: ({}, {})'.format(self.adjacency_matrix[0].shape[0], self.adjacency_matrix[0].shape[1]))
        
    def get_treatment(self):
        if self.args.cache:
            self.args.logger.info('Load treatment from cache file...')
            with open('/home/yangxiaojie/KDD2025/model/data/treat_dict.pk', 'rb') as f:
                self.treat_dict = pk.load(f)
            self.args.logger.info('Treatment cache files loaded!')
        else:
            self.args.logger.info('Start processing treatments files...')
            #self.treat_dict = self.process_treat()
            self.alltreat = self.process_treat()
            self.args.logger.info('Treatment division done!')
#             if self.args.save:
#                 self.args.logger.info('Saving treatment dictionary...')
#                 with open('../data/treat_dict.pk', 'wb') as f:
#                     pk.dump(self.treat_dict, f)
#                 self.args.logger.info('Treatment cache files saved!')
    
    def process_treat(self):
        
        treatment_path = '/home/yangxiaojie/KDD2025/samples・説明書/treatment_scores_0618.pk'
        #treatment_path = '/home/yangxiaojie/KDD2025/samples・説明書/treatment_all.pk'
        self.args.logger.info('Reading news text embedding treatments from {}...'.format(treatment_path))
        with open(treatment_path, 'rb') as f:
            treatments = pk.load(f)
        
        treat_dict = {}
        alltreat = []
        
        alltreat = np.zeros((len(self.adjacency_matrix), self.args.reg_num, self.args.treat_dim))
        alltreat[:, :, 3] = 100
        
        for d in treatments:
            for t in treatments[d]:
                treat_dict[d * self.args.tim_num + t] = {}
                for c in treatments[d][t]:
                    #treat = np.mean(treatments[d][t][c], axis = 0)
                    treat = np.mean([_ for _ in treatments[d][t][c] if len(_) == self.args.treat_dim], axis = 0)
                    if np.isnan(treat).any():
                        treat = np.array([0, 0, 0, 100, 0, 0, 0, 0, 0, 0])
                    alltreat[d * self.args.tim_num + t, c] = treat
                    #alltreat.append(treat)

#         for i in tqdm(range(len(self.adjacency_matrix)), desc="Processing no treamtment situation..."):
#             if i not in treat_dict:
#                 treat_dict[i] = {}
#                 for j in range(self.args.reg_num):
#                     if j not in treat_dict[i]:
#                         treat_dict[i][j] = [0] * self.args.treat_dim
#             else:
#                 for j in range(self.args.reg_num):
#                     if j not in treat_dict[i]:
#                         treat_dict[i][j] = [0] * self.args.treat_dim
#             treat_dict[i] = np.array(list(dict(sorted(treat_dict[i].items())).values()))
        return alltreat