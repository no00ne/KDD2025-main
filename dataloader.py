import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle as pk
from normalization import *
from datetime import datetime, timedelta

def calculate_hour(hours_passed, start_date = '2023-04-01'):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    target_datetime = start_datetime + timedelta(hours=hours_passed)
    
    is_weekend = target_datetime.weekday() >= 5 

    if is_weekend:
        result_hour = target_datetime.hour + 24
    else:
        result_hour = target_datetime.hour
    
    return result_hour

def calculate_week_hour(hours_passed, start_date = '2023-04-01'):
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    
    target_datetime = start_datetime + timedelta(hours=hours_passed)
    
    days_since_week_start = target_datetime.weekday()
    result_hour = days_since_week_start * 24 + target_datetime.hour
    
    return result_hour

def mean_along_dim0(sparse_tensor):
    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    shape = sparse_tensor.shape
    assert shape[0] > 0, "The size of dim 0 must be greater than 0"

    accumulated_values = torch.zeros((shape[1], shape[2]), dtype=values.dtype)
    count_values = torch.zeros((shape[1], shape[2]), dtype=values.dtype)

    for idx, val in zip(indices.t(), values):
        accumulated_values[idx[1], idx[2]] += val
        count_values[idx[1], idx[2]] += 1

    mean_values = accumulated_values / torch.clamp(count_values, min=1)
    mean_values[count_values == 0] = 0 

    mean_indices = mean_values.nonzero(as_tuple=False).t()
    mean_values_sparse = mean_values[mean_values != 0]
    mean_sparse_tensor = torch.sparse_coo_tensor(mean_indices, mean_values_sparse, (shape[1], shape[2]))

    return mean_sparse_tensor

def combine_sparse_tensors(tensor_list):
    batch_size = len(tensor_list)
    V = tensor_list[0].size(0)
    
    combined_indices = []
    combined_values = []

    for i, tensor in enumerate(tensor_list):
        indices = tensor._indices()
        values = tensor._values()
        
        batch_indices = torch.cat([torch.full((1, indices.size(1)), i, dtype=torch.long), indices], dim=0)
        
        if len(combined_indices) == 0:
            combined_indices = batch_indices
            combined_values = values
        else:
            combined_indices = torch.cat((combined_indices, batch_indices), dim=1)
            combined_values = torch.cat((combined_values, values))

    size = torch.Size([batch_size, V, V])

    combined_tensor = torch.sparse_coo_tensor(combined_indices, combined_values, size=size)
    
    return combined_tensor

class CausalDataset(Dataset):
    def __init__(self, args, data, mode = 'test'):
        self.args = args
        self.data = data
        self.mode = mode
        self.device = self.args.device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        indice, x, y, t, treat, adj = self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3], self.data[idx][4], self.data[idx][5]
        x = x.unsqueeze(2)
        if self.args.causal:
            treat = torch.FloatTensor(treat)
        else:
            treat = torch.zeros((x.shape))
        
        return x, y, t, adj, treat, indice

        
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
        if self.args.causal:
            prev_treats, post_treats = self.alltreat[0], self.alltreat[1]
        else:
            prev_treats, post_treats = None, None

        data = []
        for index, indice in enumerate(tqdm(self.indices, desc="Processing")):
            x = self.scaler.transform(torch.FloatTensor(self.flows[indice[:self.args.input_window], :]))
            
            y = self.scaler.transform(torch.FloatTensor(self.flows[indice[self.args.input_window:], :]))

            if self.args.causal:

                prev_treat = prev_treats[indice[:self.args.input_window]]
                post_treat = post_treats[indice[self.args.input_window:]]
                treats = torch.cat([prev_treat, post_treat], dim = 0)
            else:
                treats = None
            
            adj = combine_sparse_tensors(self.adjacency_matrix[indice[0] : indice[self.args.input_window]])
            indice = torch.LongTensor(indice)
            t = torch.LongTensor([calculate_week_hour(i.numpy().tolist()) for i in indice[:self.args.input_window]])
            t = t.unsqueeze(1).expand(-1, x.shape[1])
            
            data.append([indice, x, y, t, treats, adj])
         
        train_data, valid_data, test_data = data[:train_len], data[train_len:train_len+valid_len], data[train_len+valid_len:]
        return train_data, valid_data, test_data
    
    def get_indices(self):
        self.indices = []
        for i in range(self.flows.shape[0]- self.args.input_window - self.args.output_window):
            self.indices.append(list(range(i, i + self.args.input_window + self.args.output_window)))
    
    def get_flows(self):
        self.args.logger.info('Reading Regional flows data...')
        flows = np.load(os.path.join(self.args.path, '/data/flows.npy'))
        flows = flows[:, ::self.args.interval, :]
        self.flows = flows.reshape(len(self.adjacency_matrix), self.args.reg_num)
        self.args.logger.info('Regional flows data loaded!')
        self.args.logger.info('Regional flows datashape: ({}, {})'.format(self.flows.shape[0], self.flows.shape[1]))
    
    def get_adjacency(self):
        with open(os.path.join(self.args.path, '/data/odmetrics_sparse_tensors.pk'), 'rb') as f:
            self.adjacency_matrix = pk.load(f)
        self.args.logger.info('Regional adjacency matrix loaded!')
        self.args.logger.info('Regional adjacency matrix length: ({})'.format(len(self.adjacency_matrix)))
        self.args.logger.info('Adjacency matrix shape: ({}, {})'.format(self.adjacency_matrix[0].shape[0], self.adjacency_matrix[0].shape[1]))
        
    def get_treatment(self):

        self.args.logger.info('Start processing treatments files...')
        self.alltreat = self.process_treat()
        self.args.logger.info('Treatment division done!')

    
    def process_treat(self):
        
        prev_path = os.path.join(self.args.path, '/data/prev_treats_sum.npy')
        post_path = os.path.join(self.args.path, '/data/post_treats_sum.npy')
        
        prev_treats = np.load(prev_path)
        post_treats = np.load(post_path)

        assert prev_treats.shape[0] == len(self.adjacency_matrix)
        assert post_treats.shape[0] == len(self.adjacency_matrix)

        
        return [torch.FloatTensor(prev_treats), torch.FloatTensor(post_treats)]
