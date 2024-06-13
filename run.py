import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from model import CausalFlow
from torch.utils.data import Dataset, DataLoader
from dataloader import CausalDataset, CausalDatasetPreloader
from train import train, test
from utils import create_zip
import pickle as pk
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="dose-response estimation via neural network")

    parser.add_argument('--reg_dim', type=int, default=64)
    parser.add_argument('--tim_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--init', type=float, default=0.01)
    parser.add_argument('--dynamic_type', type=str, default='power', choices=['power', 'mlp'])
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--treat_dim', type=int, default=4096)
    parser.add_argument('--treat_hidden', type=int, default=64)
    parser.add_argument('--reg_num', type=int, default=490)
    parser.add_argument('--tim_num', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--input_window', type=int, default=6)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--output_window', type=int, default=1)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--training_rate', type=float, default=0.8)
    parser.add_argument('--validate_rate', type=float, default=0.1)
    parser.add_argument('--testing_rate', type=float, default=0.1)
    parser.add_argument('--causal', type=bool, default=False)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--early_stop_lr', type=float, default=9e-6)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--cache', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--seeds', type=int, nargs='+', default=[76, 2015, 4974, 8767, 1893, 3447, 2799, 6838, 6147, 5490, 7033, 6884, 865, 2927, 9042, 3312, 781, 8141, 4374, 5073])  # 添加种子列表参数

    args = parser.parse_args()
    
    with open('/home/yangxiaojie/KDD2025/osm_data/poi_distribution.pk', 'rb') as f:
        poi_distribution = pk.load(f)

    keys = set([poi_type for region in poi_distribution for poi_type in poi_distribution[region]])

    poi_region = np.zeros((len(poi_distribution), len(keys)))

    for i, region in enumerate(poi_distribution.keys()):
        for j, key in enumerate(keys):
            if key in poi_distribution[region]:
                poi_region[i, j] = poi_distribution[region][key]

    poi_data = (poi_region - poi_region.min()) / (poi_region.max() - poi_region.min())

    poi_data = torch.FloatTensor(poi_data)

    args.poi_num = len(keys)
    args.poi_data = poi_data
    args.pt_dim = 64
    
    return args

def set_seed(seed):
    """设置所有可能的随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pack_source(args):
    zip_name = 'source_{}.zip'.format(args.expid)
    file_list = ['dataloader.py', 'model.py', 'losses.py', 'run.py', 'train.py', 'normalization.py']
    output_dir = '/home/yangxiaojie/KDD2025/model/sources/'
    create_zip(zip_name, file_list, output_dir)
    args.logger.info('Packed source code saved!')

def save_dataloader(dataloader, file_path):
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    num_workers = dataloader.num_workers
    collate_fn = dataloader.collate_fn
    shuffle = isinstance(dataloader.sampler, torch.utils.data.sampler.RandomSampler)
    
    dataloader_params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
    }
    
    with open(file_path, 'wb') as f:
        pk.dump((dataset, dataloader_params), f)

def load_dataloader(file_path):
    with open(file_path, 'rb') as f:
        dataset, dataloader_params = pk.load(f)
    
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_params)
    return dataloader
   
def get_exp_id(directory = '/home/yangxiaojie/KDD2025/model/log/'):
    exp_ids = []
    expid = random.randint(1000, 9999)
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件名的最后四个字符
            last_four = file[-8:-4]
            exp_ids.append(int(last_four))
            
    while expid in exp_ids:
        expid = random.randint(1000, 9999)
    
    return expid
    
def main():
    args = parse_args()

    # 设置日志记录
    
    expid = get_exp_id()
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='./log/Training_{}.log'.format(expid), filemode='w')
    logger = logging.getLogger()
    args.expid = expid
    logger.info('Argument settings:')
    logger.info(args)
    args.logger = logger
    pack_source(args)
    
    # 处理随机种子
    if not args.random:
        for seed in args.seeds:
            print(f"Running experiment with seed: {seed}")
            args.logger.info(f"Running experiment with seed: {seed}")
            set_seed(seed)
            run_experiment(args, seed)
    else:
        run_experiment(args, None)

def run_experiment(args, seed):
    if not args.cache:
        dataset = CausalDatasetPreloader(args)
        train_data, valid_data, test_data = dataset.data_split()

        train_dataset, valid_dataset, test_dataset = CausalDataset(args, train_data, mode='train'), CausalDataset(args, valid_data), CausalDataset(args, test_data)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        scaler = dataset.scaler
        if args.save:
            tail = '_in{}_out{}_batch{}_causal{}.pkl'.format(args.input_window, args.output_window, args.batch_size, args.causal)
            save_dataloader(train_dataloader, '/home/yangxiaojie/KDD2025/model/cache/train' + tail)
            save_dataloader(valid_dataloader, '/home/yangxiaojie/KDD2025/model/cache/valid' + tail)
            save_dataloader(test_dataloader, '/home/yangxiaojie/KDD2025/model/cache/test' + tail)
            
            with open('/home/yangxiaojie/KDD2025/model/cache/scaler.pkl' + tail, 'wb') as f:
                pk.dump(scaler, f)
    else:
        train_dataloader = load_dataloader('./cache/train' + tail)
        valid_dataloader = load_dataloader('./cache/valid' + tail)
        test_dataloader = load_dataloader('./cache/test' + tail)
            
        with open('./cache/scaler'+tail, 'rb') as f:
            scaler = pk.load(f)
    #raise Exception("Forcefully exiting the program")
    model = CausalFlow(args).to(args.device)
    args.logger.info("Model Structure: %s", model)

    optimizer = torch.optim.Adam(model.parameters(),
        lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, 
        amsgrad=False
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_step, factor=args.lr_decay)

    model, avg_losses = train(args, model, optimizer, scheduler, train_dataloader, valid_dataloader, scaler)

    metrics = test(args, model, test_dataloader, scaler)
    
    if seed is not None:
        model_path = f'/home/yangxiaojie/KDD2025/model/models/model_{args.expid}_{seed}_{args.causal}.pth'
    else:
        model_path = f'/home/yangxiaojie/KDD2025/model/models/model_{args.expid}_random_{args.causal}.pth'
    torch.save(model, model_path)
    args.logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
