import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from model import CausalMob
from torch.utils.data import Dataset, DataLoader
from dataloader import CausalDataset, CausalDatasetPreloader
from train import train, test
from utils import *
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
    parser.add_argument('--treat_dim', type=int, default=10)
    parser.add_argument('--treat_hidden', type=int, default=64)
    parser.add_argument('--reg_num', type=int, default=490)
    parser.add_argument('--tim_num', type=int, default=24)
    parser.add_argument('--device', type=str, default='cuda:0')
    #the path will changed for anonymous review
    parser.add_argument('--path', type=str, default='./')
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444, 5555])

    args = parser.parse_args()

    # with open(os.path.join(args.path, '/osm_data/poi_distribution.pk'), 'rb') as f:
    #     poi_distribution = pk.load(f)

    # keys = sorted(set([poi_type for region in poi_distribution for poi_type in poi_distribution[region]]))

    # poi_region = np.zeros((len(poi_distribution), len(keys)))

    # key_to_index = {key: idx for idx, key in enumerate(keys)}

    # for i, region in enumerate(poi_distribution.keys()):
    #     for poi_type, count in poi_distribution[region].items():
    #         j = key_to_index[poi_type]
    #         poi_region[i, j] = count

    # poi_region = torch.FloatTensor(poi_region)

    # poi_data = (poi_region - torch.min(poi_region, dim = -1).values.unsqueeze(-1)) / (torch.max(poi_region, dim = -1).values - torch.min(poi_region, dim = -1).values).unsqueeze(-1)

    # poi_data = torch.FloatTensor(poi_data)

    poi_data = np.load('./data/poi_data.npy')

    args.poi_num = len(keys)
    args.poi_data = poi_data
    args.pt_dim = 64
    
    return args

    
def main():
    args = parse_args()

    expid = get_exp_id()
    
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=args.path + '/log/Training_{}.log'.format(expid), filemode='w')
    logger = logging.getLogger()
    args.expid = expid
    logger.info('Argument settings:')
    logger.info(args)
    args.logger = logger
    pack_source(args)

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

    model = CausalMob(args).to(args.device)
    args.logger.info("Model Structure: %s", model)

    optimizer = torch.optim.Adam(model.parameters(),
        lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, 
        amsgrad=False
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_step, factor=args.lr_decay)

    best_model, avg_losses = train(args, model, optimizer, scheduler, train_dataloader, valid_dataloader, scaler)

    metrics = test(args, best_model, test_dataloader, scaler)
    
    if seed is not None:
        model_path = args.path + f'/models/model_{args.expid}_{seed}_{args.causal}.pth'
    else:
        model_path = args.path + f'/models/model_{args.expid}_random_{args.causal}.pth'
    torch.save(best_model, model_path)
    args.logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
