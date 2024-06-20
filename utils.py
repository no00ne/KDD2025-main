import random
import numpy as np
import torch
import pickle as pk
import zipfile
import os

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
   
def get_exp_id(directory='/home/yangxiaojie/KDD2025/model/log/'):
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

def create_zip(zip_name, file_list, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建压缩包的完整路径
    zip_path = os.path.join(output_dir, zip_name)

    # 打包文件
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_list:
            zipf.write(file, os.path.basename(file))

    print(f'打包完成，压缩包路径: {zip_path}')
