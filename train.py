import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import StandardScaler
from losses import calculate_loss, validation_loss, rwt_regression_loss, IPM_loss

def train(args, model, optimizer, scheduler, train_dataloader, valid_dataloader, scaler):
    best_loss = float('inf')  # 记录前一次的验证损失，用于比较
    best_model = None
    avg_losses = []  # 用于存储每个epoch的平均损失
    args.logger.info('Training...')

    for epoch in range(1, args.num_epoch + 1):
        args.logger.info(f'Epoch {epoch}')
        model.train()  # 设置模型为训练模式
        avg_loss = []  # 存储每个batch的平均损失

        with tqdm(total=len(train_dataloader), desc=f'Epoch [{epoch}/{args.num_epoch}]', ncols=100) as pbar:
            for batch in train_dataloader:
                optimizer.zero_grad()  # 梯度清零
                losses = 0.0  # 初始化当前batch的总损失
#                 x, y, t, adj, treat, label, mask, indice = batch
#                 y = y.permute(0, 2, 1).reshape(y.shape[0] * y.shape[-1], -1)
#                 x = x.squeeze(-1).permute(0, 2, 1)
#                 t = t.permute(0, 2, 1)
                
                batch = [_.to(args.device) for _ in batch]
                x, y, t, adj, treat, mask, indice = batch
                #print(y.shape)
                y = y.permute(0, 2, 1).reshape(y.shape[0] * y.shape[-1], -1)
                #print(x.shape, y.shape, t.shape, adj.shape, treat.shape, mask.shape, indice.shape)
                y_pre, w, z, t = model(x, t, treat, adj, mask)
                
#                 if args.causal:
#                     label = label.reshape(label.shape[0] * label.shape[1])

                losses += calculate_loss(args, z, y, y_pre, scaler, w, t)
                
                #losses /= len(batch)
                losses.backward()  # 反向传播
                optimizer.step()  # 优化器更新
                torch.cuda.empty_cache()
                avg_loss.append(losses.item())  # 存储平均损失
                pbar.set_postfix({'loss ': f' {np.mean(avg_loss):.4f}'})  # 更新进度条上的损失信息
                pbar.update(1)  # 更新进度条
        avg_losses.append(np.mean(avg_loss))  # 存储每个epoch的平均损失
        
        metrics = test(args, model, valid_dataloader, scaler, mode = 'Validate')
        
        scheduler.step(metrics['RMSE'])
        
        if best_loss > metrics['RMSE']:
            best_model = model
            best_loss = metrics['RMSE']
            args.logger.info('Best model updated!')
            
        lr = optimizer.param_groups[0]['lr']
        args.logger.info('Current Learning Rate: {}'.format(lr))
        
        if optimizer.param_groups[0]['lr'] <= args.early_stop_lr:
            args.logger.info('Early Stop!')
            break
        
    args.logger.info('Training process done.')
    return best_model, avg_losses  # 返回训练后的模型和每个epoch的平均损失

def test(args, model, test_dataloader, scaler, mode = 'Test'):
    args.logger.info('Start ' + mode)
    model.eval()  # 设置模型为评估模式
    y_preds = []
    y_trues = []
    with torch.no_grad():  # 禁用梯度计算
        for batch in test_dataloader:
            batch = [_.to(args.device) for _ in batch]
            x, y, t, adj, treat, mask, indice = batch
            #y = y.permute(0, 2, 1).reshape(y.shape[0] * y.shape[-1], -1)

            y_pre, w, z, treat = model(x, t, treat, adj, mask)
            
            y_pre = y_pre.reshape(y.shape)
            
            y_preds = []
            y_trues = []
            
            #print(y.shape, y_pre.shape)
            
            single_results = {}
            for i in range(args.output_window):
                single_results[i] = {}
                
                y_pred = torch.flatten(scaler.inverse_transform(y_pre[:, i, :].cpu().squeeze())).detach().numpy().tolist()
                y_true = torch.flatten(scaler.inverse_transform(y[:, i, :].cpu().squeeze())).detach().numpy().tolist()

                single_results[i]['pred'] = y_pred
                single_results[i]['true'] = y_true

                y_preds += y_pred
                y_trues += y_true
    
    for i in range(args.output_window):
        metric = validation_loss(single_results[i]['pred'], single_results[i]['true'])
        args.logger.info(
            f'{mode} Metrics for single time slot at {i}: '
            f'MSE={metric["MSE"]:.4f}, '
            f'RMSE={metric["RMSE"]:.4f}, '
            f'MAE={metric["MAE"]:.4f}, '
            f'MAPE={metric["MAPE"]:.4f}%'
        )
    
    metrics = validation_loss(y_preds, y_trues)
    args.logger.info(
    f'{mode} Metrics: '
    f'MSE={metrics["MSE"]:.4f}, '
    f'RMSE={metrics["RMSE"]:.4f}, '
    f'MAE={metrics["MAE"]:.4f}, '
    f'MAPE={metrics["MAPE"]:.4f}%'
)
    args.logger.info(f'{mode} process done.')
    return metrics