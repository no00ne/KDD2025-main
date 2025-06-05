import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization import StandardScaler
from losses import calculate_loss, validation_loss, rwt_regression_loss, IPM_loss

def train(args, model, optimizer, scheduler, train_dataloader, valid_dataloader, scaler):
    best_loss = float('inf')
    best_model = None
    avg_losses = [] 
    args.logger.info('Training...')

    for epoch in range(1, args.num_epoch + 1):
        args.logger.info(f'Epoch {epoch}')
        model.train()
        avg_loss = []

        with tqdm(total=len(train_dataloader), desc=f'Epoch [{epoch}/{args.num_epoch}]', ncols=100) as pbar:
            print("type(train_dataloader)", type(train_dataloader))
            for batch in train_dataloader:
                optimizer.zero_grad()
                losses = 0.0

                x, y, t, adj, treat, indice = batch

                x = x.to(args.device)
                y = y.to(args.device)
                t = t.to(args.device)
                treat = treat.to(args.device)
                indice = indice.to(args.device)

                adj = [a.to(args.device) for a in adj]

                y = y.permute(0, 2, 1)
                y_pre, w, z, t = model(x, t, treat, adj)

                if args.causal:
                    losses += calculate_loss(args, z, y, y_pre, scaler, w, t, model.get_treat_base())
                else:
                    losses += calculate_loss(args, z, y, y_pre, scaler)
                
                losses.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                avg_loss.append(losses.item())
                pbar.set_postfix({'loss ': f' {np.mean(avg_loss):.4f}'})
                pbar.update(1)
        avg_losses.append(np.mean(avg_loss))
        
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
    return best_model, avg_losses

def test(args, model, test_dataloader, scaler, mode = 'Test'):
    args.logger.info('Start ' + mode)
    model.eval()
    y_preds = []
    y_trues = []
    single_results = {}
    for i in range(args.output_window):
        single_results[i] = {}
        single_results[i]['pred'] = []
        single_results[i]['true'] = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [_.to(args.device) for _ in batch]
            x, y, t, adj, treat, indice = batch
            y = y.permute(0, 2, 1)

            y_pre, w, z, treat = model(x, t, treat, adj)
            
            y_pre = y_pre.unsqueeze(-1).reshape(y.shape)
            
            for i in range(args.output_window):                
                y_pred = torch.flatten(scaler.inverse_transform(y_pre[:, :, i].cpu().squeeze())).detach().numpy().tolist()
                y_true = torch.flatten(scaler.inverse_transform(y[:, :, i].cpu().squeeze())).detach().numpy().tolist()

                single_results[i]['pred'] += y_pred
                single_results[i]['true'] += y_true

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