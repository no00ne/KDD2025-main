import math
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 配置参数
class Config:
    input_size = 7  # [lat, lon, speed, heading, course, timestamp, draught]
    hidden_size = 64
    num_layers = 2
    output_size = 5  # [heading, course, timestamp, latitude, longitude]
    sequence_length = 12  # 使用12个时间步预测下一个位置
    prediction_length = 24  # 预测未来24个时间步
    batch_size = 16
    learning_rate = 0.001
    epochs = 5
    dropout = 0.2
    data_dir = "../ship_trajectories_orig"  # 数据目录
    model_save_path = "./models/vessel_trajectory_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# 数据处理类
class AISDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scaler_features = None
        self.scaler_targets = None
        self.all_data = []

    def load_data(self):
        """加载所有jsonl文件的数据"""
        jsonl_files = glob.glob(os.path.join(self.data_dir, "*.jsonl"))
        jsonl_files = jsonl_files[:1]

        for file_path in tqdm(jsonl_files, desc="Loading data files"):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        vessel_data = json.loads(line)
                        # 确保路径中有足够的点来训练
                        if len(vessel_data.get('Path', [])) >= config.sequence_length + config.prediction_length:
                            self.all_data.append(vessel_data)
                            if self.all_data.__len__() > 30:
                                break
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON line in {file_path}")
                        continue

        print(f"Loaded {len(self.all_data)} valid vessel trajectories")
        return self.all_data

    def preprocess_data(self):
        """预处理所有路径数据为训练格式"""
        features = []
        targets = []

        for vessel in tqdm(self.all_data, desc="Preprocessing data"):
            path = vessel['Path']

            # 转换时间戳为相对秒数
            base_time = pd.to_datetime(path[0]['timestamp'])

            for i in range(len(path) - config.sequence_length - config.prediction_length + 1):
                # 输入序列
                seq_data = []
                for j in range(i, i + config.sequence_length):
                    point = path[j]
                    # 计算相对时间（秒）
                    rel_time = (pd.to_datetime(point['timestamp']) - base_time).total_seconds() / 3600  # 转为小时

                    # 特征：[lat, lon, speed, heading, course, time, draught]
                    features_point = [
                        point['latitude'],
                        point['longitude'],
                        point.get('speed', 0),
                        point.get('heading', 0),
                        point.get('course', 0),
                        rel_time,
                        point.get('draught', 0)
                    ]

                    seq_data.append(features_point)

                # 目标序列（未来路径）
                target_seq = []
                for j in range(i + config.sequence_length, i + config.sequence_length + config.prediction_length):
                    point = path[j]
                    rel_time = (pd.to_datetime(point['timestamp']) - base_time).total_seconds() / 3600

                    # 目标：[heading, course, timestamp, latitude, longitude]
                    target_point = [
                        point.get('heading', 0),
                        point.get('course', 0),
                        rel_time,
                        point['latitude'],
                        point['longitude']
                    ]

                    target_seq.append(target_point)

                features.append(seq_data)
                targets.append(target_seq)

        # 转换为numpy数组
        features = np.array(features)
        targets = np.array(targets)

        # 归一化特征
        features_shape = features.shape
        features = features.reshape(-1, features.shape[-1])
        self.scaler_features = MinMaxScaler()
        features = self.scaler_features.fit_transform(features)
        features = features.reshape(features_shape)

        # 归一化目标
        targets_shape = targets.shape
        targets = targets.reshape(-1, targets.shape[-1])
        self.scaler_targets = MinMaxScaler()
        targets = self.scaler_targets.fit_transform(targets)
        targets = targets.reshape(targets_shape)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        return X_train, X_val, y_train, y_val, self.scaler_features, self.scaler_targets


# 定义数据集类
class VesselTrajectoryDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# 修改后的Transformer模型，支持teacher forcing
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        # 特征投影层
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(output_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)

        self.d_model = d_model

    def forward(self, src, tgt_len=None, tgt=None, teacher_forcing_ratio=0):
        """
        前向传播函数，支持teacher forcing

        Args:
            src: 源序列 [batch_size, src_seq_len, input_size]
            tgt_len: 目标序列长度
            tgt: 目标序列 [batch_size, tgt_seq_len, output_size]
            teacher_forcing_ratio: teacher forcing的比例

        Returns:
            output: 预测序列 [batch_size, tgt_seq_len, output_size]
        """
        # src: [batch_size, src_seq_len, input_size]
        device = src.device
        batch_size = src.size(0)

        # 如果没有提供目标长度，使用预测长度
        if tgt_len is None:
            tgt_len = config.prediction_length

        # 投影到模型维度并添加位置编码
        src_projected = self.pos_encoder(self.input_projection(src))

        # 准备解码器输入
        if tgt is None:
            # 推理模式：创建全零目标序列
            decoder_input = torch.zeros((batch_size, tgt_len, self.output_size), device=device)
        else:
            # 训练模式：使用teacher forcing
            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # 直接使用真实目标
                decoder_input = tgt
            else:
                # 不使用teacher forcing时也创建全零序列
                decoder_input = torch.zeros((batch_size, tgt_len, self.output_size), device=device)

        # 投影解码器输入并添加位置编码
        tgt_projected = self.pos_encoder(self.output_projection(decoder_input))

        # 创建目标掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # Transformer处理
        output = self.transformer(src_projected, tgt_projected, tgt_mask=tgt_mask)

        # 输出投影
        output = self.fc_out(output)

        return output


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}") as pbar:
            for src, tgt in pbar:
                src = src.to(config.device)
                tgt = tgt.to(config.device)

                optimizer.zero_grad()

                # 现在forward方法接受teacher_forcing_ratio参数
                output = model(src, tgt.size(1), tgt, teacher_forcing_ratio=0.5)

                loss = criterion(output, tgt)
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        val_loss = evaluate_model(model, val_loader, criterion, config)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    # 绘制训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    return train_losses, val_losses


# 评估函数
def evaluate_model(model, data_loader, criterion, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(config.device)
            tgt = tgt.to(config.device)

            # 使用修改后的forward方法
            output = model(src, tgt.size(1), tgt, teacher_forcing_ratio=0)

            loss = criterion(output, tgt)
            total_loss += loss.item()

    return total_loss / len(data_loader)


# 预测函数
def predict_trajectory(model, input_sequence, prediction_length, scaler_features, scaler_targets):
    model.eval()

    # 确保输入的格式正确 [1, seq_len, feature_dim]
    if not torch.is_tensor(input_sequence):
        input_sequence = torch.FloatTensor(input_sequence).unsqueeze(0)

    input_sequence = input_sequence.to(config.device)

    with torch.no_grad():
        # 使用模型进行预测，不使用teacher forcing
        output = model(input_sequence, prediction_length)

        # 获取预测结果
        predictions = output.cpu().numpy()[0]  # [pred_len, output_dim]

    # 反归一化预测结果
    predictions_shape = predictions.shape
    predictions = predictions.reshape(-1, predictions.shape[-1])
    predictions = scaler_targets.inverse_transform(predictions)
    predictions = predictions.reshape(predictions_shape)

    return predictions


# 可视化预测结果
def visualize_trajectory(actual_path, predicted_path):
    plt.figure(figsize=(10, 8))

    # 绘制实际轨迹
    plt.plot(
        [p[4] for p in actual_path],  # longitude
        [p[3] for p in actual_path],  # latitude
        'b-', marker='o', markersize=3, label='Actual Path'
    )

    # 绘制预测轨迹
    plt.plot(
        [p[4] for p in predicted_path],  # longitude
        [p[3] for p in predicted_path],  # latitude
        'r-', marker='x', markersize=3, label='Predicted Path'
    )

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Vessel Trajectory Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('trajectory_prediction.png')
    plt.close()


# 计算预测误差
def calculate_metrics(actual_path, predicted_path):
    # 计算位置误差
    lat_errors = [abs(actual_path[i][3] - predicted_path[i][3]) for i in range(len(actual_path))]
    lon_errors = [abs(actual_path[i][4] - predicted_path[i][4]) for i in range(len(actual_path))]

    # 计算Haversine距离（实际距离，单位：千米）
    distances = []
    for i in range(len(actual_path)):
        lat1, lon1 = actual_path[i][3], actual_path[i][4]
        lat2, lon2 = predicted_path[i][3], predicted_path[i][4]

        # Haversine公式
        R = 6371  # 地球半径，单位：千米
        dLat = np.radians(lat2 - lat1)
        dLon = np.radians(lon2 - lon1)
        a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
            dLon / 2) * np.sin(dLon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        distances.append(distance)

    # 计算航向误差
    heading_errors = [
        min(abs(actual_path[i][0] - predicted_path[i][0]), 360 - abs(actual_path[i][0] - predicted_path[i][0])) for i in
        range(len(actual_path))]

    return {
        'mean_lat_error': np.mean(lat_errors),
        'mean_lon_error': np.mean(lon_errors),
        'max_lat_error': np.max(lat_errors),
        'max_lon_error': np.max(lon_errors),
        'mean_distance_error_km': np.mean(distances),
        'max_distance_error_km': np.max(distances),
        'mean_heading_error': np.mean(heading_errors),
        'max_heading_error': np.max(heading_errors)
    }


# 主函数
def main():
    print(f"Using device: {config.device}")

    # 数据处理
    processor = AISDataProcessor(config.data_dir)
    all_data = processor.load_data()
    X_train, X_val, y_train, y_val, scaler_features, scaler_targets = processor.preprocess_data()

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    # 创建数据加载器
    train_dataset = VesselTrajectoryDataset(X_train, y_train)
    val_dataset = VesselTrajectoryDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = TransformerModel(
        input_size=config.input_size,
        output_size=config.output_size,
        dropout=config.dropout
    ).to(config.device)

    # 打印模型架构
    print(model)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, config)

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(config.model_save_path))

    # 选择一条路径进行预测和可视化
    test_idx = np.random.randint(0, len(X_val))
    test_input = X_val[test_idx]
    test_target = y_val[test_idx]

    # 预测未来轨迹
    predicted_path = predict_trajectory(model, test_input, config.prediction_length, scaler_features, scaler_targets)

    # 可视化结果
    visualize_trajectory(test_target, predicted_path)

    # 计算并打印评估指标
    metrics = calculate_metrics(test_target, predicted_path)
    print("\nPrediction Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 保存几个测试样例的预测结果
    num_test_samples = min(5, len(X_val))
    for i in range(num_test_samples):
        idx = np.random.randint(0, len(X_val))
        test_input = X_val[idx]
        test_target = y_val[idx]

        predicted_path = predict_trajectory(model, test_input, config.prediction_length, scaler_features,
                                            scaler_targets)
        visualize_trajectory(test_target, predicted_path)
        plt.savefig(f'trajectory_sample_{i + 1}.png')
        plt.close()

    print(f"\nModel training and evaluation complete. Best model saved to {config.model_save_path}")

    return model, scaler_features, scaler_targets


# 预测新的船舶轨迹
def predict_new_vessel_trajectory(model, vessel_data, scaler_features, scaler_targets, config):
    """
    为新的船舶数据预测未来轨迹

    Args:
        model: 训练好的模型
        vessel_data: 船舶历史轨迹数据
        scaler_features: 特征归一化器
        scaler_targets: 目标归一化器
        config: 配置对象

    Returns:
        预测的轨迹点列表，每个点包含[heading, course, timestamp, latitude, longitude]
    """
    # 确保路径有足够的点
    if len(vessel_data['Path']) < config.sequence_length:
        raise ValueError(f"Vessel path must have at least {config.sequence_length} points")

    # 处理输入数据
    path = vessel_data['Path'][-config.sequence_length:]
    base_time = pd.to_datetime(path[0]['timestamp'])

    # 准备输入序列
    input_seq = []
    for point in path:
        rel_time = (pd.to_datetime(point['timestamp']) - base_time).total_seconds() / 3600

        # 特征：[lat, lon, speed, heading, course, time, draught]
        features_point = [
            point['latitude'],
            point['longitude'],
            point.get('speed', 0),
            point.get('heading', 0),
            point.get('course', 0),
            rel_time,
            point.get('draught', 0)
        ]

        input_seq.append(features_point)

    # 归一化输入
    input_seq = np.array(input_seq)
    input_seq = scaler_features.transform(input_seq)
    input_seq = torch.FloatTensor(input_seq).unsqueeze(0)  # [1, seq_len, input_dim]

    # 预测未来轨迹
    predicted_path = predict_trajectory(model, input_seq, config.prediction_length, scaler_features, scaler_targets)

    # 将预测结果转换为可读格式
    last_timestamp = pd.to_datetime(path[-1]['timestamp'])
    result_path = []

    for i, point in enumerate(predicted_path):
        # 计算实际时间戳
        time_hours = point[2]  # 相对小时数
        timestamp = last_timestamp + pd.Timedelta(hours=time_hours)

        point_dict = {
            'heading': float(point[0]),
            'course': float(point[1]),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'latitude': float(point[3]),
            'longitude': float(point[4])
        }

        result_path.append(point_dict)

    return result_path


if __name__ == "__main__":
    # 训练和评估模型
    model, scaler_features, scaler_targets = main()

    # 预测示例
    print("\nGenerating prediction example for a new vessel...")

    # 假设我们从验证集中获取一条船舶数据作为示例
    processor = AISDataProcessor(config.data_dir)
    all_data = processor.load_data()

    if len(all_data) > 0:
        sample_vessel = all_data[0]

        try:
            # 只使用部分历史轨迹作为输入
            sample_vessel_input = {
                'MMSI': sample_vessel['MMSI'],
                'IMO': sample_vessel['IMO'],
                'Name': sample_vessel['Name'],
                'Path': sample_vessel['Path'][:config.sequence_length]
            }

            # 预测未来轨迹
            predicted_trajectory = predict_new_vessel_trajectory(
                model, sample_vessel_input, scaler_features, scaler_targets, config
            )

            print("Predicted trajectory for vessel:", sample_vessel['Name'])
            for i, point in enumerate(predicted_trajectory[:5]):  # 只打印前5个点
                print(f"Point {i + 1}: Lat={point['latitude']:.6f}, Lon={point['longitude']:.6f}, "
                      f"Heading={point['heading']:.1f}, Time={point['timestamp']}")
            print("...")

            # 实际对比数据
            actual_future = sample_vessel['Path'][
                            config.sequence_length:config.sequence_length + config.prediction_length]

            # 可视化预测vs实际
            plt.figure(figsize=(10, 8))

            # 绘制历史轨迹
            history_path = sample_vessel['Path'][:config.sequence_length]
            plt.plot(
                [p['longitude'] for p in history_path],
                [p['latitude'] for p in history_path],
                'b-', marker='o', markersize=3, label='Historical Path'
            )

            # 绘制实际未来轨迹
            plt.plot(
                [p['longitude'] for p in actual_future],
                [p['latitude'] for p in actual_future],
                'g-', marker='o', markersize=3, label='Actual Future Path'
            )

            # 绘制预测轨迹
            plt.plot(
                [p['longitude'] for p in predicted_trajectory],
                [p['latitude'] for p in predicted_trajectory],
                'r-', marker='x', markersize=3, label='Predicted Path'
            )

            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Vessel Trajectory Prediction - {sample_vessel["Name"]}')
            plt.legend()
            plt.grid(True)
            plt.savefig('sample_prediction.png')

        except Exception as e:
            print(f"Error generating prediction example: {e}")