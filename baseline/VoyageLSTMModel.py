# VoyageLSTMModel.py

import torch
import torch.nn as nn

class VoyageLSTMModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_size, static_hidden):
        super(VoyageLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_input_dim, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
        )
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size + static_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, seq_feats, static_feats):
        lstm_out, (h_n, c_n) = self.lstm(seq_feats)
        h_seq = h_n[-1]                           # (batch, hidden_size)
        h_static = self.static_fc(static_feats)   # (batch, static_hidden)
        h = torch.cat([h_seq, h_static], dim=1)   # (batch, hidden_size+static_hidden)
        y_pred = self.output_fc(h)                # (batch, 1)
        return y_pred.squeeze(1)
