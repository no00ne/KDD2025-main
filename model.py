import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from losses import *

class Truncated_power(nn.Module):
    def __init__(self, degree, knots):
        super(Truncated_power, self).__init__()
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        if x.dim() == 2:
            x = x.mean(dim = -1)
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis, device=x.device)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                out[:, _] = x**_
            else:
                out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out 

class MLP_treatnet(nn.Module):
    def __init__(self, num_out, n_hidden=10, num_in=4) -> None:
        super(MLP_treatnet, self).__init__()
        self.num_in = num_in
        
        self.hidden1 = torch.nn.Linear(num_in, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, num_out)
        self.act = nn.ReLU()
    
    def forward(self, x):
        x_mix = torch.zeros([x.shape[0], self.num_in])
        x_mix = x_mix.to(x.device)
        x_mix[:, 0] = 0
        if x.dim() == 2:
            x_mix[:, 1] = torch.cos(x * np.pi).mean(dim = -1)
            x_mix[:, 2] = torch.sin(x * np.pi).mean(dim = -1)
        elif x.dim() == 1:
            x_mix[:, 1] = torch.cos(x * np.pi)
            x_mix[:, 2] = torch.sin(x * np.pi)
        x_mix[:, -1] = x.mean(dim = -1)
        h = self.act(self.hidden1(x_mix))     # relu
        y = self.predict(h)

        return y

class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0, dynamic_type='power'):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        if dynamic_type == 'power':
            self.spb = Truncated_power(degree, knots)
            self.d = self.spb.num_of_basis # num of basis
        else:
            self.spb = MLP_treatnet(num_out=64, n_hidden=64, num_in=4)
            self.d = 64

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'soft':
            self.act = nn.Softplus()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        treat_dim = 64
        x_feature = x[:, treat_dim:]
        x_treat = x[:, :treat_dim]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d
        
        x_treat_basis = self.spb(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias
        
        if x_treat.dim() == 1:
            x_treat = torch.unsqueeze(x_treat, 1)
               
        if self.act is not None:
            out = self.act(out)
        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)
        
        return out

class ADMIT(nn.Module):
    def __init__(self, args):
        super(ADMIT, self).__init__()
        #(input_dim, treat_dim, dynamic_type)
        self.args = args
        input_dim = args.input_dim
        dynamic_type = args.dynamic_type, 
        init=args.init
        self.cfg_hidden = [(input_dim, 64, 1, 'relu'), (64, 64, 1, 'relu')]
        #self.cfg_hidden = [(input_dim, 64, 1, 'relu')]
        self.cfg = [(64, 64, 1, 'relu'), (64, self.args.output_window, 1, 'id')]
        #self.cfg = [(128, 1, 1, 'id')]
        self.degree = 2
        self.knots = [0.33, 0.66]

        # construct the representation network#
        hidden_blocks = []
        hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(self.cfg_hidden):
            # fc layer
            if layer_idx == 0:
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                hidden_blocks.append(self.feature_weight)
            else:
                hidden_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                hidden_blocks.append(nn.ReLU(inplace=True))
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*hidden_blocks)
        self.drop_hidden = nn.Dropout(p=self.args.dropout)

        self.hidden_dim = self.args.hidden_dim

        # construct the inference network
        blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1, dynamic_type = dynamic_type)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0, dynamic_type = dynamic_type))
        blocks.append(last_layer)

        self.out = nn.Sequential(*blocks)

        # construct the rw-weighting network
        rwt_blocks = []
        for layer_idx, layer_cfg in enumerate(self.cfg):
            if layer_idx == len(self.cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1, dynamic_type='mlp')
                
            else:
                rwt_blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0, dynamic_type='mlp'))
        rwt_blocks.append(last_layer)

        self.rwt = nn.Sequential(*rwt_blocks)

        self._initialize_weights(init)

    def forward(self, x, treat):
        if treat.dim() == 1:
            treat = torch.unsqueeze(treat, 1)
        hidden = self.hidden_features(x)
        hidden = self.drop_hidden(hidden)
        t_hidden = torch.cat((treat, hidden), 1)
        w = self.rwt(t_hidden)
        w = torch.sigmoid(w) * 2
        w = torch.exp(w) / torch.exp(w).sum(dim=0) * w.shape[0]
        out = self.out(t_hidden)

        return out, w, hidden

    def _initialize_weights(self, init):
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                # m.weight.data.normal_(0, 1.)
                m.weight.data.normal_(0, init)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()

    
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, in_features, hiddden_features, out_features, dropout=0):
        super(GCN, self).__init__()        
        self.gcn1 = GraphConvolution(in_features, hiddden_features)
        self.gcn2 = GraphConvolution(hiddden_features, out_features)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        return self.gcn2(x, adj)

class ResNormal(nn.Module):
    def __init__(self, args):
        super(ResNormal, self).__init__()
        self.args = args
        self.gcn1 = GCN(self.args.hidden_dim, self.args.hidden_dim, self.args.hidden_dim, self.args.dropout)
        self.gcn2 = GCN(self.args.hidden_dim, self.args.hidden_dim, self.args.hidden_dim, self.args.dropout)

    def forward(self, x, adj):
        cl_conv1 = self.gcn1(x, adj)
        cl_conv2 = self.gcn2(cl_conv1, adj)
        return cl_conv2 + x

class PTTrans(nn.Module):
    def __init__(self, args):
        super(PTTrans, self).__init__()
        self.args = args
        self.tim_embedding = nn.Embedding(168, self.args.tim_dim)
        self.linear = nn.Linear(self.args.tim_dim, self.args.poi_num)

    def forward(self, poi_in, t):
        weights = self.linear(self.tim_embedding(t).mean(dim = -2))
        #print(weights.shape, t.shape, poi_in.shape)
        poi_time = poi_in * weights, self.tim_embedding(t).mean(dim = -2)
        return poi_time
    
class CausalFlow(nn.Module):
    def __init__(self, args):
        super(CausalFlow, self).__init__()
        self.args = args
        self.reg_num = args.reg_num
        #self.tim_num = args.tim_num
        self.tim_num = 168
        self.reg_dim = args.reg_dim
        self.tim_dim = args.tim_dim
        
        self.hidden_dim = args.hidden_dim           
        self.reg_embedding = nn.Embedding(self.reg_num, self.reg_dim)
        self.pt_trans = PTTrans(self.args)
        self.linear = nn.Sequential(nn.Linear(self.args.input_window + self.args.poi_num + self.args.tim_dim + self.args.reg_dim, self.hidden_dim))
        self.res_blocks = nn.ModuleList()
        
        for _ in range(2):
            self.res_blocks.append(ResNormal(self.args))
            
        if self.args.causal:
            self.treat_gru_prev = nn.GRU(self.args.treat_dim, self.args.treat_hidden, batch_first=True)
            self.treat_gru_post = nn.GRU(self.args.treat_dim, self.args.treat_hidden, batch_first=True)
            self.treat_linear = nn.Linear(2 * self.args.treat_hidden, self.args.treat_hidden)
            self.admit = ADMIT(self.args)

        else:
            self.out = nn.Sequential(
                           nn.Linear(self.hidden_dim, self.args.output_window))
    
    def get_treat_base(self):
        treat_prev = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * self.args.input_window).unsqueeze(0).to(self.args.device)
        treat_post = torch.FloatTensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * self.args.output_window).unsqueeze(0).to(self.args.device)
        base_prev, _ = self.treat_gru_prev(treat_prev)
        base_post, _ = self.treat_gru_post(treat_post)
        base = self.treat_linear(torch.cat([base_prev[:, -1, :], base_post[:, -1, :]], dim = -1))
        return base
    
    def treat_encoder(self, x, treat):
        
        treat = treat.permute(0, 2, 1, 3)
        treat = treat.reshape(treat.shape[0] * treat.shape[1], treat.shape[2], treat.shape[3])

        treat_prev = treat[:, :self.args.input_window, :]
        treat_post = torch.flip(treat[:, self.args.input_window:, :], [1])
        
        treat_hidden_prev, _ = self.treat_gru_prev(treat_prev)
        treat_hidden_post, _ = self.treat_gru_post(treat_post)
        treat_hidden = self.treat_linear(torch.cat([treat_hidden_prev[:, -1, :], treat_hidden_post[:, -1, :]], dim = -1))
        return treat_hidden
    
    def forward(self, x, t, treat, adj):
        
        if self.args.causal:
            treat = self.treat_encoder(x, treat)
        
        x = x.squeeze(-1).permute(0, 2, 1)
        t = t.permute(0, 2, 1)
        
        emb_reg = self.reg_embedding.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        feature = self.args.poi_data.to(self.args.device)
        poi_time, t_emb_mean = self.pt_trans(feature, t)

        output = self.linear(torch.cat([x, poi_time, emb_reg, t_emb_mean], dim=-1))

        z = []
        for i in range(x.shape[0]):
            hidden = output[i, ...]
            for res_block in self.res_blocks:
                hidden = res_block(hidden, adj[i][-1])
            z.append(hidden)
        
        z = torch.stack(z, dim = 0)
        z = z.reshape(z.shape[0] * z.shape[1], self.args.hidden_dim)
        #output is confounder z  
        if self.args.causal:


            out, w, _ = self.admit(z, treat)
            return out, w, z, treat
        else:
            return self.out(z), None, None, None