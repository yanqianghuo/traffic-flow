import math
import torch
from torch import nn
from torch.nn import functional as F


class SpatialAttention(torch.nn.Module):
    """ Compute Spatial attention scores.

    Args:
        num_nodes: Number of nodes.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, num\_nodes, num\_nodes)`.
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(SpatialAttention, self).__init__()

        self.w1 = torch.nn.Conv2d(c_in, 1, 1, bias=False)
        self.w2 = torch.nn.Linear(f_in, c_in, bias=False)
        self.w3 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32),
            requires_grad=True
        )
        self.vs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        self.bs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        torch.nn.init.kaiming_uniform_(self.vs, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.bs, a =math.sqrt(5))
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(self.w1(x).squeeze(dim=1))
        y2 = torch.matmul(x, self.w3)

        product = torch.matmul(y1, y2)
        y = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        y = F.softmax(y, dim=-1)
        return y

class TemporalAttention(torch.nn.Module):
    """ Compute Temporal attention scores.

    Args:
        num_nodes: Number of vertices.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, c_in, c_in)`.
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(TemporalAttention, self).__init__()

        self.w1 = torch.nn.Parameter(
            torch.randn(num_nodes, dtype=torch.float32),
            requires_grad=True
        )
        self.w2 = torch.nn.Linear(f_in, num_nodes, bias=False)
        self.w3 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32),
            requires_grad=True
        )
        self.be = torch.nn.Parameter(
            torch.randn(1, c_in, c_in, dtype=torch.float32),
            requires_grad=True
        )
        self.ve = torch.nn.Parameter(
            torch.zeros(c_in, c_in, dtype=torch.float32),
            requires_grad=True
        )

        torch.nn.init.kaiming_uniform_(self.ve, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.be, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w1)
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(torch.matmul(x.transpose(2, 3), self.w1))
        y2 = torch.matmul(x, self.w3).transpose(1, 2)

        product = torch.matmul(y1, y2)
        E = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        E = F.softmax(E, dim=-1)
        return E

class GraphConv(torch.nn.Module):
    r"""
    Graph Convolution with self feature modeling.

    Args:
        f_in: input size.
        num_cheb_filter: output size.
        conv_type:
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.
        activation: default relu.
    """
    def __init__(self, f_in, num_cheb_filter, conv_type=None, **kwargs):
        super(GraphConv, self).__init__()
        self.K = kwargs.get('K', 3) if conv_type == 'cheb' else 1
        self.with_self = kwargs.get('with_self', True)
        self.w_conv = torch.nn.Linear(f_in * self.K, num_cheb_filter, bias=False)
        if self.with_self:
            self.w_self = torch.nn.Linear(f_in, num_cheb_filter)
        self.conv_type = conv_type
        self.activation = kwargs.get('activation', torch.relu)

    def cheb_conv(self, x, adj_mx):
        if adj_mx.dim() == 3:
            # h = x.unsqueeze(dim=1)
            # h = torch.matmul(adj_mx, h).transpose(1, 2).reshape(bs, num_nodes, -1)
            adj_mx = adj_mx.unsqueeze(dim=1)
            h_list = [x, torch.matmul(adj_mx, x)]
            for _ in range(2, self.K):
                h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])
            h = torch.cat(h_list, dim=-1)
        else:
            bs, num_nodes, _ = x.size()
            h_list = [x, torch.matmul(adj_mx, x)]
            for _ in range(2, self.K):
                h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])
            h = torch.cat(h_list, dim=-1)

        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def gcn_conv(self, x, adj_mx):
        h = torch.matmul(adj_mx, x)
        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def forward(self, x, adj_mx):
        self.conv_func = self.cheb_conv if self.conv_type == 'cheb' else self.gcn_conv
        return self.conv_func(x, adj_mx)

class GraphLearn(torch.nn.Module):
    """
    Graph Learning Modoel for AdapGL.

    Args:
        num_nodes: The number of nodes.
        init_feature_num: The initial feature number (< num_nodes).
    """
    def __init__(self, num_nodes, init_feature_num):
        super(GraphLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = torch.nn.Parameter(
            torch.rand(num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        self.w1 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )

        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, adj_mx):
        new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))
        attn = torch.sigmoid(self.attn(torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx

class STGNetBlock(torch.nn.Module):
    def __init__(self, c_in, f_in, num_nodes, num_cheb_filter, num_time_filter, kernel_size,
                 conv_type, K=3):
        super(STGNetBlock, self).__init__()

        self.padding = (kernel_size - 1) // 2
        self.graph_conv_p = GraphConv(f_in, num_cheb_filter // 2, conv_type=conv_type,
                                      K=K, activation=None, with_self=False)
        self.graph_conv_n = GraphConv(f_in, num_cheb_filter // 2, conv_type=conv_type,
                                      K=K, activation=None, with_self=False)
        self.temporal_att = TemporalAttention(num_nodes, f_in, c_in)

        self.time_conv = torch.nn.Conv2d(
            in_channels=num_cheb_filter,
            out_channels=num_time_filter,
            kernel_size=(1, kernel_size),
            padding=(0, self.padding)
        )

        self.residual_conv = torch.nn.Conv2d(
            in_channels=f_in,
            out_channels=num_time_filter,
            kernel_size=(1, 1)
        )

        self.ln = torch.nn.LayerNorm(num_time_filter)

    def forward(self, x, adj_mx):
        b, c, n_d, f = x.size()

        temporal_att = self.temporal_att(x)
        x_tat = (torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f))
        if adj_mx.dim() == 3:
            hp = self.graph_conv_p(x_tat, adj_mx)
            hn = self.graph_conv_n(x_tat, adj_mx.transpose(1,2))
            h = torch.relu(torch.cat((hp, hn), dim=-1))
        else:
            hp = self.graph_conv_p(x_tat.reshape(-1, n_d, f), adj_mx)
            hn = self.graph_conv_n(x_tat.reshape(-1, n_d, f), adj_mx.T)
            h = torch.relu(torch.cat((hp, hn), dim=-1).reshape(b, c, n_d, -1))



        h = self.time_conv(h.transpose(1, 3)).transpose(1, 3)
        h_res = self.residual_conv(x.transpose(1, 3)).transpose(1, 3)

        h = torch.relu(h + h_res)
        return self.ln(h)

class STGNet(torch.nn.Module):

    def __init__(self, **kwargs):
        super(STGNet, self).__init__()

        num_block = kwargs.get('num_block', 2)
        num_nodes = kwargs.get('num_nodes', None)
        c_in = kwargs.get('step_num_in', 12)
        c_out = kwargs.get('step_num_out', 12)
        f_in = kwargs.get('input_size', 1)
        kernel_size = kwargs.get('kernel_size', 3)
        num_time_filter = kwargs.get('num_time_filter', 64)
        num_cheb_filter = kwargs.get('num_cheb_filter', 64)
        conv_type = kwargs.get('conv_type', 'gcn')
        K = kwargs.get('K', 1)

        activation = kwargs.get('activation', 'relu')
        activation = getattr(torch, activation)

        self.epsilon = 1 / num_nodes * 0.5
        self.block_list = torch.nn.ModuleList()
        for i in range(num_block):
            temp_h = f_in if i == 0 else num_time_filter
            self.block_list.append(STGNetBlock(
                c_in, temp_h, num_nodes, num_cheb_filter,
                num_time_filter, kernel_size, conv_type, K=K
            ))

        self.final_conv = torch.nn.Conv2d(c_in, c_out, (1, num_time_filter))
        self.dropout = nn.Dropout(0.5, inplace=True)#dropout可以进行调整
        self.nodes = nn.Sequential(
            nn.Conv2d(f_in, 64, kernel_size=(1, 1)),#输入写为三
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 6, kernel_size=(1, c_in))#输出维度可以进行调整
        )
        # self.attn = torch.nn.Conv2d(f_in, num_nodes, kernel_size=(1,c_in))
        self.SAt = SpatialAttention( num_nodes, f_in, c_in)

        # self.adj_conv = torch.nn.Conv2d(3, 6, (1, c_in))#设定输出维度
    def forward(self, x, adj_mx):
        nodes = self.nodes(x.transpose(1, 3)).squeeze(3).transpose(1, 2)
        # attn = torch.relu(self.SAt(x))
        # attn = torch.relu(self.SAt(x.transpose(1, 3)).unsqueeze(dim=0)).squeeze().mean(axis=0)
        self.dropout(nodes)
        m = nodes
        A_mi = torch.einsum('bud,bvd->buv', [m, m])
        # A_mi = torch.sigmoid(A_mi )
        # # new_adj_mx = attn*A_mi +  adj_mx
        # # A_adj =F.normalize( F.relu(new_adj_mx-0.05) , p=1, dim=-1)
        A_mi = F.relu(F.normalize(A_mi, p=1, dim=-1)-0.05)
        A_adj = F.normalize(F.relu(A_mi + adj_mx ), p=1, dim=-1)
        h = x
        for net_block in self.block_list:
            h = net_block(h, A_adj)
        h = self.final_conv(h).squeeze(dim=-1)
        return h

    def __str__(self):
        return 'STGNet'
