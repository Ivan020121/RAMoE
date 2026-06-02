"""
This file includes modified code from:
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018
Paper: https://arxiv.org/abs/1801.07455
Original implementation: https://github.com/wanjinchang/st-gcn

Modifications:
- Remove adaptation for multiple layouts.

Original code is licensed under the BSD-2-Clause license.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph:
    def __init__(self, max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge()
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):
        self.num_node = 17
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            # 头部
            (0, 1),
            (1, 3),  # Nose -> Left Eye -> Left Ear
            (0, 2),
            (2, 4),  # Nose -> Right Eye -> Right Ear
            # 躯干
            (5, 6),  # Left Shoulder <-> Right Shoulder
            (5, 11),  # Left Shoulder -> Left Hip
            (6, 12),  # Right Shoulder -> Right Hip
            (11, 12),  # Left Hip <-> Right Hip
            # 左侧肢体
            (5, 7),  # Left Shoulder -> Left Elbow
            (7, 9),  # Left Elbow -> Left Wrist
            (11, 13),  # Left Hip -> Left Knee
            (13, 15),  # Left Knee -> Left Ankle
            # 右侧肢体
            (6, 8),  # Right Shoulder -> Right Elbow
            (8, 10),  # Right Elbow -> Right Wrist
            (12, 14),  # Right Hip -> Right Knee
            (14, 16),  # Right Knee -> Right Ankle
        ]
        self.edge = self_link + neighbor_link
        self.center = 1

    def get_adjacency(self):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        A = []
        for hop in valid_hop:
            a_root = np.zeros((self.num_node, self.num_node))
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if self.hop_dis[j, i] == hop:
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif (
                            self.hop_dis[j, self.center] > self.hop_dis[i, self.center]
                        ):
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
            if hop == 0:
                A.append(a_root)
            else:
                A.append(a_root + a_close)
                A.append(a_further)
        self.A = np.stack(A)


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


class st_gcn(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class STGCN_Encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(
        self, in_channels, edge_importance_weighting, max_hop=1, dilation=1, **kwargs
    ):
        super().__init__()
        self.name = "st_gcn"

        # load graph
        self.graph = Graph(max_hop, dilation)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 128, kernel_size, 2, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 256, kernel_size, 2, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
                st_gcn(256, 512, kernel_size, 2, **kwargs),
            )
        )

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        multiple_perspectives = False if x.ndim == 4 else True
        # data normalization
        if multiple_perspectives:
            N, C, T, V, M = x.size()
            x = x.permute(0, 4, 3, 1, 2).contiguous()
            x = x.view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T)
            x = x.permute(0, 1, 3, 4, 2).contiguous()
            x = x.view(N * M, C, T, V)
        else:
            N, C, T, V = x.size()
            x = x.permute(0, 3, 1, 2).contiguous()
            x = x.view(N, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, V, C, T)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(N, C, T, V)
        

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        if multiple_perspectives:
            x = x.view(N, M, -1, 1, 1).mean(dim=1).view(N, -1)
        else:
            x = torch.flatten(x, 1)
        return x
