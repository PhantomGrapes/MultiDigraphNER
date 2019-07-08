import torch
import torch.nn as nn
from src.utils.gpu_tools import move2cuda


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, layerUtil):
        super(GGNN, self).__init__()

        self.state_dim = layerUtil.getGraphDim()
        self.n_edge_types = layerUtil.getEdgeTypes()
        self.n_steps = layerUtil.Config.model.graph_emb.n_steps
        self.dropout = layerUtil.getDropOut()
        self.useAttention = layerUtil.Config.model.graph_emb.attention

        self.fcs = nn.ModuleList([nn.Linear(self.state_dim, self.state_dim) for i in range(self.n_edge_types)])

        # Propogation Model
        self.propogator = layerUtil.getPropagator(self.state_dim, self.n_edge_types)

        self.softmax = nn.Softmax(1)

        self._initialization()
        self.gaWeight = nn.Parameter(torch.FloatTensor(layerUtil.initGa))
        self.gaSigmoid = nn.Sigmoid()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, A0, n_node):
        gaWeightSoft = self.gaSigmoid(self.gaWeight)
        A = torch.cat([A0[:, :, n_node * edgeIdx: n_node * (edgeIdx + 1)] * gaWeightSoft[edgeIdx] for edgeIdx in range(self.n_edge_types)], 2)
        self.propogator.init(A.shape[0], A.shape[1])
        for i_step in range(self.n_steps):
            states = []
            for i in range(self.n_edge_types):
                states.append(self.fcs[i](prop_state))

            prop_state = self.propogator(states, prop_state, A, n_node)

        return prop_state
