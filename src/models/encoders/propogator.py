import torch
import torch.nn as nn

class GRUProp(nn.Module):
    """
    Gated Propogator for GGNN
    Using GRU gating mechanism
    """
    def __init__(self, state_dim, n_edge_types, useGpu):
        super(GRUProp, self).__init__()

        self.n_edge_types = n_edge_types
        self.state_dim = state_dim

        self.cell = nn.GRUCell(state_dim * self.n_edge_types, state_dim)
        self.useGpu = useGpu

    def forward(self, states, state_cur, A, n_node):
        batchSize = state_cur.shape[0]
        state_dim = state_cur.shape[-1]

        flows = []
        for typeIdx in range(self.n_edge_types):
            flows.append(torch.bmm(A[:, :, n_node * typeIdx: n_node * (typeIdx + 1)], states[typeIdx]))
        a = torch.cat(flows, 2)

        output = self.cell(a.view(batchSize * n_node, state_dim * self.n_edge_types), state_cur.view(batchSize * n_node, state_dim))
        return output.view(batchSize, n_node, state_dim)

    def init(self, batchSize, n_node):
        pass


