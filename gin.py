import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch.nn import Linear

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        mlp = torch.nn.Sequential(
            Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(mlp, train_eps=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=-1)