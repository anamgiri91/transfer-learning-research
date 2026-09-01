import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet

class SchNetRegressor(nn.Module):
    """
    3D Geometric Deep Learning Model (SchNet) for predicting affinity
    directly from atomic coordinates.
    """
    def __init__(self, hidden_channels=64, num_filters=64, num_interactions=3):
        super(SchNetRegressor, self).__init__()
        # We use SchNet to get graph-level embeddings.
        # By default SchNet outputs per-atom predictions, so we use readout.
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            readout='mean'
        )
        
    def forward(self, z, pos, batch=None):
        """
        z: Atomic numbers [num_atoms]
        pos: Coordinates [num_atoms, 3]
        batch: Batch indices [num_atoms]
        """
        return self.schnet(z, pos, batch=batch)
