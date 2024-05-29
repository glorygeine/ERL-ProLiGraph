import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool as gep
from torch_geometric.nn import SuperGATConv
from torch_geometric.utils import dropout_adj


# GraphSAGE_SuperGAT Model
class SAGE_SuperGAT(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_lig=78, output_dim=128, dropout=0.2):
        super(SAGE_SuperGAT, self).__init__()

        print('GraphSAGE_SuperGAT Loaded')
        self.n_output = n_output
        self.lig_c1 = SAGEConv(num_features_lig, num_features_lig)
        self.lig_c2 = SAGEConv(num_features_lig, num_features_lig * 2)
        self.lig_c3 = SuperGATConv(num_features_lig * 2, num_features_lig * 4)
        self.lig_fc_g1 = torch.nn.Linear(num_features_lig * 4, 1024)
        self.lig_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_c1 = SAGEConv(num_features_pro, num_features_pro)
        self.pro_c2 = SAGEConv(num_features_pro, num_features_pro * 2)
        self.pro_c3 = SuperGATConv(num_features_pro * 2, num_features_pro * 4)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_lig, data_pro):
        # get graph input
        lig_x, lig_edge_index, lig_batch = data_lig.x, data_lig.edge_index, data_lig.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch


        x = self.lig_c1(lig_x, lig_edge_index)
        x = self.relu(x)

        x = self.lig_conv2(x, lig_edge_index)
        x = self.relu(x)

        x = self.lig_c3(x, lig_edge_index)
        x = self.relu(x)
        x = gep(x, lig_batch)  # global pooling

        # flatten
        x = self.relu(self.lig_fc_g1(x))
        x = self.dropout(x)
        x = self.lig_fc_g2(x)
        x = self.dropout(x)

        xt = self.pro_c1(target_x, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_c2(xt, target_edge_index)
        xt = self.relu(xt)

        xt = self.pro_c3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
