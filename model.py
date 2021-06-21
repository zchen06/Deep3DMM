import torch
import torch.nn.functional as F

from layers import ChebConv_Coma, Pool
from attpool import AttPool

class Coma(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        self.n_layers = config['n_layers']
        self.filters_enc = config['filter_enc']
        self.filters_dec = config['filter_dec']
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.cheb_enc = torch.nn.ModuleList([ChebConv_Coma(self.filters_enc[i], self.filters_enc[i+1], self.K[i])
                                             for i in range(len(self.filters_enc)-1)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters_dec[i], self.filters_dec[i+1], self.K[i])
                                             for i in range(len(self.filters_dec)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters_enc[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters_dec[0]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters_enc[0])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, self.filters_enc[0])
        return x

    def enc_forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters_enc[0])
        x = self.encoder(x)
        return x

    def dec_forward(self, data):
        x = self.decoder(data)
        x = x.reshape(-1, self.filters_enc[0])
        return x

    def encoder(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i]))
            x = self.pool(x, self.downsample_matrices[i])
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x

    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters_dec[0])
        for i in range(self.n_layers):
            x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[0], self.A_norm[0])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
        torch.nn.init.constant_(self.enc_lin.bias, 0.1)
        torch.nn.init.constant_(self.dec_lin.bias, 0.1)


class ComaAtt(torch.nn.Module):

    def __init__(self, dataset, config, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes, V_ref):
        super(ComaAtt, self).__init__()
        self.n_layers = config['n_layers']
        self.filters_enc = config['filter_enc']
        self.filters_dec = config['filter_dec']
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_att = config['downsample_att']
        self.upsample_att = config['upsample_att']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        
        self.cheb_enc = torch.nn.ModuleList([ChebConv_Coma(self.filters_enc[i], self.filters_enc[i+1], self.K[i])
                                             for i in range(len(self.filters_enc)-1)])
        self.cheb_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters_dec[i], self.filters_dec[i+1], self.K[i])
                                             for i in range(len(self.filters_dec)-1)])
        self.cheb_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool()
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters_enc[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters_dec[0]*self.upsample_matrices[-1].shape[1])

        if self.upsample_att is True or self.downsample_att is True:
            self.attpoolenc, self.attpooldec = self.init_attpool(config, V_ref)
            self.attpoolenc = torch.nn.ModuleList(self.attpoolenc)
            self.attpooldec = torch.nn.ModuleList(self.attpooldec)
        self.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters_enc[0])
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.reshape(-1, self.filters_enc[0])
        return x

    def enc_forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        x = x.reshape(batch_size, -1, self.filters_enc[0])
        x = self.encoder(x)
        return x

    def dec_forward(self, data):
        x = self.decoder(data)
        x = x.reshape(-1, self.filters_enc[0])
        return x


    def encoder(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.cheb_enc[i](x, self.A_edge_index[i], self.A_norm[i]))
            if self.downsample_att is True:
                x = self.attpoolenc[i](x)
            else:
                x = self.pool(x, self.downsample_matrices[i])
        x = x.reshape(x.shape[0], self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x


    def decoder(self, x):
        x = F.relu(self.dec_lin(x))
        x = x.reshape(x.shape[0], -1, self.filters_dec[0])
        for i in range(self.n_layers):
            if self.upsample_att is True:
                x = self.attpooldec[-i-1](x)
            else:
                x = self.pool(x, self.upsample_matrices[-i-1])
            x = F.relu(self.cheb_dec[i](x, self.A_edge_index[self.n_layers-i-1], self.A_norm[self.n_layers-i-1]))
        x = self.cheb_dec[-1](x, self.A_edge_index[0], self.A_norm[0])
        return x


    def init_attpool(self, config, V_ref=None):
        attpooldec = []
        attpoolenc = []
        if V_ref is not None:
            for i in range(len(V_ref)-1):
                idx_k, idx_q = i, i+1
                k_init, q_init = V_ref[idx_k].t(), V_ref[idx_q].t()
                config_enc = config
                config_enc['num_k'], config_enc['num_q'] = k_init.size(1), q_init.size(1)
                config_enc['top_k'] = config['top_k_enc'][min(i,len(config['top_k_enc'])-1)] \
                    if isinstance(config['top_k_enc'],list) else config['top_k_enc']
                attpoolenc.append(AttPool(config=config_enc, const_k=k_init, const_q=q_init,
                                          prior_map=self.downsample_matrices[i].to_dense()))
                config_dec = config
                config_dec['num_k'], config_dec['num_q'] = q_init.size(1), k_init.size(1)
                config_dec['top_k'] = config['top_k_dec'][min(i,len(config['top_k_dec'])-1)] \
                    if isinstance(config['top_k_dec'],list) else config['top_k_dec']
                attpooldec.append(AttPool(config=config_dec, const_k=q_init, const_q=k_init,
                                          prior_map=self.upsample_matrices[i].to_dense()))
        return attpoolenc, attpooldec
    

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
        torch.nn.init.constant_(self.enc_lin.bias, 0.1)
        torch.nn.init.constant_(self.dec_lin.bias, 0.1)
