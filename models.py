import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv


class GCN_indot(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(GCN_indot, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.deoder_GCN = GCNConv(z_dim, h_dim)
    def forward(self, z, edge_index_t, edge_all_t = None):
        if edge_all_t == None:
            z_decode = self.deoder_GCN(z, edge_index_t)
            value = (z[edge_index_t[0]] * z[edge_index_t[1]]).sum(dim=1)
        else:
            z_decode = self.deoder_GCN(z, edge_index_t)
            value = (z[edge_all_t[0]] * z[edge_all_t[1]]).sum(dim=1)

        return z_decode, torch.sigmoid(value)
class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)

class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, device, bias=True):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.device = torch.device(device)

        # gru weights
        self.weight_xz = torch.nn.ModuleList()
        self.weight_hz = torch.nn.ModuleList()
        self.weight_xr = torch.nn.ModuleList()
        self.weight_hr = torch.nn.ModuleList()
        self.weight_xh = torch.nn.ModuleList()
        self.weight_hh = torch.nn.ModuleList()
        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size))
                self.weight_xr.append(GCNConv(input_size, hidden_size))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size))
                self.weight_xh.append(GCNConv(input_size, hidden_size))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size(), device=self.device)
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        out = h_out
        return out, h_out


class ISG(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, dropout, device, tau, therold, u, conv='GCN', bias=False, lea_feature=False, num=2708):
        super(ISG, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.enconv = conv
        self.activte = nn.Sigmoid()
        self.fea = Parameter((torch.ones(num, x_dim)), requires_grad=True)
        self.learnable_feature = lea_feature
        self.max_num_nodes = num
        self.tau = tau
        self.u = u
        self.therold = therold
        if lea_feature:
            self.eps1 = nn.Parameter(torch.FloatTensor(size=(num, z_dim)).normal_())
        else:
            self.eps1 = nn.Parameter(torch.FloatTensor(size=(x_dim, z_dim)).normal_())
        self.device = torch.device(device)
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim, self.z_dim), nn.BatchNorm1d(self.z_dim),
                                       nn.Dropout(self.dropout), nn.ReLU(), nn.Linear(self.z_dim, self.max_num_nodes),
                                       nn.Sigmoid())

        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        if conv == 'GCN':
            self.enc = GCNConv(h_dim + h_dim, h_dim)
            if self.u :
                self.enc_mean = GCNConv(h_dim + 1, z_dim)
                self.enc_std = GCNConv(h_dim + 1, z_dim)
            else:
                self.enc_mean = GCNConv(h_dim, z_dim)
                self.enc_std = GCNConv(h_dim, z_dim)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, self.device,  bias)

            self.encoder_random_1 = GCNConv(self.h_dim, self.z_dim)
            self.encoder_random_2 = GCNConv(self.z_dim, self.z_dim)
        elif conv == 'SAGE':
            pass

        elif conv == 'GIN':
            pass

    # encoder
    def encoder(self, feature, edge_index, h, u, y_cf):
        enc_t = F.relu(self.enc(torch.cat([feature, h[-1]], 1), edge_index))
        enc_t = F.dropout(enc_t, self.dropout)
        if u == None :
            enc_mean_t = self.enc_mean(enc_t, edge_index)
            enc_std_t = F.softplus(self.enc_std(enc_t, edge_index))
        else:
            enc_mean_t = self.enc_mean(torch.cat((enc_t, u), dim=1), edge_index)
            enc_std_t = F.softplus(self.enc_std(torch.cat((enc_t, u), dim=1), edge_index))

        return enc_mean_t, enc_std_t
    # prior
    def prior_params(self, h, u):
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        return prior_mean_t, prior_std_t

    def decoder(self, feature, y_cf, edge_index, u):
        adj_recons = self.decoder_a(feature)
        return adj_recons

    def random_encoder(self, feature, edge_index, u):
        enc_random_t = self.encoder_random_1(feature, edge_index)
        enc_random_t = F.relu(enc_random_t)
        enc_random_t = F.dropout(enc_random_t, self.dropout)
        enc_random_t = self.encoder_random_2(enc_random_t, edge_index)
        return enc_random_t

    def forward(self, x, edge_all_list_t, edge_random_list_t, adj_orig_dense_list_t, u, y_cf, hidden_in=None, pridiction=False):
        kld_loss = 0
        graph_distance_loss = 0
        mi_loss = 0
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim)).to(self.device)
        else:
            h = Variable(hidden_in).to(self.device)

        # to device
        edge_all_list_t = edge_all_list_t.to(self.device)
        edge_random_list_t = edge_random_list_t.to(self.device)
        adj_orig_dense_list_t = adj_orig_dense_list_t.to(self.device)

        # get x by mlp
        if self.learnable_feature:
            phi_x_t = self.phi_x(self.fea)
        else:
            phi_x_t = self.phi_x(x)

        # encoder
        enc_mean_t, enc_std_t = self.encoder(phi_x_t, edge_all_list_t, h, u, y_cf)

        # prior
        prior_mean_t, prior_std_t = self.prior_params(h, y_cf)

        # sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = self.phi_z(z_t)

        # recurrence
        _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_all_list_t, h)

        z_random_t, z_random_t_std = self.encoder(phi_x_t, edge_random_list_t, h, u, y_cf)
        adj_recon_p = self.decoder(z_t, z_random_t, edge_all_list_t, u)

        kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
        graph_distance_loss += self.weight_graph_rec(adj_recon_p, adj_orig_dense_list_t)
        mi_loss += self.mutualinfo_loss(z_t, z_random_t)

        cf_adjs = torch.mul(adj_recon_p + self.eps, adj_orig_dense_list_t)
        cf_adjs_edge_value = cf_adjs[torch.nonzero(cf_adjs)[:,0], torch.nonzero(cf_adjs)[:,1]]
        value_sort, _= cf_adjs_edge_value.sort()
        th_value = value_sort[int(value_sort.shape[0] * self.therold)]

        cf_adjs_bias = torch.where(cf_adjs >= th_value, torch.ones_like(cf_adjs),
                                        torch.zeros_like(cf_adjs))
        cf_adjs_informative = torch.where((0 < cf_adjs) & (cf_adjs < th_value), torch.ones_like(cf_adjs),
                                            torch.zeros_like(cf_adjs))

        return h, z_t, z_random_t, kld_loss, graph_distance_loss, mi_loss, cf_adjs_bias, cf_adjs_informative

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1).to(self.device)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                       (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) /
                       torch.pow(std_2 + self.eps, 2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def weight_graph_rec(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0,1])
        return - nll_loss

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def mutualinfo_loss(self, z1: torch.Tensor, z2: torch.Tensor):

        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        return loss.mean()


class DGMAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps,  dropout, device, conv='GCN', decoder_type = 'GCN', bias=False, lea_feature=False, num=2708):
        super(DGMAE, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.enconv = conv
        self.deconv = decoder_type
        self.activte = nn.Sigmoid()
        self.fea = Parameter((torch.ones(num, x_dim)), requires_grad=True)
        self.learnable_feature = lea_feature
        if lea_feature:
            self.eps1 = nn.Parameter(torch.FloatTensor(size=(num, z_dim)).normal_())
        else:
            # self.eps1 = nn.Parameter(torch.FloatTensor(size=(x_dim, z_dim)).normal_())
            self.eps1 = nn.Parameter(torch.FloatTensor(size=(num, z_dim)).normal_())
        self.device = torch.device(device)
        self.decoder2 = nn.Linear(z_dim+z_dim, z_dim)
        if conv == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

            self.enc = GCNConv(h_dim + h_dim, h_dim)
            self.enc_mean = GCNConv(h_dim, z_dim)
            self.enc_std = GCNConv(h_dim, z_dim)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, self.device,  bias)

        elif conv == 'SAGE':
            pass

        elif conv == 'GIN':
            pass

        if decoder_type == 'innerdot':
            self.deccode = InnerProductDecoder(act=lambda x: x)
        elif decoder_type == 'mlp':
            self.deccode = nn.Sequential(nn.Linear(z_dim, h_dim))
        elif decoder_type == 'GCN':
            self.deccode = GCN_indot(self.z_dim, self.h_dim)

    def forward(self, x, edge_all_list, edge_idx_list, edge_droped_idx_list, hidden_in=None, pridiction=False):

        kld_loss = 0
        org_rec_loss = 0
        mask_rec_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_t, all_z_t = [], []
        all_test_rec_h = []

        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(self.device)
        else:
            h = Variable(hidden_in).to(self.device)

        for t in range(x.size(0)):

            edge_all_list[t] = edge_all_list[t].to(self.device)
            edge_idx_list[t] = edge_idx_list[t].to(self.device)
            edge_droped_idx_list[t] = edge_droped_idx_list[t].to(self.device)

            if self.learnable_feature:
                phi_x_t = self.phi_x(self.fea)
            else:
                phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = F.relu(self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t]))
            enc_t = F.dropout(enc_t, self.dropout)
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = F.softplus(self.enc_std(enc_t, edge_idx_list[t]))

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            if self.deconv =='innerdot':
                dec_t = self.deccode(z_t)
                kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                mask_rec_loss += self.mask_rec(z_t, edge_all_list[t], edge_droped_idx_list[t])
                # org_rec_loss += self.mask_rec(z_t, edge_all_list[t], edge_idx_list[t])
                all_dec_t.append(z_t)

            elif self.deconv in ('mlp'):
                dec_t = self.deccode(z_t)
                kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                mask_rec_loss += self.mask_rec(dec_t, edge_all_list[t], edge_droped_idx_list[t])
                # org_rec_loss += self.mask_rec(dec_t, edge_all_list[t], edge_all_list[t])
                all_dec_t.append(dec_t)

            elif self.deconv == 'GCN':
                # 这个地方传的目的在于，为了计算全局的重构损失，就是直接计算全局的重构 pos值，mlp那个地方没有算，是所有的一块集中算的，
                # 之前mlp效果差 是因为self.mask_rec(z_t)了，应该是dec_t, z_t是编码的embedding值，需要使用mlp解码的dec来操作
                z_dec, pos_value = self.deccode(z_t, edge_idx_list[t], edge_all_t = edge_all_list[t] )
                # neg_edge_index = negative_sampling(edge_all_list[t], z_t.size(0), edge_all_list[t].shape[1])
                # neg_value = torch.sigmoid((z_dec[neg_edge_index[0]] * z_dec[neg_edge_index[1]]).sum(dim=1))
                # org_rec_loss += self.rec_loss(pos_value, neg_value)
                kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
                mask_rec_loss += self.mask_rec(z_dec, edge_all_list[t], edge_droped_idx_list[t])
                all_dec_t.append(z_dec)

            # use hidden state to pri and new pri
            if pridiction == True:
                z_hiden_t,_ = self.deccode(prior_mean_t, edge_idx_list[t])
                all_test_rec_h.append(z_hiden_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_z_t.append(z_t)
        return kld_loss, org_rec_loss, mask_rec_loss, all_enc_mean, all_test_rec_h, all_dec_t, h, all_z_t


    def _reparameterized_sample(self, mean, std):
        return self.eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                       (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) /
                       torch.pow(std_2 + self.eps, 2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def mask_rec(self, z, edge_all_list, edge_droped_idx_list):
        pos_loss = -torch.log(self.activte((z[edge_droped_idx_list[0]] * z[edge_droped_idx_list[1]]).sum(dim=1)) + 1e-10).mean()
        neg_edge_index = negative_sampling(edge_all_list, z.size(0), edge_droped_idx_list.shape[1])
        neg_loss = -torch.log(1 -
                              self.activte((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-10).mean()
        return pos_loss + neg_loss

    def rec_loss(self, pos_value, neg_value):
        pos_loss = -torch.log(pos_value + 1e-10).mean()
        neg_loss = -torch.log(1 - neg_value + 1e-10).mean()
        return pos_loss + neg_loss

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, m_dim, out_dim, drop_rate):
        super().__init__()
        self.linear1 = nn.Linear(num_dim, m_dim)
        self.linear2 = nn.Linear(m_dim, out_dim)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()
        self.drop_rate = drop_rate
    def forward(self, x, *args):
        logits = self.linear2(x)
        return logits






