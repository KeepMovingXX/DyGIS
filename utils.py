import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from torch.nn import functional as F


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def load_dataset(dataset):
    with open('../data/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pkl.load(handle, encoding='latin1')

    with open('../data/{}/adj_orig_dense_list.pickle'.format(dataset), 'rb') as handle:
        adj_orig_dense_list = pkl.load(handle, encoding='bytes')
    return adj_time_list, adj_orig_dense_list
def load_cora(dataset):
    # 特征输入
    x = torch.tensor(np.loadtxt('../data/{}/features/features.txt'.format(dataset)),dtype=torch.float)
    with open('../data/{}/adj_time_list.pickle'.format(dataset), 'rb') as handle:
        adj_time_list = pkl.load(handle)
    with open('../data/{}/adj_orig_dense_list.pickle'.format(dataset), 'rb') as handle:
        adj_orig_dense_list = pkl.load(handle)

    return x, adj_time_list, adj_orig_dense_list

def load_labels(dataset):
    labels = np.loadtxt('../data/{}/labels/label_matrix.txt'.format(dataset)).astype(np.int16)
    if dataset in 'cora':
        num_class = labels.max()+1
    else:
        num_class = labels.max()
        labels = labels - 1
    return labels, num_class


def mask_edges_det(adjs_list, test_len):
    adj_train_l, train_edges_l, val_edges_l = [], [], []
    val_edges_false_l, test_edges_l, test_edges_false_l = [], [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0
        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges = sparse_to_tuple(adj)[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] / 10.))
        num_val = int(np.floor(edges.shape[0] / 20.))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        if i > (len(adjs_list) - (test_len+1) ) :
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        else:
            train_edges = edges

        edges_list.append(edges)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        # adj_train = sp.coo_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        adj_train_l.append(adj_train)
        train_edges_l.append(train_edges)
        val_edges_l.append(val_edges)
        val_edges_false_l.append(val_edges_false)
        test_edges_l.append(test_edges)
        test_edges_false_l.append(test_edges_false)
        node_num = int(np.max(np.vstack(edges_list))) + 1

    return adj_train_l, train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l, edges_list, node_num

def random_graph_gen(edge_list, node_num):
    ER_edge_list = []
    adj_time_list = []
    for i in range(0, len(edge_list)):
        edge_num_t = edge_list[i].shape[0]
        ER_graph_t = nx.gnm_random_graph(node_num, edge_num_t/2)
        if ER_graph_t.is_directed():
            ER_graph_t = ER_graph_t.to_undirected()
        ER_graph_t_coo = nx.to_scipy_sparse_array(ER_graph_t).tocoo()
        row, col, data = ER_graph_t_coo.row, ER_graph_t_coo.col, ER_graph_t_coo.data
        ER_edges_index = np.vstack((row, col)).transpose()
        ER_edge_list.append(ER_edges_index)
        adj_time_list.append(ER_graph_t_coo)

    return ER_edge_list

def random_graph_gen_dirct(edge_list, node_num):
    ER_edge_list = []
    adj_time_list = []
    for i in range(0, len(edge_list)):
        edge_num_t = edge_list[i].shape[0]
        ER_graph_t = nx.gnm_random_graph(node_num, edge_num_t)
        ER_edges_index = np.array(ER_graph_t.edges())
        ER_edge_list.append(ER_edges_index)
    return ER_edge_list

def dense_list_to_edge_list(all_adj_mask, all_adj_perturb):
    edge_mask_list, edge_perturb_list = [], []
    for i in range(len(all_adj_mask)):
        adj_mask_t_dense = all_adj_mask[i].cpu()
        adj_mask_t_coo = sp.coo_matrix(adj_mask_t_dense)
        print(i, 'mask_edge_num:', adj_mask_t_coo)
        mask_t_edge_index = torch.tensor(np.vstack((adj_mask_t_coo.row, adj_mask_t_coo.col)),dtype=torch.long)

        adj_perturb_t_dense = all_adj_perturb[i].cpu()
        adj_perturb_t_coo = sp.coo_matrix(adj_perturb_t_dense)
        perturb_t_edge_index = torch.tensor(np.vstack((adj_perturb_t_coo.row, adj_perturb_t_coo.col)),dtype=torch.long)
        print(i, 'perturb_edge_num:', adj_perturb_t_coo)
        edge_mask_list.append(mask_t_edge_index)
        edge_perturb_list.append(perturb_t_edge_index)

    return edge_mask_list, edge_perturb_list

def dense_list_to_edge(all_adj_mask_t, all_adj_perturb_t):


    adj_mask_t_dense = all_adj_mask_t.cpu()
    adj_mask_t_coo = sp.coo_matrix(adj_mask_t_dense)
    mask_t_edge_index = torch.tensor(np.vstack((adj_mask_t_coo.row, adj_mask_t_coo.col)),dtype=torch.long)
    # print('mask_edge_num:', adj_mask_t_coo)
    adj_perturb_t_dense = all_adj_perturb_t.cpu()
    adj_perturb_t_coo = sp.coo_matrix(adj_perturb_t_dense)
    # print('perturb_edge_num:', adj_perturb_t_coo)
    perturb_t_edge_index = torch.tensor(np.vstack((adj_perturb_t_coo.row, adj_perturb_t_coo.col)),dtype=torch.long)

    return mask_t_edge_index, perturb_t_edge_index

def get_roc_score(edges_pos, edges_neg, embs):
    auc_scores = []
    ap_scores = []

    for i in range(len(edges_pos)):
        emb_t = embs[i]
        edges_pos_t = torch.tensor(np.transpose(edges_pos[i]),dtype=torch.long, device= emb_t.device)
        edges_neg_t = torch.tensor(np.transpose(edges_neg[i]),dtype=torch.long, device= emb_t.device)
        pos_pred = torch.sigmoid((emb_t[edges_pos_t[0]] * emb_t[edges_pos_t[1]]).sum(dim=1))
        neg_pred = torch.sigmoid((emb_t[edges_neg_t[0]] * emb_t[edges_neg_t[1]]).sum(dim=1))
        pos_y = emb_t.new_ones(edges_pos_t.size(1))
        neg_y = emb_t.new_zeros(edges_neg_t.size(1))
        preds =torch.cat([pos_pred, neg_pred], dim=0)
        y = torch.cat([pos_y, neg_y], dim=0)
        y, pred = y.detach().cpu().numpy(), preds.detach().cpu().numpy()
        auc_scores.append(roc_auc_score(y, pred))
        ap_scores.append(average_precision_score(y, pred))

    return auc_scores, ap_scores

def mask_edges_prd(adjs_list):
    pos_edges_l, false_edges_l = [], []
    edges_list = []
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        # edges = sparse_to_tuple(adj)[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_false = int(edges.shape[0])

        pos_edges_l.append(edges)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(edges_false)

    return pos_edges_l, false_edges_l
def mask_edges_prd_new(adjs_list, adj_orig_dense_list):
    pos_edges_l, false_edges_l = [], []

    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    adj = adjs_list[0]
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_false = int(edges.shape[0])

    pos_edges_l.append(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    edges_false = []
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if edges_false:
            if ismember([idx_j, idx_i], np.array(edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(edges_false)):
                continue
        edges_false.append([idx_i, idx_j])

    assert ~ismember(edges_false, edges_all)
    false_edges_l.append(np.asarray(edges_false))

    for i in range(1, len(adjs_list)):
        edges_pos = np.transpose(np.asarray(np.where((adj_orig_dense_list[i] - adj_orig_dense_list[i - 1]) > 0)))
        num_false = int(edges_pos.shape[0])

        adj = adjs_list[i]
        # Remove diagonal elements
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        # Check that diag is zero:
        assert np.diag(adj.todense()).sum() == 0

        adj_triu = sp.triu(adj)
        adj_tuple = sparse_to_tuple(adj_triu)
        edges = adj_tuple[0]
        edges_all = sparse_to_tuple(adj)[0]

        edges_false = []
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            edges_false.append([idx_i, idx_j])

        assert ~ismember(edges_false, edges_all)

        false_edges_l.append(np.asarray(edges_false))
        pos_edges_l.append(edges_pos)

    return pos_edges_l, false_edges_l

def get_acc_score(y_pred_list, y_true_list):
    acc_score = []
    for i in range(y_pred_list.shape[0]):
        y_pred_t = y_pred_list[i]
        y_true_t = y_true_list[i]
        y, pred = y_true_t.detach().cpu().numpy(), y_pred_t.detach().cpu().numpy()
        acc_score.append(accuracy_score(y, pred))
    return np.array(acc_score).mean()
