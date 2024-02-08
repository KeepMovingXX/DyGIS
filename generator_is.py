from models import ISG
import torch
import numpy as np
from utils import random_graph_gen, dense_list_to_edge, random_graph_gen_dirct
from tqdm.auto import tqdm


def random_loader(dataset, edge_list, node_num):
    if dataset in ['soc-wiki-elec', 'ia-contacts_dublin']:
        random_edge_list = random_graph_gen_dirct(edge_list, node_num)
    else:
        random_edge_list = random_graph_gen(edge_list, node_num)
    edge_all_list, random_all_list = [], []
    for i in range(len(edge_list)):
        random_all_list.append(torch.tensor(np.transpose(random_edge_list[i]), dtype=torch.long))
    return random_all_list


def generate_is_subgraph(args, x, edge_list, adj_orig_dense_list, node_num):
    random_edge_list = random_loader(args.dataset, edge_list, node_num)
    if args.dataset in ['soc-wiki-elec', 'ia-contacts_dublin']:
        x_dim = args.lea_feature_dim
    elif args.dataset in ['cora']:
        x_dim = 1433
    else:
        x_dim = node_num

    model = ISG(x_dim, args.h_dim, args.z_dim, args.n_layers, args.eps, args.dropout, args.device, args.tau,
                  args.therold,
                  u=args.u, conv=args.conv_type, lea_feature=args.lea_feature, num=node_num)# lea_feature=args.lea_feature,
    model = model.to(args.device)
    x = x.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_shots = list(range(0, len(edge_list)))
    print('begin informative subgraph generating...')
    for epo in tqdm(range(args.epochs)):
        model.train()
        loss_list = []
        h = None
        for t in train_shots:
            if args.u:
                u = torch.zeros((node_num, 1)).to(args.device) + t
            else:
                u = None
            edge_list_t = torch.tensor(edge_list[t].transpose(), dtype=torch.long)
            random_edge_list_t = random_edge_list[t]
            adj_orig_dense_list_t = adj_orig_dense_list[t]
            optimizer.zero_grad()
            hidden_st, _, _, kld_loss, graph_distance_loss, mi_loss, adj_mask, adj_perturb = model(
                x, edge_list_t, random_edge_list_t, adj_orig_dense_list_t, u, None, h)
            loss = graph_distance_loss + kld_loss + args.trade_weight * mi_loss
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            h = hidden_st
    edge_all_list, edge_bias_list, edge_informative_list = [], [], []
    test_h =None

    for tt in train_shots:
        if args.u:
            u = torch.zeros((node_num, 1)).to(args.device) + t
        else:
            u = None
        edge_list_t = torch.tensor(edge_list[t].transpose(), dtype=torch.long)
        random_edge_list_t = random_edge_list[tt]
        adj_orig_dense_list_t = adj_orig_dense_list[tt]
        model.eval()
        hidden_st, _, _, _, _, _, all_adj_bias, all_adj_informative = model(x,  edge_list_t, random_edge_list_t, adj_orig_dense_list_t,u,
                                                                None, test_h)
        test_h = hidden_st
        edge_bias_index, edge_informative_index = dense_list_to_edge(all_adj_bias, all_adj_informative)
        edge_bias_list.append(edge_bias_index)
        edge_informative_list.append(edge_informative_index)
        edge_all_list.append(torch.tensor(edge_list[tt].transpose(), dtype=torch.long))

    return edge_all_list, edge_bias_list, edge_informative_list


