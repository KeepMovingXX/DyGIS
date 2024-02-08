import argparse
from models import DGMAE
from utils import mask_edges_det, load_dataset
from utils import get_roc_score
from generator_is import generate_is_subgraph
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from time import time
import logging
import os


seed = 3
np.random.seed(seed)

parser = argparse.ArgumentParser(description='DyGIS')
parser.add_argument('--tune_description', type=str, default='', help='Dataset name')
parser.add_argument('--dataset', default='dblp', help='Dataset name')
parser.add_argument('--epochs', type=int, default=100, help='the number of ISG epochs to train')
parser.add_argument('--epochs_task', type=int, default=1000, help='the number of DGMAE epochs to train')
parser.add_argument('--lr', type=float, default=0.02, help='Initial learning rate of ISG')
parser.add_argument('--lr_task', type=float, default=0.005, help='Initial learning rate Of DGMAE')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay of ISG')
parser.add_argument('--weight_decay_task', type=float, default=1e-3, help='weight_decay of DGMAE')
parser.add_argument('--tau', type=float, default=0.7, help='the tempure hp')
parser.add_argument('--therold', type=float, default=0.1, help='informative subgraph ratio')
parser.add_argument('--h_dim', type=int, default=128, help='hidden dims of gcn ')
parser.add_argument('--z_dim', type=int, default=32, help='output dims of gcn ')
parser.add_argument('--trade_weight', type=float, default=0.5, help='the trade weight of MI loss')
parser.add_argument('--u',type=bool, default=False, help=' if use u')
parser.add_argument('--run_times', type=int, default=5, help='run times ')
parser.add_argument('--conv_type', type=str, default='GCN', help='encoder type: GCN, GraphSAGE, GIN ')
parser.add_argument('--decode_type', type=str, default='GCN', help='decoder type: GCN, mlp, innerdot ')
parser.add_argument('--device', type=str, default='cuda:3', help='device')
parser.add_argument('--lea_feature', type=bool, default=False, help='x feature for different datasets')
parser.add_argument('--lea_feature_dim', type=int, default=512, help='feature dim for different datasets')
parser.add_argument('--model_name', type=str, default='DyGIS', help='which model while be conducted')
parser.add_argument('--n_layers', type=int, default=1, help='the number of model layers')
parser.add_argument('--eps', type=int, default=1e-10)
parser.add_argument('--clip', type=int, default=10, help='clip the degrade')
parser.add_argument('--seq_start', type=int, default=0, help='the strat idx of train set ')
parser.add_argument('--spilt_len', type=int, default=3, help='the length of test')
parser.add_argument('--test_after', type=int, default=50, help='start test ')
parser.add_argument('--dropout', type=float, default=0.5, help='the rate of dropout  ')
parser.add_argument('--patience', type=int, default=150, help='patience for early stop')
parser.add_argument('--task', type=str, default='link_detection', help='which task is conducted')

args = parser.parse_args()

adj_time_list, adj_orig_dense_list = load_dataset(args.dataset)
outs = mask_edges_det(adj_time_list, args.spilt_len)
adj_train_l = outs[0]
train_edges_l = outs[1]
val_edges_l = outs[2]
val_edges_false_l = outs[3]
test_edges_l = outs[4]
test_edges_false_l = outs[5]
edge_list = outs[6]
num_nodes = outs[7]

seq_len = len(train_edges_l)
x_dim = num_nodes

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32))
    x_in_list.append(torch.tensor(x_temp))
x_in = Variable(torch.stack(x_in_list)).to(args.device)

log_path = f'logs/{args.task}/{args.model_name}/{args.dataset}'
log_fname = f'{log_path}/{args.model_name}_log.out'
os.makedirs(log_path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=log_fname,
    filemode='a'
    )

if args.dataset in ['soc-wiki-elec', 'ia-contacts_dublin']:
    args.lea_feature = True # using learning feature for these datasets
    x_dim = args.lea_feature_dim

test_auc_mean, test_ap_mean = [],[]
val_auc_mean, val_ap_mean = [], []
for repeat in range(args.run_times):
    patience = 0
    best_auc_val = 0
    best_ap_val = 0
    best_auc_test = 0
    best_ap_test = 0
    edge_all_list, edge_droped_idx_list, edge_idx_list= generate_is_subgraph(args, x_in_list[0], train_edges_l, adj_orig_dense_list,num_nodes)
    model = DGMAE(x_dim, args.h_dim, args.z_dim, args.n_layers, args.eps, args.dropout, args.device,
                  conv=args.conv_type, decoder_type=args.decode_type, bias=True, lea_feature=args.lea_feature, num=num_nodes)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_task, weight_decay=args.weight_decay_task)
    logging.info(args)
    start_time = time()
    for k in range(args.epochs_task):
        model.train()
        if patience > args.patience:
            print('early stopping')
            break
        optimizer.zero_grad()
        kld_loss, _, mask_rec_loss, _, _, _, hidden_st, _ = model(x_in[args.seq_start:(seq_len - args.spilt_len)]
                                                                         , edge_all_list[
                                                                           args.seq_start:(seq_len - args.spilt_len)]
                                                                         , edge_idx_list[
                                                                           args.seq_start:(seq_len - args.spilt_len)]
                                                                         , edge_droped_idx_list[
                                                                           args.seq_start:(seq_len - args.spilt_len)])
        loss = kld_loss + mask_rec_loss
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        if k > args.test_after:
            model.eval()
            _, _, _, enc_means, _, rec_adj, _, _ = model(x_in[(seq_len - args.spilt_len):seq_len]
                                                         , edge_all_list[(seq_len - args.spilt_len):seq_len]
                                                         , edge_all_list[(seq_len - args.spilt_len):seq_len]
                                                         , edge_droped_idx_list[(seq_len - args.spilt_len):seq_len]
                                                         , hidden_st)
            auc_scores_det_val, ap_scores_det_val = get_roc_score(val_edges_l[(seq_len - args.spilt_len):seq_len]
                                                                  , val_edges_false_l[(seq_len - args.spilt_len):seq_len]
                                                                  , rec_adj)

            auc_scores_det_test, ap_scores_det_tes = get_roc_score(test_edges_l[(seq_len - args.spilt_len):seq_len]
                                                                   , test_edges_false_l[(seq_len - args.spilt_len):seq_len]
                                                                   , rec_adj)

            auc_scores_det_val = np.mean(np.array(auc_scores_det_val))
            ap_scores_det_val = np.mean(np.array(ap_scores_det_val))
            auc_scores_det_test = np.mean(np.array(auc_scores_det_test))
            ap_scores_det_tes = np.mean(np.array(ap_scores_det_tes))

            if auc_scores_det_val > best_auc_val and ap_scores_det_val > best_ap_val:
                patience = 0
                best_auc_val = auc_scores_det_val
                best_ap_val = ap_scores_det_val
                best_auc_test = auc_scores_det_test
                best_ap_test = ap_scores_det_tes
                print('Link Detection')
                print('val_link_det_auc_mean', best_auc_val)
                print('val_link_det_ap_mean', best_ap_val)
                print('test_link_det_auc_mean', best_auc_test)
                print('test_link_det_ap_mean', best_ap_test)
                print('----------------------------------')
            else:
                patience += 1
        print('epoch: ', k)
        print('kld_loss =', kld_loss.mean().item())
        print('mask_rec_loss =', mask_rec_loss.mean().item())
        print('loss =', loss.mean().item())

    end_time = time()
    logging.info(f' Total training time... {end_time - start_time:.2f}s')
    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ("-----Link Detection------",
                 'Validation metrics:',
                 f'ROC-AUCs, AP-AUCs: {best_auc_val, best_ap_val}',
                 f'test metrics:',
                 f'ROC-AUCs, AP-AUCs: {best_auc_test, best_ap_test}',
                 f'---------------------------------\n',
                 )
            )
        )
    print('Link Detection')
    print('val_link_det_auc_mean', best_auc_val)
    print('val_link_det_ap_mean', best_ap_val)
    print('test_link_det_auc_mean', best_auc_test)
    print('test_link_det_ap_mean', best_ap_test)
    print('----------------------------------')
    val_auc_mean.append(best_auc_val)
    val_ap_mean.append(best_ap_val)
    test_auc_mean.append(best_auc_test)
    test_ap_mean.append(best_ap_test)

val_auc_mean_value,val_auc_mean_std, val_ap_mean_value, val_ap_mean_std = np.array(val_auc_mean).mean(), np.array(val_auc_mean).std(),\
                                      np.array(val_ap_mean).mean(),  np.array(val_ap_mean).std()
test_auc_mean_value, test_auc_mean_std, test_ap_mean_value, test_ap_mean_std = np.array(test_auc_mean).mean(), np.array(test_auc_mean).std(),\
                                        np.array(test_ap_mean).mean(), np.array(test_ap_mean).std()

with open(log_fname, 'a') as f:
    f.write(
        '\n'.join(
            (f'-----Run for Link Detection times: {args.run_times}',
             'Validation metrics muitiple times:',
             f'ROC-AUCs, AP-AUCs: {val_auc_mean_value, val_auc_mean_std, val_ap_mean_value, val_ap_mean_std}',
             f'test metrics muitiple times:',
             f'ROC-AUCs, AP-AUCs: {test_auc_mean_value, test_auc_mean_std, test_ap_mean_value, test_ap_mean_std}',
             f'---------------------------------\n',
             )
        )
    )
print(val_auc_mean_value,val_auc_mean_std, val_ap_mean_value, val_ap_mean_std,
      test_auc_mean_value, test_auc_mean_std, test_ap_mean_value, test_ap_mean_std)
