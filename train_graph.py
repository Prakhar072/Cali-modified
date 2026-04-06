import logging


from model import *
from scipy.spatial import distance

from logistic_regression_eval import *
from torch.optim import Adam
from tqdm import tqdm
from transforms import *

from linear_eval import *
from utils import *

from aug import TUDataset_aug as TUDataset
from aug import *
from torch_geometric.data import DataLoader

import random
import argparse
from time import perf_counter as t

from evaluate_embedding import *

from copy import deepcopy

mse = torch.nn.MSELoss()
def save_eval_results_to_excel(args, save_dir, init_result, epoch_results, final_result):
    
    graph_encoder_layer = [args.graph_encoder_layer_input, args.graph_encoder_layer_output]
    encoder_layers = '_'.join(map(str, [input_size] + graph_encoder_layer))
    folder_name = (
        f"GIN_{encoder_layers}_block_size_{args.block_size}_proj_{args.projector_hidden_size}_"
        f"tau_{args.tau}"
    )
    save_dir = save_dir + args.dataset
    save_path = os.path.join(save_dir, folder_name)
    file_path = os.path.join(save_path, 'eval_results.xlsx')

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data = [init_result]          
    data.extend(epoch_results)    
    data.append(final_result)     
    

    df = pd.DataFrame(data, columns=['Mean', 'Std'])

    df.to_excel(file_path, index=False)

def get_dis_data_mask(x, n_neighbors_pos=50, n_neighbors_neg=100): 
    
    x_np = x.detach().cpu().numpy()
    num = torch.from_numpy(x_np).size(0)

    dist_matrix = distance.cdist(x_np, x_np, 'euclidean')
    knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:n_neighbors_pos + 1] 

    pos_row_idx = np.repeat(np.arange(knn_indices.shape[0]), knn_indices.shape[1])
    pos_col_idx = knn_indices.flatten()
    index_pos = torch.tensor([pos_row_idx, pos_col_idx], device=device)

    index_neg_list = []
    for i in range(num):
        all_indices = set(range(num))
        all_indices.discard(i)
        knn_set = set(knn_indices[i])
        candidates = list(all_indices - knn_set)
        sampled_neg = np.random.choice(candidates, size=min(n_neighbors_neg, len(candidates)), replace=False)
        for neg in sampled_neg:
            index_neg_list.append((i, neg))
    
    index_neg = torch.tensor(index_neg_list, dtype=torch.long, device=device).t()

    return index_pos, index_neg

def balanced_bce_loss(pred, pos_num, neg_num):
    pred_pos = pred + torch.log(torch.tensor(pos_num, dtype=torch.float32))
    pred_neg = (1 - pred) + torch.log(torch.tensor(neg_num, dtype=torch.float32))
    power_pos = torch.exp(pred_pos[:pos_num]) / (torch.exp(pred_pos[:pos_num]) + torch.exp(pred_neg[:pos_num]))
    power_neg = torch.exp(pred_neg[pos_num:]) / (torch.exp(pred_pos[pos_num:]) + torch.exp(pred_neg[pos_num:]))
    loss_pos = -torch.log(power_pos).sum()
    loss_neg = -torch.log(power_neg).sum()
    loss = (loss_pos + loss_neg) / (pos_num + neg_num)
    return loss

def train(model: Model, dataloader):

    model.train()

    loss_all = 0
    for data in dataloader:
        data, data_aug = data
        optimizer.zero_grad()

        node_num, _ = data.x.size()
        data = data.to(device)

        if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]

            data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                        not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

        data_aug = data_aug.to(device)

        loss = model.block_loss(data.x, data.edge_index, data.batch, data_aug.x, data_aug.edge_index, data_aug.batch, batch_size=args.batch)
        loss_all += loss * data.num_graphs
        optimizer.step()

    return loss_all

def Init_GCN(model: Model, x, edge_index, y, epochs, device):
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        edge_index_1 = dropout_edge(edge_index, p=args.drop_edge_p1)[0]
        edge_index_2 = dropout_edge(edge_index, p=args.drop_edge_p2)[0]

        x_1 = drop_feature(x, args.drop_feat_p1)
        x_2 = drop_feature(x, args.drop_feat_p2)

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, z2, None, None, None, batch_size=0)
        loss.backward()
        optimizer.step()

def Init_d(model, model_d, x, edge_index, training_epochs, index_pos, index_neg, pos_encoding):

    model = copy.deepcopy(model.encoder).eval()
    model_d.train()
    optimizer_d.zero_grad()

    z = model(x, edge_index)  

    z_pos1 = z[index_pos[0, :], :]  
    z_pos2 = z[index_pos[1, :], :]  
    z_neg1 = z[index_neg[0, :], :]  
    z_neg2 = z[index_neg[1, :], :]  

    pos_feat_source = torch.cat((z_pos1, pos_encoding[index_pos[0, :], :]), dim=-1)
    pos_feat_end = torch.cat((z_pos2, pos_encoding[index_pos[1, :], :]), dim=-1)
    neg_feat_source = torch.cat((z_neg1, pos_encoding[index_neg[0, :], :]), dim=-1)
    neg_feat_end = torch.cat((z_neg2, pos_encoding[index_neg[1, :], :]), dim=-1)

    num_pos = z_pos1.size(0)
    num_neg = z_neg1.size(0)
    lbl_1 = torch.ones(1, num_pos)
    lbl_2 = torch.zeros(1, num_neg)
    lbl = torch.cat((lbl_1, lbl_2), 1).t().to(device)

    pos_logits = model_d(pos_feat_source, pos_feat_end)
    neg_logits = model_d(neg_feat_source, neg_feat_end)
    logits = torch.cat((pos_logits, neg_logits), dim=0)

    b_xent = nn.BCEWithLogitsLoss()

    loss = balanced_bce_loss(logits, num_pos, num_neg) 
    loss.backward()
    optimizer_d.step()

def train_d(model, model_d, enc_feature, epoch, index_pos, index_neg):
    model.eval()
    model_d.train()
    optimizer.zero_grad()
    optimizer_d.zero_grad()
    enc_feature = enc_feature.to(device)
    
    z_pos1 = enc_feature[index_pos[0, :], :]
    z_pos2 = enc_feature[index_pos[1, :], :]
    z_neg1 = enc_feature[index_neg[0, :], :]
    z_neg2 = enc_feature[index_neg[1, :], :]

    pos_feat_source = z_pos1
    pos_feat_end = z_pos2
    neg_feat_source = z_neg1
    neg_feat_end = z_neg2

    pos_logits = model_d(pos_feat_source, pos_feat_end)
    neg_logits = model_d(neg_feat_source, neg_feat_end)

    logits = torch.cat((pos_logits, neg_logits), dim=0)

    num_pos = z_pos1.size(0)
    num_neg = z_neg1.size(0)
    lbl_1 = torch.ones(1, num_pos)
    lbl_2 = torch.zeros(1, num_neg)
    lbl = torch.cat((lbl_1, lbl_2), 1).t().to(device)

    b_xent = nn.BCEWithLogitsLoss()

    loss = balanced_bce_loss(logits, num_pos, num_neg)
    loss.backward()
    optimizer.step()
    optimizer_d.step()

    return loss

def batch_training_with_discriminator(model: Model, model_d,args):
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset, aug=args.aug).shuffle()
    first_column = [data[0] for data in dataset]
    second_column = [data[1] for data in dataset]
    dataloader_view1 = DataLoader(first_column, batch_size = args.batch_size,shuffle=True)
    dataloader_view2 = DataLoader(second_column, batch_size = args.batch_size,shuffle=True)
    
    model.train()
    model_d.eval()
    optimizer.zero_grad()
    z1, enc_y1 = model.encoder.get_embeddings(dataloader_view1)
    z2, enc_y2 = model.encoder.get_embeddings(dataloader_view2)
    total_loss = 0
    z1 = torch.from_numpy(z1)
    z2 = torch.from_numpy(z2)
    z1 = z1.to(device)
    z2 = z2.to(device)
    z1_proj = model.projection(z1)
    z2_proj = model.projection(z2)

    intra_block1 = torch.cat([
        z1_proj
    ], dim=1)
    intra_block2 = torch.cat([
        z2_proj
    ], dim=1)

    inter_logits = model_d(intra_block1, intra_block2)
    intra_logits1 = model_d(intra_block1, intra_block1)
    intra_logits2 = model_d(intra_block2, intra_block2)
    
    batch_loss1 = model.dis_batch_loss(
        z1_proj, z2_proj, inter_logits, intra_logits1
    )  
    batch_loss2 = model.dis_batch_loss(
        z1_proj, z2_proj, inter_logits, intra_logits2
    )
    total_loss += 0.5 * (batch_loss1 + batch_loss2)
    total_loss.backward()
    optimizer.step()
    returnloss = total_loss.item()
    return returnloss

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':

    seed = 2025
    set_random_seeds(seed)
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_seed', type=int, default=seed)
    parser.add_argument('--num_eval_splits', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=2)


    parser.add_argument('--dataset', type=str, default='MUTAG')


    parser.add_argument('--dataset_dir', type=str, default='./dataset/', help='Where the dataset resides.')
    parser.add_argument('--result_folder', type=str, default='result/' , help='Where the dataset resides.')
    # Training hyperparameters.
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_epoch_glo', type=int, default=5)
    parser.add_argument('--num_train_gcn', type=int, default=15)
    parser.add_argument('--num_epochs_d', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2.e-3, help='The learning rate for model training.')
    parser.add_argument('--Discriminator_lr', type=float, default=0.008, help='The learning rate for model training.')
    parser.add_argument('--weight_decay', type=float, default=0.e-4, help='The value of the weight decay for training.')
    parser.add_argument('--mm', type=float, default=0.99, help='The momentum for moving average.')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0, help='Warmup period for learning rate.')
    parser.add_argument('--block_size', type=int, default=256, help='')  
    parser.add_argument('--dis_block_size', type=int, default=32, help='block size for discriminator') 
    parser.add_argument('--pos_enc_dim', type=int, default=64, help='Dimension of position encoding.')
    parser.add_argument('--pos_block_size', type=int, default=16, help='Dimension of position encoding.')


    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--batch', type=int, default=493, help='')
    parser.add_argument('--aug', type=str, default='dnodes')

    # Augmentation
    parser.add_argument('--drop_edge_p1', type=float, default=0.2, help='Probability of edge dropout 1.')
    parser.add_argument('--drop_feat_p1', type=float, default=0.4, help='Probability of node feature dropout 1.')
    parser.add_argument('--drop_edge_p2', type=float, default=0.3, help='Probability of edge dropout 2.')
    parser.add_argument('--drop_feat_p2', type=float, default=0.4, help='Probability of node feature dropout 2.')


    # Architecture.
    parser.add_argument('--graph_encoder_layer_input', type=int, default=128, help='Conv layer sizes.')
    parser.add_argument('--graph_encoder_layer_output', type=int, default=128, help='Conv layer sizes.')
    parser.add_argument('--graph_encoder_layer', type=list, default=[128, 128], help='Conv layer sizes.')
    parser.add_argument('--projector_hidden_size', type=int, default=128, help='Hidden size of projector.')
    parser.add_argument('--D_hidden_dim', type=int, default=128, help='Hidden size of Discriminator.')
    parser.add_argument('--GCNtrain_batch_size', type=int, default=493, help='')
    parser.add_argument('--batch_size', type=int, default=188, help='') 
    parser.add_argument('--layer_num', type=int, default=2, help='')  
    parser.add_argument('--tau', type=float, default=0.05, help='Hidden size of projector.')  

    # Evaluation
    parser.add_argument('--eval_epochs', type=int, default=200, help='Evaluate every eval_epochs.')
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))
    dataset = TUDataset(root=args.dataset_dir, name=args.dataset, aug=args.aug).shuffle()
    dataset_eval = TUDataset(root=args.dataset_dir, name=args.dataset, aug='none').shuffle()
    graph_num = len(dataset)
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1
    dataloader = DataLoader(dataset, batch_size = args.batch_size)
    
    column_data = [data[0] for data in dataset_eval]
    dataloader_eval = DataLoader(column_data, batch_size = args.batch_size)
    
    # build networks
    graph_encoder_layer = [args.graph_encoder_layer_input, args.graph_encoder_layer_output]
    input_size, representation_size = dataset_num_features, graph_encoder_layer[-1]
    encoder = GIN(input_size, representation_size, args.layer_num, device=device, batchnorm=True)
    model = Model(encoder, representation_size * args.layer_num, args.projector_hidden_size, graph_num, tau=args.tau,
                  block_size=args.block_size).to(device)
    predictor = MLP_Predictor(representation_size * args.layer_num, representation_size * args.layer_num,
                              hidden_size=args.projector_hidden_size)
    optimizer = Adam(model.trainable_parameters() + list(predictor.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    start = t()
    prev = start

    for epoch in tqdm(range(1, args.epochs + 1)):
        loss = train(model, dataloader)
    now = t()
    print(now - start, 's')
    model.eval()
    
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    _, _, acc, acc_std = evaluate_embedding(emb, y)
    
    GIN_init_result = (acc, acc_std)
    epoch_result = []
    model_d = Discriminator(representation_size * args.layer_num * 2, args.D_hidden_dim).to(device)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=args.Discriminator_lr, weight_decay=1.e-5)  # 0.05,0.008

    num_epoch_glo = args.num_epoch_glo  
    num_train_gcn = args.num_train_gcn  
    num_epochs_d = args.num_epochs_d 
    start = t()
    prev = start
    for epoch_glo in range(1, num_epoch_glo + 1):

        for epoch in range(1, num_train_gcn + 1):
            encoder_loss = batch_training_with_discriminator(model, model_d, args)

            enc_feature, enc_y = model.encoder.get_embeddings(dataloader_eval)
            enc_feature = torch.from_numpy(enc_feature)
            enc_y = torch.from_numpy(enc_y)
            index_pos, index_neg = get_dis_data_mask(enc_feature)
            for epoch in range(1, num_epochs_d + 1):
                Discriminator_loss = train_d(model, model_d, enc_feature, num_epochs_d, index_pos,
                                             index_neg)
        _, _, acc, acc_std = evaluate_embedding(enc_feature, enc_y)
        epoch_result.append((acc, acc_std))
    time_used = t() - prev
    print(f"time used: {time_used:.2f}s")
    final_result = (acc, acc_std)
    result_folder = args.result_folder
    save_eval_results_to_excel(args, result_folder, GIN_init_result, epoch_result, final_result)