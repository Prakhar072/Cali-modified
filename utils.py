import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import pickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch.sparse


from typing import Optional, Tuple
from torch import Tensor


def symmtrical_Laplacian_matrix(normalized_adj, n, pos_enc_dim):
    sym_L = sp.eye(n) - normalized_adj
    EigVal, EigVec = sp.linalg.eigs(sym_L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    increasing_eig_values = EigVal[EigVal.argsort()]
    increasing_eig_values = torch.from_numpy(increasing_eig_values[1:pos_enc_dim+1]).float()
    return lap_pos_enc, increasing_eig_values

def symmetric_laplacian_matrix(edge_index, num_nodes, pos_enc_dim, device='cuda'):
    edge_weight = torch.ones(edge_index.size(1), device=device)
    adj = torch.sparse_coo_tensor(
        indices=edge_index,
        values=edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    )
    

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt).to_sparse()
    norm_adj = torch.sparse.mm(
        torch.sparse.mm(deg_inv_sqrt_mat, adj),
        deg_inv_sqrt_mat
    )
    

    identity = torch.eye(num_nodes, device=device)
    sym_L = identity - norm_adj.to_dense()
    

    eig_values, eig_vecs = torch.linalg.eigh(sym_L)
    

    lap_pos_enc = eig_vecs[:, 1:pos_enc_dim+1]
    eig_values = eig_values[1:pos_enc_dim+1]
    
    return lap_pos_enc, eig_values

def adj_transform(x, edge_index):

    n_nodes = x.shape[0]
    adj_index = edge_index.detach().cpu()
    value = np.ones(len(adj_index[0]))

    sp_adj = sp.coo_matrix((value, (adj_index[0], adj_index[1])), shape=(n_nodes, n_nodes)).tocsr()
    nodes_to_keep = torch.LongTensor(np.arange(x.shape[0]))
    adj_matrix = sp_adj[nodes_to_keep][:, nodes_to_keep]

    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
    adj_matrix = preprocess_adj(adj_matrix)


    return adj_matrix


def augmented_adj_transform(x, edge_index):

    n_nodes = x.shape[0]
    adj_index = edge_index.detach().cpu()
    value = np.ones(len(adj_index[0]))

    sp_adj = sp.coo_matrix((value, (adj_index[0], adj_index[1])), shape=(n_nodes, n_nodes)).tocsr()
    nodes_to_keep = torch.LongTensor(np.arange(x.shape[0]))
    adj_matrix = sp_adj[nodes_to_keep][:, nodes_to_keep]

    adj_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
    adj_matrix = preprocess_adj(adj_matrix)

    return adj_matrix


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_adj_like_sparse_weight(data, pos_weight):

    num_nodes = 2708

    indices = data.edge_index.detach().cpu()
    values = pos_weight.squeeze(dim=0)

    return torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))


def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, self_loop=True):
    if self_loop is True:
        adj_normalized = normalized_adj(adj + sp.eye(adj.shape[0]))
    else:
        adj_normalized = normalized_adj(adj)
    return adj_normalized


def batch_block_sim(z1: torch.Tensor, z2: torch.Tensor, block_size=8, batch_size=256, norm=False, distance='cosine'):
    fea_dim = z1.size(1)
    if norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    block_num = (fea_dim + block_size - 1) // block_size
    result = torch.zeros(z1.size(0), z2.size(0), block_num, device=z1.device)
    for i in range(0, z1.size(0), batch_size):
        z1_batch = z1[i:i+batch_size]
        batch_result = torch.zeros(z1_batch.size(0), z2.size(0), block_num, device=z1.device)
        for j in range(0, z2.size(0), batch_size):
            z2_batch = z2[j:j+batch_size]
            
            for k in range(block_num):
                start = k * block_size
                end = min((k + 1) * block_size, fea_dim)
                
                z1_block = z1_batch[:, start:end]
                z2_block = z2_batch[:, start:end]
                
                if distance == 'L2':
                    sim = dis_fun(z1_block, z2_block)
                else:
                    sim = torch.mm(z1_block, z2_block.t())
                
                batch_result[:z1_batch.size(0), j:j+batch_size, k] = sim
        
        result[i:i+batch_size] = batch_result
    
    return result


def block_sim(z1: torch.Tensor, z2: torch.Tensor, block_size=8, norm=False, distance='cosine', batch_size = 0):# distance='L2'
    if batch_size == 0:
        fea_dim = z1.size(1)
        if norm is True:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)

        if fea_dim % block_size == 0:
            block_num = fea_dim // block_size
        else:
            block_num = fea_dim // block_size + 1
        block_sim_list = []

        for i in range(block_num):
            if i == block_num - 1:
                start = i * block_size
                end = fea_dim
            else:
                start = i * block_size
                end = (i + 1) * block_size
            z1_block = z1[:, start: end]
            z2_block = z2[:, start: end]
            if distance == 'L2':
                block_sim_list.append(dis_fun(z1_block, z2_block).unsqueeze(dim=2))
            else:
                block_sim_list.append(torch.mm(z1_block, z2_block.t()).unsqueeze(dim=2))
        block_smi = torch.cat(block_sim_list, dim=-1)
    else:
        block_smi = batch_block_sim(z1, z2, block_size=8, batch_size = batch_size , norm=False, distance='cosine')
    
    return block_smi


def blockwise_similarity(z1: torch.Tensor, z2: torch.Tensor, block_size: int = 256, norm: bool = True, distance: str = 'cosine'
) -> torch.Tensor:
    if distance not in ['cosine', 'L2']:
        raise ValueError(f"Invalid distance type: {distance}, expected 'cosine' or 'L2'")

    if norm:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

    fea_dim = z1.size(1)
    block_num = (fea_dim + block_size - 1) // block_size 

    output = torch.zeros(z1.size(0), z2.size(0), block_num, device=z1.device)

    for block_idx in range(block_num):
        start = block_idx * block_size
        end = min(start + block_size, fea_dim)
        
        z1_block = z1[:, start:end]
        z2_block = z2[:, start:end]

        with torch.no_grad():
            if distance == 'L2':

                diff = z1_block.unsqueeze(1) - z2_block.unsqueeze(0)  # [N1, N2, D']
                block_sim = torch.norm(diff, p=2, dim=-1)  # [N1, N2]
            else:

                block_sim = torch.mm(z1_block, z2_block.t())  # [N1, N2]
            
            output[:, :, block_idx] = block_sim

    return output

def pair_block_sim(z1: torch.Tensor, z2: torch.Tensor, block_size=8, norm=False):
    fea_dim = z1.size(1)
    if norm is True:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

    if fea_dim % block_size == 0:
        block_num = fea_dim // block_size
    else:
        block_num = fea_dim // block_size + 1
    block_sim_list = []

    for i in range(block_num):
        if i == block_num - 1:
            start = i * block_size
            end = fea_dim
        else:
            start = i * block_size
            end = (i + 1) * block_size
        z1_block = z1[:, start: end]
        z2_block = z2[:, start: end]
        pair_sim = (z1_block * z2_block).sum(dim=1).unsqueeze(dim=1)

        block_sim_list.append(pair_sim)
    block_smi = torch.cat(block_sim_list, dim=-1)
    return block_smi


def blockwise_similarity_batch(z1: torch.Tensor, z2: torch.Tensor, block_size: int = 64, batch_size: int = 128, norm: bool = True, distance: str = 'cosine'
) -> torch.Tensor:
    
    if distance not in ['cosine', 'L2']:
        raise ValueError(f"Invalid distance: {distance}")
    if batch_size < 0:
        raise ValueError("Batch size cannot be negative")

    if norm:
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

    D = z1.size(1)
    num_blocks = (D + block_size - 1) // block_size

    output_blocks = []

    for block_idx in range(num_blocks):
        s = block_idx * block_size
        e = min(s + block_size, D)

        block_output = torch.zeros(z1.size(0), z2.size(0), dtype=z1.dtype, device='cpu')
        
        for i in range(0, z1.size(0), batch_size):
            z1_batch = z1[i:i+batch_size, s:e].to('cpu')
            for j in range(0, z2.size(0), batch_size):
                z2_batch = z2[j:j+batch_size, s:e].to('cpu')
                
                if distance == 'L2':
                    diff = z1_batch.unsqueeze(1) - z2_batch.unsqueeze(0)
                    sim = torch.norm(diff, p=2, dim=-1)
                else:
                    sim = torch.matmul(z1_batch, z2_batch.t())
                
                block_output[i:i+batch_size, j:j+batch_size] = sim
        

        output_blocks.append(block_output.to(z1.device).unsqueeze(-1))
        

    return torch.cat(output_blocks, dim=-1)


def compute_block(z1_block, z2_block, batch_size, distance):

    n1, n2 = z1_block.size(0), z2_block.size(0)
    block = torch.zeros(n1, n2, device=z1_block.device)
    
    for i in range(0, n1, batch_size):
        for j in range(0, n2, batch_size):
            batch1 = z1_block[i:i+batch_size]
            batch2 = z2_block[j:j+batch_size]
            
            with torch.no_grad():
                if distance == 'L2':
                    sim = torch.cdist(batch1, batch2)
                else:
                    sim = torch.mm(batch1, batch2.t())
            
            block[i:i+batch_size, j:j+batch_size] = sim
            del batch1, batch2, sim
            torch.cuda.empty_cache()
    
    return block



def dis_fun(x, c):
    xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
    cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
    xx_cc = xx + cc
    xc = x @ c.T
    distance = xx_cc - 2 * xc
    return distance


def cal_pos_encoding(x, edge_index, pos_enc_dim):
    device = x.device
    pos_encoding1, eigen_values = symmetric_laplacian_matrix(edge_index, x.size(0), pos_enc_dim, device)
    eigen_values = eigen_values.to(device)
    pos_encoding = pos_encoding1.to(device) / (eigen_values+1e-8)
    return pos_encoding


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask