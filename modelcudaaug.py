import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict

from torch_geometric.nn import BatchNorm, GCNConv, global_add_pool, GINConv
import torch.nn.init as Init

from torch.nn import Sequential, Linear, ReLU, ELU
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_dim,D_hidden_dim):
        super(Discriminator, self).__init__()

        hidden_dim1 = D_hidden_dim
        self.model = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim1),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Linear(hidden_dim1, 128),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Linear(128, 64),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            
            nn.Linear(input_dim, hidden_dim1),
            nn.LeakyReLU(0.3, inplace=True),
            # nn.ReLU(),
            # nn.ELU(),
            # nn.BatchNorm1d(128),
            nn.Linear(hidden_dim1, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 1),
            # nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        return self.model(x)

    def pred(self, input):
        return self.model(input)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, graph_num, tau: float = 0.5, block_size=32):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.block_size = block_size

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.graph_num = graph_num

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch) -> torch.Tensor:
        return self.encoder(x, edge_index, batch)

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters()) + list(self.fc1.parameters()) + list(self.fc2.parameters())

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def block_sim(self, z1: torch.Tensor, z2: torch.Tensor, random=True):
        fea_dim = z1.size(1)
        if random is True:
            random_order = torch.randperm(fea_dim)
            z1 = F.normalize(z1[:, random_order])
            z2 = F.normalize(z2[:, random_order])
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)

        if fea_dim % self.block_size == 0:
            block_num = fea_dim // self.block_size
        else:
            block_num = fea_dim // self.block_size + 1
        block_sim_list = []

        for i in range(block_num):
            if i == block_num - 1:
                start = i * self.block_size
                end = fea_dim
            else:
                start = i * self.block_size
                end = (i + 1) * self.block_size
            z1_block = z1[:, start: end]
            z2_block = z2[:, start: end]
            block_sim_list.append(torch.mm(z1_block, z2_block.t()))
        return block_sim_list
    def block_sim_intro(self, z1: torch.Tensor, z2: torch.Tensor, block_zise_intro, random=True):
        fea_dim = z1.size(1)
        if random is True:
            random_order = torch.randperm(fea_dim)
            z1 = F.normalize(z1[:, random_order])
            z2 = F.normalize(z2[:, random_order])
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)

        if fea_dim % block_zise_intro == 0:
            block_num = fea_dim // block_zise_intro
        else:
            block_num = fea_dim // block_zise_intro + 1
        block_sim_list = []

        for i in range(block_num):
            if i == block_num - 1:
                start = i * block_zise_intro
                end = fea_dim
            else:
                start = i * block_zise_intro
                end = (i + 1) * block_zise_intro
            z1_block = z1[:, start: end]
            z2_block = z2[:, start: end]
            block_sim_list.append(torch.mm(z1_block, z2_block.t()))

        return block_sim_list

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    
    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)
    
    def semi_block_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_block_sim_list = self.block_sim(z1, z2)
        between_block_sim_list = self.block_sim(z2, z1)
        block_num = len(refl_block_sim_list)
        refl_block_sim = 0
        between_block_sim = 0
        for i in range(block_num):
            refl_block_sim = refl_block_sim + f(refl_block_sim_list[i])
            between_block_sim = between_block_sim + f(between_block_sim_list[i])

        return -torch.log(between_block_sim.diag() / (refl_block_sim.sum(1) + between_block_sim.sum(1) - refl_block_sim.diag()))

    def dis_batch_loss(self, z1: torch.Tensor, z2: torch.Tensor, inter_preds, intra_preds, threshold_pos=0.6, threshold_neg=0.6):
        f = lambda x: torch.exp(x / self.tau)

        num = z1.size(0)
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        refl_block_sim_list = self.block_sim(h1, h1)
        between_block_sim_list = self.block_sim(h2, h1)
        block_num = len(refl_block_sim_list)

        refl_block_sim = 0
        between_block_sim = 0
        for i in range(block_num):
            refl_block_sim = refl_block_sim + f(refl_block_sim_list[i])
            between_block_sim = between_block_sim + f(between_block_sim_list[i])

        inter_sim_matrix_dis = inter_preds.reshape(num, -1)
        intra_sim_matrix_dis = intra_preds.reshape(num, -1)

        inter_pos_flag = inter_sim_matrix_dis > threshold_pos
        intra_pos_flag = intra_sim_matrix_dis > threshold_pos
        inter_neg_flag = inter_sim_matrix_dis < threshold_neg
        intra_neg_flag = intra_sim_matrix_dis < threshold_neg


        refl_sim_pos = refl_block_sim * intra_pos_flag
        between_sim_pos = between_block_sim * inter_pos_flag
        refl_sim_neg = refl_block_sim * intra_neg_flag
        between_sim_neg = between_block_sim * inter_neg_flag


        pos1 = between_sim_pos.sum(1)
        pos2 = refl_sim_pos.sum(1)
        pos = pos1 + pos2
        neg1 = between_sim_neg.sum(1)
        neg2 = refl_sim_neg.sum(1)
        neg = neg1 + neg2

        valid_indices = torch.where(pos != 0)

        loss = -torch.log(pos[valid_indices] / (between_sim_pos.diag()[valid_indices] + neg[valid_indices]))

        return loss.mean()


    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):

        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def block_loss(self, x_1, edge_index_1, batch1, x_2, edge_index_2, batch2, mean: bool = True, batch_size: int = 0):


        if batch_size == 0:
            assert False
            l1 = self.semi_block_loss(h1, h2)
            l2 = self.semi_block_loss(h2, h1)
        else:
            l1 = self.batched_semi_block_loss(x_1, edge_index_1, batch1, x_2, edge_index_2, batch2, batch_size, mean)
            l2 = self.batched_semi_block_loss(x_2, edge_index_2, batch2, x_1, edge_index_1, batch1, batch_size, mean)
        
        ret = (l1 + l2) * 0.5

        return ret

    def batched_semi_block_loss(self, x1, edge_index1, batch1, x2, edge_index2, batch2, batch_size: int, mean: bool):
        device = x1.device
        num_nodes = self.graph_num

        num_batches = (self.graph_num - 1) // batch_size + 1

        total_loss = 0

        indices = torch.arange(self.graph_num, device=device)
        f = lambda x: torch.exp(x / self.tau)

        z1 = self.projection(self.forward(x1, edge_index1, batch1)[0])  # shape [N, D]
        z2 = self.projection(self.forward(x2, edge_index2, batch2)[0])  # shape [N, D]



        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]

            z1_batch = z1[mask]

            refl_block_sim_list = self.block_sim(z1_batch, z1)  # [B, N]
            between_block_sim_list = self.block_sim(z1_batch, z2)  # [B, B]


            refl_sim = sum(f(sim) for sim in refl_block_sim_list)  # [B, N]
            between_sim = sum(f(sim) for sim in between_block_sim_list)  # [B, B]

            diag = between_sim.diag()  # shape [B]

            refl_diag = refl_sim[torch.arange(len(mask), device=device), mask]
            
            numerator = diag
            denominator = refl_sim.sum(1) + between_sim.sum(1) - refl_diag
            loss = -torch.log(numerator / denominator)

            if mean:
                loss = loss.mean()
            else:
                loss = loss.sum()

            loss = loss / num_batches
            loss.backward(retain_graph=True)
            total_loss += loss.item()

        return total_loss



class GCN(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=False, weight_standardization=False):
        super().__init__()

        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization
        self.BN = batchnorm

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):

            layers.append(('x, edge_index -> x: ' + str(i), GCNConv(in_dim, out_dim)))
            if batchnorm:
                layers.append(('with BN: ' + str(i), BatchNorm(out_dim, momentum=batchnorm_mm)))
            else:
                layers.append(('Identity: ' + str(i), nn.Identity()))

            layers.append(('activation: ' + str(i), nn.PReLU()))

        self.model = Sequential(OrderedDict(layers))

    def forward(self, x, edge_index, bn=True):
        if self.weight_standardization:
            self.standardize_weights()
        x = x.detach()
        model_size = len(self.model)
        if self.BN is True:
            model_size /= 3
            for i in range(int(model_size)):
                x = self.model[3 * i](x, edge_index)
                if bn is True:
                    x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)

        else:
            model_size /= 3
            for i in range(int(model_size)):
                x = self.model[3 * i](x, edge_index)
                x = self.model[3 * i + 1](x)
                x = self.model[3 * i + 2](x)
        return x

    def reset_parameters(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob

    x = x.clone()
    x[:, drop_mask] = 0

    return x


class GIN(nn.Module):
    def __init__(self, in_features, out_features, num_gc_layers, device='cpu', batchnorm=True):
        super(GIN, self).__init__()


        self.device = device

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(size=(in_features, out_features)), requires_grad=True)
        self._weight = nn.Parameter(torch.FloatTensor(size=(1, 5)), requires_grad=True)
        Init.uniform_(self._weight, 0, 1)

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                gnn = Sequential(Linear(out_features, out_features), ReLU(), Linear(out_features, out_features))
            else:
                gnn = Sequential(Linear(in_features, out_features), ReLU(), Linear(out_features, out_features))
            conv = GINConv(gnn)
            bn = torch.nn.BatchNorm1d(out_features)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(self.device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.elu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if not hasattr(data, 'batch'):
                    data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)

                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())

        return torch.from_numpy(np.concatenate(ret, 0)).to(self.device), torch.from_numpy(np.concatenate(y, 0)).to(self.device)
    def get_embedding2(self, data):
        ret = []
        y = []
        with torch.no_grad():
            if not hasattr(data, 'batch'):
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

            data.to(self.device)
            x, edge_index, batch = data.x, data.edge_index, data.batch

            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(self.device)

            x, _ = self.forward(x, edge_index, batch)

            ret.append(x.cpu().numpy())
            y.append(data.y.cpu().numpy())

        return torch.from_numpy(np.concatenate(ret, 0)), torch.from_numpy(np.concatenate(y, 0))
    def get_embeddings_v(self, loader):

        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y