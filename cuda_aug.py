import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=None):
        self.name = name
        self.cleaned = cleaned
        super(TUDataset_aug, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

        self.aug = aug

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        # 自动加 self-loop
        node_num = data.edge_index.max().item() + 1
        sl = torch.arange(node_num, device=data.edge_index.device)
        self_loops = torch.stack([sl, sl], dim=0)
        data.edge_index = torch.cat((data.edge_index, self_loops), dim=1)

        return data

    '''
    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif self.aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif self.aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif self.aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False


        elif self.aug == 'random3':
            print(3333333333)
            n = np.random.randint(3)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = permute_edges(deepcopy(data))
            elif n == 2:
               data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
            print(444444444)

        elif self.aug == 'random4':
            n = np.random.randint(4)
            if n == 0:
               data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
               data_aug = permute_edges(deepcopy(data))
            elif n == 2:
               data_aug = subgraph(deepcopy(data))
            elif n == 3:
               data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False

        else:
            print('augmentation error')
            assert False

        # print(data, data_aug)
        # assert False

        return data, data_aug
        '''

def drop_nodes(data):
    device = data.x.device
    node_num = data.x.size(0)
    drop_num = int(node_num / 10)

    idx_drop = torch.randperm(node_num, device=device)[:drop_num]
    idx_mask = torch.ones(node_num, dtype=torch.bool, device=device)
    idx_mask[idx_drop] = False

    edge_index = data.edge_index
    mask = idx_mask[edge_index[0]] & idx_mask[edge_index[1]]
    data.edge_index = edge_index[:, mask]

    return data



def permute_edges(data):
    device = data.x.device
    node_num = data.x.size(0)
    edge_index = data.edge_index
    edge_num = edge_index.size(1)
    permute_num = int(edge_num / 10)

    keep_indices = torch.randperm(edge_num, device=device)[permute_num:]
    new_edges = torch.randint(0, node_num, (2, permute_num), device=device)

    data.edge_index = torch.cat([edge_index[:, keep_indices], new_edges], dim=1)
    return data


def subgraph(data):
    device = data.x.device
    node_num = data.x.size(0)
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index
    idx_sub = [torch.randint(0, node_num, (1,), device=device).item()]
    visited = set(idx_sub)

    for _ in range(sub_num - 1):
        neighbors = edge_index[1][edge_index[0] == idx_sub[-1]]
        neighbors = [n.item() for n in neighbors if n.item() not in visited]
        if not neighbors:
            break
        next_node = torch.tensor(neighbors, device=device)[torch.randint(0, len(neighbors), (1,))].item()
        idx_sub.append(next_node)
        visited.add(next_node)

    idx_sub = torch.tensor(idx_sub, dtype=torch.long, device=device)
    idx_mask = torch.zeros(node_num, dtype=torch.bool, device=device)
    idx_mask[idx_sub] = True

    mask = idx_mask[edge_index[0]] & idx_mask[edge_index[1]]
    data.edge_index = edge_index[:, mask]
    return data


def mask_nodes(data):
    device = data.x.device
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = torch.randperm(node_num, device=device)[:mask_num]
    noise = torch.randn((mask_num, feat_dim), device=device) * 0.5 + 0.5
    data.x[idx_mask] = noise

    return data


class GraphAugmentationTransform:
    def __init__(self, aug_type='none'):
        self.aug_type = aug_type

    def __call__(self, data):
        if self.aug_type == 'dnodes':
            return drop_nodes(data)
        elif self.aug_type == 'pedges':
            return permute_edges(data)
        elif self.aug_type == 'subgraph':
            return subgraph(data)
        elif self.aug_type == 'mask_nodes':
            return mask_nodes(data)
        elif self.aug_type == 'none':
            return data
        elif self.aug_type.startswith('random'):
            import random
            options = {
                'random2': ['dnodes', 'subgraph'],
                'random3': ['dnodes', 'pedges', 'subgraph'],
                'random4': ['dnodes', 'pedges', 'subgraph', 'mask_nodes']
            }.get(self.aug_type, [])
            return GraphAugmentationTransform(random.choice(options))(data)
        else:
            raise ValueError(f"Unknown augmentation type: {self.aug_type}")

