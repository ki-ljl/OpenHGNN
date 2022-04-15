"""Dataset adapters for re-purposing a dataset for a different kind of training task."""

import os
import json
import numpy as np
from dgl.convert import graph as create_dgl_graph
from dgl.sampling.negative import _calc_redundancy
from dgl.data import utils, DGLDataset
from dgl.data.adapter import negative_sample
from dgl import backend as F

__all__ = ['AsNodeClassificationDataset']


class AsNodeClassificationDataset(DGLDataset):
    """Repurpose a dataset for a standard semi-supervised transductive
    node prediction task.

    The class converts a given dataset into a new dataset object that:

      - Contains only one graph, accessible from ``dataset[0]``.
      - The graph stores:

        - Node labels in ``g.ndata['label']``.
        - Train/val/test masks in ``g.ndata['train_mask']``, ``g.ndata['val_mask']``,
          and ``g.ndata['test_mask']`` respectively.
      - In addition, the dataset contains the following attributes:

        - ``num_classes``, the number of classes to predict.
        - ``train_idx``, ``val_idx``, ``test_idx``, train/val/test indexes.

    If the input dataset contains heterogeneous graphs, users need to specify the
    ``target_ntype`` argument to indicate which node type to make predictions for.
    In this case:

      - Node labels are stored in ``g.nodes[target_ntype].data['label']``.
      - Training masks are stored in ``g.nodes[target_ntype].data['train_mask']``.
        So do validation and test masks.

    The class will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given spplit ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    target_ntype : str, optional
        The node type to add split mask for.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.

    Examples
    --------
    >>> ds = dgl.data.AmazonCoBuyComputerDataset()
    >>> print(ds)
    Dataset("amazon_co_buy_computer", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsNodePredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("amazon_co_buy_computer-as-nodepred", num_graphs=1, save_path=...)
    >>> print('train_mask' in new_ds[0].ndata)
    True
    """

    def __init__(self,
                 dataset,
                 split_ratio=None,
                 target_ntype=None,
                 **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        super().__init__(self.dataset.name + '-as-nodepred',
                         hash_key=(split_ratio, target_ntype, dataset.name, 'nodepred'), **kwargs)

    def process(self):
        is_ogb = hasattr(self.dataset, 'get_idx_split')
        if is_ogb:
            g, label = self.dataset[0]
            self.g = g.clone()
            self.g.ndata['label'] = F.reshape(label, (g.num_nodes(),))
        else:
            self.g = self.dataset[0].clone()

        if 'label' not in self.g.nodes[self.target_ntype].data:
            raise ValueError("Missing node labels. Make sure labels are stored "
                             "under name 'label'.")

        if self.split_ratio is None:
            if is_ogb:
                split = self.dataset.get_idx_split()
                train_idx, val_idx, test_idx = split['train'], split['valid'], split['test']
                n = self.g.num_nodes()
                train_mask = utils.generate_mask_tensor(utils.idx2mask(train_idx, n))
                val_mask = utils.generate_mask_tensor(utils.idx2mask(val_idx, n))
                test_mask = utils.generate_mask_tensor(utils.idx2mask(test_idx, n))
                self.g.ndata['train_mask'] = train_mask
                self.g.ndata['val_mask'] = val_mask
                self.g.ndata['test_mask'] = test_mask
            else:
                assert "train_mask" in self.g.nodes[self.target_ntype].data, \
                    "train_mask is not provided, please specify split_ratio to generate the masks"
                assert "val_mask" in self.g.nodes[self.target_ntype].data, \
                    "val_mask is not provided, please specify split_ratio to generate the masks"
                assert "test_mask" in self.g.nodes[self.target_ntype].data, \
                    "test_mask is not provided, please specify split_ratio to generate the masks"
        else:
            if self.verbose:
                print('Generating train/val/test masks...')
            utils.add_nodepred_split(self, self.split_ratio, self.target_ntype)

        self._set_split_index()

        self.multi_label = getattr(self.dataset, 'multi_label', None)
        if self.multi_label is None:
            self.multi_label = len(self.g.nodes[self.target_ntype].data['label'].shape) == 2

        self.num_classes = getattr(self.dataset, 'num_classes', None)
        if self.num_classes is None:
            if self.multi_label:
                self.num_classes = self.g.nodes[self.target_ntype].data['label'].shape[1]
            else:
                self.num_classes = len(F.unique(self.g.nodes[self.target_ntype].data['label']))

        self.meta_paths = getattr(self.dataset, 'meta_paths', None)
        self.meta_paths_dict = getattr(self.dataset, 'meta_paths_dict', None)

    def has_cache(self):
        return os.path.isfile(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))

    def load(self):
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'r') as f:
            info = json.load(f)
            if (info['split_ratio'] != self.split_ratio
                    or info['target_ntype'] != self.target_ntype):
                raise ValueError('Provided split ratio is different from the cached file. '
                                 'Re-process the dataset.')
            self.split_ratio = info['split_ratio']
            self.target_ntype = info['target_ntype']
            self.num_classes = info['num_classes']
        gs, _ = utils.load_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)))
        self.g = gs[0]
        self._set_split_index()

    def save(self):
        utils.save_graphs(os.path.join(self.save_path, 'graph_{}.bin'.format(self.hash)), [self.g])
        with open(os.path.join(self.save_path, 'info_{}.json'.format(self.hash)), 'w') as f:
            json.dump({
                'split_ratio': self.split_ratio,
                'target_ntype': self.target_ntype,
                'num_classes': self.num_classes}, f)

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1

    def _set_split_index(self):
        """Add train_idx/val_idx/test_idx as dataset attributes according to corresponding mask."""
        ndata = self.g.nodes[self.target_ntype].data
        self.train_idx = F.nonzero_1d(ndata['train_mask'])
        self.val_idx = F.nonzero_1d(ndata['val_mask'])
        self.test_idx = F.nonzero_1d(ndata['test_mask'])

    def get_idx(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.g.nodes[self.target_ntype].data['label']

    @property
    def category(self):
        return self.target_ntype
