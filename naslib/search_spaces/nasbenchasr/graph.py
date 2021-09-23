import os
import pickle
import logging
import numpy as np
import copy
import random
import torch
import torch.nn as nn
import itertools

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.utils import get_project_root
from naslib.search_spaces.nasbenchasr.conversions import flatten, copy_structure

from .conversions import (
    convert_naslib_to_op_indices,
    convert_op_indices_to_naslib
)
from .primitives import CellLayerNorm, PadConvReluNorm, _ops, _branch_ops, get_loss, Head
import torch_edit_distance as ed
from ctcdecode import CTCBeamDecoder


OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']
logger = logging.getLogger(__name__)


def set_succesive_edge_op(edge, filters):
    try:
        s = edge.data
    except AttributeError as e:
        s = edge
    
    s.set(
        "op",
        [
            _ops[oname](filters, filters, name=oname, skip_connection=False) for oname in OP_NAMES
        ] + [
            _ops[oname](filters, filters, name=oname+'-skip', skip_connection=True) for oname in OP_NAMES
        ],
    )

def set_non_succesive_edge_op(edge):
    try:
        s = edge.data
    except AttributeError as e:
        s = edge
    s.set(
        "op",
        [_branch_ops[ctor]() for ctor in range(2)]
    )

class ASRConfigs:
    # TODO move to a better place
    use_rnn = False
    use_norm = True
    dropout_rate = 0.2
    num_blocks = 4
    num_classes = 48

    features = 80
    filters = [600, 800, 1000, 1200]
    cnn_time_reduction_kernels = [8, 8, 8, 8]
    cnn_time_reduction_strides = [1, 1, 2, 2] 
    scells_per_block = [3, 4, 5, 6]


class NasBenchASRSearchSpace(Graph):
    """
    Contains the interface to the tabular benchmark of nas-bench-asr.
    Note: currently we do not support building a naslib object for
    nas-bench-asr architectures.
    
    """

    QUERYABLE = True


    OPTIMIZER_SCOPE = [
        f"stage_{i}" for i in range(1, 5)
    ]


    def __init__(self):
        super().__init__()
        self.load_labeled = False
        self.max_epoch = 40
        self.max_nodes = 3
        self.accs = None
        self.compact = None
        self.asr_configs = config = ASRConfigs()
        self._loss = get_loss()

        # Template Cell definition (same as NASBench201)
        #   See Fig 1 in NAS-Bench-ASR paper for more details.
        # 
        # Input node
        cell = Graph()
        cell.name = 'cell'
        cell.add_node(1)
        # Intermediate nodes
        cell.add_node(2)
        cell.add_node(3)
        # Output node
        cell.add_node(4)
        # Edges
        cell.add_edges_densly()

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Macro architecture definition
        
        total_num_edges = 2 + config.num_blocks + sum(config.scells_per_block) * (2 if config.use_norm else 1)
        self.add_nodes_from(range(1, total_num_edges + 1))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_edges)])

        # Create the operations accordingly 

        curr_node_idx = 1
        for i in range(config.num_blocks):
            logger.debug(f'Create {curr_node_idx},{curr_node_idx+1}')
            self.edges[curr_node_idx, curr_node_idx+1].set(
                "op", 
                PadConvReluNorm(
                    in_channels=config.features if i==0 else config.filters[i-1], 
                    out_channels=config.filters[i], 
                    kernel_size=config.cnn_time_reduction_kernels[i], 
                    dilation=1,
                    strides=config.cnn_time_reduction_strides[i],
                    groups=1,
                    name=f'conv_norm{i}'
                ))
            
            curr_node_idx += 1

            for j in range(config.scells_per_block[i]):
                c_cell = cell.copy().set_scope(f'stage_{i+1}')
                # assign searchable operation to _cell first
                """

                For the successive nodes, we have 2 separate edge
                one for normal operation
                another for skip connection from previous inputs
                
                Current solution:
                For succesive nodes edge, 0-5 represents the skip, 6-12 represents disable skip
                """                
                features = config.filters[i]
                logger.debug(f'Block-{i}#{j}: feature {features}')
                for k in range(1, 4):
                    set_succesive_edge_op(c_cell.edges[k, k+1], features)
                    # logger.debug(f'set successive edge {k, k+1} to {c_cell.edges[k, k+1]}')
                    for l in range(k+2, 5):
                        set_non_succesive_edge_op(c_cell.edges[k, l])
                        # logger.debug(f'set non successive edge {k, l} to {c_cell.edges[k, l]}')
                
                self.edges[curr_node_idx, curr_node_idx+1].set("op", c_cell)
                curr_node_idx += 1
                if config.use_norm:
                    self.edges[curr_node_idx, curr_node_idx+1].set("op", CellLayerNorm(features))
                    curr_node_idx += 1
            # End adding search cells
        # final layer
        self.edges[curr_node_idx, curr_node_idx + 1].set("op", Head(config))
        curr_node_idx += 1
        
        # import ipdb; ipdb.set_trace()
        assert curr_node_idx == total_num_edges, f'Current Node Index {curr_node_idx} should match Total Nodes {total_num_edges}'
        # logger.info(f'current node {curr_node_idx} v.s. total nodes {total_num_nodes}')
        logger.info('Finish construction of NASBench-ASR Graph')
    
    def forward(self, x, edge_data=None):
        return super().forward(x, edge_data)

    def loss(self, output, output_len, targets, targets_len):
        return self._loss(output, output_len, targets, targets_len)

    def query(self, metric=None, dataset=None, path=None, epoch=-1,
              full_lc=False, dataset_api=None):
        """
        Query results from nas-bench-asr
        """
        metric_to_asr = {
            Metric.VAL_ACCURACY: "val_per",
            Metric.TEST_ACCURACY: "test_per",
            Metric.PARAMETERS: "params",
            Metric.FLOPS: "flops",
        }

        assert self.compact is not None
        assert metric in [
            Metric.TRAIN_ACCURACY,
            Metric.TRAIN_LOSS,
            Metric.VAL_ACCURACY,
            Metric.TEST_ACCURACY,
            Metric.PARAMETERS,
            Metric.FLOPS,
            Metric.TRAIN_TIME,
            Metric.RAW,
        ]
        query_results = dataset_api["asr_data"].full_info(self.compact)


        if metric != Metric.VAL_ACCURACY:
            if metric == Metric.TEST_ACCURACY:
                return query_results[metric_to_asr[metric]]
            elif (metric == Metric.PARAMETERS) or (metric == Metric.FLOPS):
                return query_results['info'][metric_to_asr[metric]]
            elif metric in [Metric.TRAIN_ACCURACY, Metric.TRAIN_LOSS,
                            Metric.TRAIN_TIME, Metric.RAW]:
                return -1
        else:
            if full_lc and epoch == -1:
                return [
                    loss for loss in query_results[metric_to_asr[metric]]
                ]
            elif full_lc and epoch != -1:
                return [
                    loss for loss in query_results[metric_to_asr[metric]][:epoch]
                ]
            else:
                # return the value of the metric only at the specified epoch
                return float(query_results[metric_to_asr[metric]][epoch])

    def get_compact(self):
        assert self.compact is not None
        return self.compact

    def get_hash(self):
        return self.get_compact()

    def set_compact(self, compact):
        self.compact = compact

    def sample_random_architecture(self, dataset_api=None):
        search_space = [[len(OP_NAMES)] + [2]*(idx+1) for idx in
                        range(self.max_nodes)]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)

        compact = m
        self.set_compact(compact)
        
        op_indices = self.compact_to_op_indices(compact)
        self.set_op_indices(op_indices)
        return compact
    
    # Translate the architecture [[1,0], [1,0,0], [1,0,0,0]] to [0-12, 0-1, 0-1, 0-12, 0-1, 0-12]
    @staticmethod
    def compact_to_op_indices(compact):
        if len(compact) == 3:
            compact = list(itertools.chain.from_iterable(compact))
        COMPACT_TO_OP_IDX = [
            # Compact index / Edge
            0, # 1,2
            3, # 1,3
            6, # 1,4
            2, # 2,3
            7, # 2,4
            5, # 3, 4
        ]
        IS_BRANCH_IDX = {
            # Add 6 if 0
            1: 0, # 1,2 skip
            4: 3, # 2,3 skip
            8: 5, # 3,4 skip
        }
        # import ipdb; ipdb.set_trace()
        op_indices = [compact[c] for c in COMPACT_TO_OP_IDX]
        for cid, oid in IS_BRANCH_IDX.items():
            op_indices[oid] += compact[cid] * 6
        logger.debug(f'compact {compact} to op_indices {op_indices}')
        return op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def get_op_indices(self):
        # assert self.compact is not None
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices
        # return self.compact_to_op_indices(self.compact)
    
    def set_op_indices(self, op_indices):
        if op_indices is None:
            raise ValueError('Op-indices cannot be None!')
        self.op_indices = op_indices
        convert_op_indices_to_naslib(op_indices, self)
    
    def set_spec(self, op_indices, dataset_api=None):
        return self.set_op_indices(op_indices)

    def mutate(self, parent, mutation_rate=1, dataset_api=None):
        """
        This will mutate the cell in one of two ways:
        change an edge; change an op.
        Todo: mutate by adding/removing nodes.
        Todo: mutate the list of hidden nodes.
        Todo: edges between initial hidden nodes are not mutated.
        """
        parent_compact = parent.get_compact()
        compact = copy.deepcopy(parent_compact)

        for _ in range(int(mutation_rate)):
            mutation_type = np.random.choice([2])

            if mutation_type == 1:
                # change an edge
                # first pick up a node
                node_id = np.random.choice(3)
                node = compact[node_id]
                # pick up an edge id
                edge_id = np.random.choice(len(node[1:])) + 1
                # edge ops are in [identity, zero] ([0, 1])
                new_edge_op = int(not compact[node_id][edge_id])
                # apply the mutation
                compact[node_id][edge_id] = new_edge_op

            elif mutation_type == 2:
                # change an op
                node_id = np.random.choice(3)
                node = compact[node_id]
                op_id = node[0]
                list_of_ops_ids = list(range(len(OP_NAMES)))
                list_of_ops_ids.remove(op_id)
                new_op_id = random.choice(list_of_ops_ids)
                compact[node_id][0] = new_op_id

        self.set_compact(compact)


    def get_nbhd(self, dataset_api=None):
        """
        Return all neighbors of the architecture
        """
        compact = self.get_compact()
        #edges, ops, hiddens = compact
        nbhd = []

        def add_to_nbhd(new_compact, nbhd):
            print(new_compact)
            nbr = NasBenchASRSearchSpace()
            nbr.set_compact(new_compact)
            nbr_model = torch.nn.Module()
            nbr_model.arch = nbr
            nbhd.append(nbr_model)
            return nbhd

        for node_id in range(len(compact)):
            node = compact[node_id]
            for edge_id in range(len(node)):
                if edge_id == 0:
                    edge_op = compact[node_id][0]
                    list_of_ops_ids = list(range(len(OP_NAMES)))
                    list_of_ops_ids.remove(edge_op)
                    for op_id in list_of_ops_ids:
                        new_compact = copy.deepcopy(compact)
                        new_compact[node_id][0] = op_id
                        nbhd = add_to_nbhd(new_compact, nbhd)
                else:
                    edge_op = compact[node_id][edge_id]
                    new_edge_op = int(not edge_op)
                    new_compact = copy.deepcopy(compact)
                    new_compact[node_id][edge_id] = new_edge_op
                    nbhd = add_to_nbhd(new_compact, nbhd)

        random.shuffle(nbhd)
        return nbhd

    def get_type(self):
        return 'nasbenchasr'

    def get_max_epochs(self):
        return 39

