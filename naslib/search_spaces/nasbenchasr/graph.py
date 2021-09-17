import os
import pickle
import logging
import numpy as np
import copy
import random
import torch

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.core.graph import Graph
from naslib.utils.utils import get_project_root
from naslib.search_spaces.nasbenchasr.conversions import flatten, copy_structure

from .conversions import (
    convert_naslib_to_op_indices,
    convert_op_indices_to_naslib
)
from .primitives import CellLayerNorm, PadConvReluNorm, _ops, _branch_ops

OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']


def set_succesive_edge_op(edge, filters):
    edge.data.set(
        "op",
        [
            _ops[oname](filters, filters, name=oname, skip_connection=False) for oname in OP_NAMES
        ] + [
            _ops[oname](filters, filters, name=oname+'-skip', skip_connection=True) for oname in OP_NAMES
        ],
    )

def set_non_succesive_edge_op(edge):
    edge.data.set(
        "op",
        [ctor() for ctor in _branch_ops]
    )

class ASRConfigs:
    # TODO move to a better place
    use_rnn = False
    use_norm = True
    dropout_rate = 0.0
    num_blocks = 4
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
        
        total_num_nodes = 2 + config.num_blocks + sum(config.scells_per_block)
        # create linear graph structure for the 
        self.add_nodes_from(range(1, total_num_nodes+1))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

        # Create the operations accordingly 

        curr_node_idx = 0
        for i in range(config.num_blocks):
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

                for i in range(1, 4):

                    set_succesive_edge_op(c_cell.edges[i, i+1], features)
                    logging.debug(f'set successive edge {i, i+1} to {c_cell.edges[i, i+1]}')
                    for j in range(i+2, 5):
                        set_non_succesive_edge_op(c_cell.edges[i, j])
                        logging.debug(f'set non successive edge {i, j} to {c_cell.edges[i, j]}')
                
                # Handle cell wise layer norm here.
                if config.use_norm:
                    c_cell.add_node(5)
                    c_cell.add_edge([4, 5])
                    c_cell.edges[4,5].set("op", CellLayerNorm(features))
                    logging.debug(f'add norm layer to c_cell')
                
                self.edges[curr_node_idx, curr_node_idx+1].set("op", c_cell)
                curr_node_idx += 1

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

    def sample_random_architecture(self, dataset_api):
        search_space = [[len(OP_NAMES)] + [2]*(idx+1) for idx in
                        range(self.max_nodes)]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        m = copy_structure(m, search_space)

        compact = m
        self.set_compact(compact)
        return compact
    
    # Translate the architecture [[1,0], [1,0,0], [1,0,0,0]] to [0-12, 0-1, 0-1, 0-12, 0-1, 0-12]
    @staticmethod
    def compact_to_op_indices(compact):
        import ipdb; ipdb.set_trace()

    def get_op_indices(self):
        assert self.compact is not None
        return self.compact_to_op_indices(self.compact)
    
    def set_op_indices(self, op_indices):
        self.op_indices = op_indices
            

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
        return 'asr'

    def get_max_epochs(self):
        return 39

