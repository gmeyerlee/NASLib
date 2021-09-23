from collections.abc import Sequence

NEW_OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero', 
              'linear-skip', 'conv5-skip', 'conv5d2-skip', 'conv7-skip', 'conv7d2-skip', 'zero-skip']
BRANCH_NAMES = ['zero', 'identity']

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
IS_BRANCH = (False, True, True, False, True, False)

COMPACT_TEMPLATE = ((1,0), (1,0,0), (1,0,0,0))


# utils to work with nested collections
def recursive_iter(seq):
    ''' Iterate over elements in seq recursively (returns only non-sequences)
    '''
    if isinstance(seq, Sequence):
        for e in seq:
            for v in recursive_iter(e):
                yield v
    else:
        yield seq


def flatten(seq):
    ''' Flatten all nested sequences, returned type is type of ``seq``
    '''
    return list(recursive_iter(seq))


def copy_structure(data, shape):
    ''' Put data from ``data`` into nested containers like in ``shape``.
        This can be seen as "unflatten" operation, i.e.:
            seq == copy_structure(flatten(seq), seq)
    '''
    d_it = recursive_iter(data)

    def copy_level(s):
        if isinstance(s, Sequence):
            return type(s)(copy_level(ss) for ss in s)
        else:
            return next(d_it)
    return copy_level(shape)


def convert_naslib_to_op_indices(naslib_object):
    # TODO check if this is the cell object
    import ipdb; ipdb.set_trace()
    cell = naslib_object._get_child_graphs(single_nstances=True)[0]
    op_indices = []
    for ind, (i, j) in enumerate(EDGE_LIST):
        op_name = cell.edges[i, j]["op"].get_op_name()
        op_indices.append(NEW_OP_NAMES.index(op_name) if not IS_BRANCH[ind] else BRANCH_NAMES.index(op_name))
    return op_indices

def convert_op_indices_to_naslib(op_indices, naslib_object):
    edge_op_dict = {}
    for i, index in enumerate(op_indices):
        edge_op_dict[EDGE_LIST[i]] = NEW_OP_NAMES[index] if not IS_BRANCH[i] else BRANCH_NAMES[index]

    def add_op_index(edge):
        if (edge.head, edge.tail) in edge_op_dict:
            for i, op in enumerate(edge.data.op):
                # print(op.get_op_name())
                if op.get_op_name() == edge_op_dict[(edge.head, edge.tail)]:
                    index = i
                    break
            
            edge.data.set('op_index', index, shared=True)

    def update_ops(edge):
        if isinstance(edge.data.op, list):
            primitives = edge.data.op
        else:
            primitives = edge.data.primitives
        edge.data.set('op', primitives[edge.data.op_index])
        edge.data.set('primitives', primitives)
    
    naslib_object.update_edges(
        add_op_index, scope=naslib_object.OPTIMIZER_SCOPE, private_edge_data=False
    )

    naslib_object.update_edges(
        update_ops, scope=naslib_object.OPTIMIZER_SCOPE, private_edge_data=False
    )

