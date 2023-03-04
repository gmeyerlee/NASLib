import torch
import logging
import numpy as np

from naslib.optimizers import OneShotNASOptimizer
from naslib.utils import utils
from naslib.search_spaces.darts.conversions import convert_compact_to_genotype, convert_genotype_to_compact, make_compact_mutable

logger = logging.getLogger(__name__)


class EvolvingOptimizer(OneShotNASOptimizer):
    """
    Implementation of the Random NAS with weight sharing as in
        Li et al. 2019: Random Search and Reproducibility for Neural Architecture Search.
    """

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the architectural weights to the edges.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            torch.zeros(size=[len_primitives], requires_grad=False)
        )
        edge.data.set("alpha", alpha, shared=True)

    def __init__(
        self,
        config,
        op_optimizer=torch.optim.SGD,
        arch_optimizer=None,
        loss_criteria=torch.nn.CrossEntropyLoss(),
        innov_protect=False
    ):

        super(EvolvingOptimizer, self).__init__(
            config, op_optimizer, arch_optimizer, loss_criteria
        )
        self.history = []
        self.population = []
        self.innov_protect = innov_protect

    def before_training(self):
        """
        Move the graph into cuda memory if available.
        """
        self.graph = self.graph.to(self.device)
        self.architectural_weights = self.architectural_weights.to(self.device)
        self.initialize_pop()

    def initialize_pop(self, P=30):
        for i in range(P):
            tmp_graph = self.search_space.clone()
            tmp_graph.sample_random_architecture()

            if self.graph.get_type() == "nasbench201":
                sample = tmp_graph.get_op_indices()
            elif self.graph.get_type() == "darts":
                sample = convert_compact_to_genotype(tmp_graph.get_compact())
            self.population.append((sample, 0, 0))
        self.pop_iter = iter(range(len(self.population)))

    def step(self, data_train, data_val):
        input_train, target_train = data_train
        input_val, target_val = data_val

        try:
            arch_id = next(self.pop_iter)
            arch, prev_val, tr_ct = self.population[arch_id]
        except StopIteration:
            self.sample_and_mutate()
            arch_id = next(self.pop_iter)
            arch, prev_val, tr_ct = self.population[arch_id]

        self.set_alphas_from_path(arch)

        # Update op weights
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        logits_val = self.graph(input_val)
        val_loss = self.loss(logits_val, target_val)
        #val_loss.backward()
        val_acc = utils.accuracy(logits_val, target_val)[0]

        self.population[arch_id] = (arch, val_acc, tr_ct + 1)

        return logits_train, logits_val, train_loss, val_loss

    def sample_random_and_update_alphas(self):
        tmp_graph = self.search_space.clone()
        tmp_graph.sample_random_architecture()

        if self.graph.get_type() == "nasbench201":
            sample = tmp_graph.get_op_indices()
        elif self.graph.get_type() == "darts":
            sample = convert_compact_to_genotype(tmp_graph.get_compact())
        self.set_alphas_from_path(sample)

    def get_final_architecture(self):
        all_archs = self.population[:]
        all_archs.extend(self.history[:])
        s_all_archs = sorted(all_archs, key=lambda x : x[1])
        top_arch = s_all_archs[-1][0]
        self.set_alphas_from_path(top_arch)
        logger.info(
            "Arch weights before discretization: {}".format(
                [a for a in self.architectural_weights]
            )
        )
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                alphas = edge.data.alpha.detach().cpu()
                edge.data.set("op", primitives[np.argmax(alphas)])

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.prepare_evaluation()
        graph.parse()
        graph = graph.to(self.device)
        return graph

    def _pareto_fronts(self, sample):
    
        def _is_dominated(indiv, others):
            for i in others:
                if self.population[i][1] > indiv[1] and self.population[i][2] <= indiv[2]:
                    return True
            return False

        front1 = []
        front2 = []
        front3 = []
        notf1 = []
        for i in sample:
            if _is_dominated(self.population[i], sample):
                notf1.append(i)
            else:
                front1.append(i)
        for i in notf1:
            if _is_dominated(self.population[i], notf1):
                front3.append(i)
            else:
                front2.append(i)
        return front1, front2, front3

    def sample_and_mutate(self, n_select=5, n_sample=10):

        def mutate(graph, parent, mutation_rate=1):
            if graph.get_type() == "nasbench201":
                parent_op_indices = parent
                op_indices = list(parent_op_indices)

                edge = np.random.choice(len(parent_op_indices))
                available = [o for o in range(5) if o != parent_op_indices[edge]]
                op_index = np.random.choice(available)
                op_indices[edge] = op_index
                return op_indices

            elif graph.get_type() == "darts":
                parent_compact = convert_genotype_to_compact(parent)
                parent_compact = make_compact_mutable(parent_compact)
                compact = parent_compact

                for _ in range(int(mutation_rate)):
                    cell = np.random.choice(2)
                    pair = np.random.choice(8)
                    num = np.random.choice(2)
                    if num == 1:
                        compact[cell][pair][num] = np.random.choice(7)
                    else:
                        inputs = pair // 2 + 2
                        choice = np.random.choice(inputs)
                        if pair % 2 == 0 and compact[cell][pair + 1][num] != choice:
                            compact[cell][pair][num] = choice
                        elif pair % 2 != 0 and compact[cell][pair - 1][num] != choice:
                            compact[cell][pair][num] = choice

                return convert_compact_to_genotype(compact)
            else:
                return

        sample = list(range(len(self.population))) # np.random.choice(list(range(len(self.population))), size=n_sample, replace=False)
        if self.innov_protect:
            f1, f2, f3 = self._pareto_fronts(sample)
            sf1 = sorted(f1, key = lambda x : self.population[x][1])
            sf2 = sorted(f2, key = lambda x : self.population[x][1])
            sf3 = sorted(f3, key = lambda x : self.population[x][1])
            sf = sf3 + sf2 + sf1
            #top_a = sf1[(-1*n_select):]
            #bot_a = sf3[:n_select]
            bot_a = sf[:n_select]
            top_a = sf[(-1*n_select):]
        else:
            ssample = sorted(sample, key = lambda x : self.population[x][1])
            bot_a = ssample[:n_select]
            top_a = ssample[(-1*n_select):]
        to_add = []
        for i in top_a:
            to_add.append(mutate(self.graph, self.population[i][0]))
        to_remove = sorted(bot_a, reverse=True)
        for i in to_remove:
            try:
                remd = self.population.pop(i)
            except IndexError:
                logger.info("Selected inds: "+str(to_remove))
                raise IndexError
            self.history.append(remd)
        for c_arch in to_add:
            self.population.append((c_arch, 0, 0))
        self.pop_iter = iter(range(len(self.population)))

