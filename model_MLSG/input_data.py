import numpy
from collections import deque

numpy.random.seed(12345)
import graph
import random
import networkx as nx
from scipy.sparse import issparse


class InputData:
    def __init__(self, min_count):
        self.get_nodes(min_count)
        self.node_pair_catch = deque()
        self.init_sample_table()
        self.context_count_1 = deque()
        print('Node Count:%d' % len(self.node2id))
        print('Sentence Length:%d' % (self.sentence_length))

    def read_graph(self):
        G = nx.Graph()
        matrix = graph.load_matfile('blogcatalog.mat', variable_name='network')
        if issparse(matrix):
            cx=matrix.tocoo()
            for i,j,v in zip(cx.row,cx.col,cx.data):
                G.add_edge(i, j)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        # G=G.to_undirected()
        return G

    def get_nodes(self, min_count):
        nx_G = self.read_graph()
        self.G = graph.Graph(nx_G, False, 0.2, 0.2)
        self.G.preprocess_transition_probs()
        self.sentence_length = 0
        self.sentence_count = 0
        node_frequency = dict()
        for walk in graph.simulate_walks(G=self.G,num_walks=10, walk_length=80):
            self.sentence_count += 1
            self.sentence_length += len(walk)
            for w in walk:
                try:
                    node_frequency[w] += 1
                except:
                    node_frequency[w] = 1
            self.node2id = dict()
            self.id2node = dict()
            wid = 0
            self.node_frequency = dict()
            for w, c in node_frequency.items():
                if c < min_count:
                    self.sentence_length -= c
                    continue
                self.node2id[w] = wid
                self.id2node[wid] = w
                self.node_frequency[wid] = c
                wid += 1
        self.node_count = len(self.node2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.node_frequency.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    def get_node_pairs(self, window_size):
        node_pairs = []
        for walk in graph.simulate_walks(G=self.G,num_walks=10, walk_length=80):
            node_ids = []
            for node in walk:
                try:
                    node_ids.append(self.node2id[node])
                except:
                    continue
            for i, u in enumerate(node_ids):
                local_node_pairs = []
                for j, v in enumerate(node_ids[max(i - window_size, 0):i + window_size + 1]):
                    assert u < self.node_count
                    assert v < self.node_count
                    if i == j:
                        continue
                    local_node_pairs.append((u, v))
                node_pairs.append(local_node_pairs)
        return node_pairs

    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def evaluate_pair_count(self, window_size):
        return len(self.get_node_pairs(window_size))
        # return self.sentence_length * (2 * window_size - 1) - (self.sentence_count - 1) * (
        #             1 + window_size) * window_size


def test():
    a = InputData(0)
    print(len(a.get_node_pairs(5)))
    print(a.evaluate_pair_count(5))


if __name__ == '__main__':
    test()
