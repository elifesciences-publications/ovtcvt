import networkx as nx
import numpy as np

g = nx.Graph()

nodes = range(-10, 11)

clicque = [(+1, -1), (+1, 0), (+1, +1), (0, -1),
           (0, +1), (+1, -1), (-1, 0), (-1, +1)]

for x in nodes:
    for y in nodes:
        for xc, yx in clicque:
            g.add_edge((x, y), (x + xc, y + yx))


def path(g, start):
    g = g.copy()
    ns = g.neighbors(start)

    if len(ns) == 0 and len(g.nodes()) == 1:
        return [start]
    if len(ns) == 0:
        return False

    g.remove_node(start)

    while len(ns) > 0:
        nextnode = np.random.randint(0, len(ns), 1)
        node = ns[nextnode]
        del ns[nextnode]
        p = path(g, node)
        if p:
            p.append(node)
            return p
    return False


def make_tsp_file(fname, g):
    f = open(fname, 'w')
    f.write('''NAME : GridGraph
COMMENT : Test
TYPE : TSP
DIMENSION : %d
EDGE_DATA_FORMAT : EDGE_LIST
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION\n''' % len(g.nodes()))
    mapping = {}
    for i, n in enumerate(g.nodes()):
        mapping[n] = i + 1
    for i, (x, y) in enumerate(g.nodes()):
        f.write('%03.0d %03.0d %03.0d\n' % (i + 1, x, y))
    f.write('EDGE_DATA_SECTION\n')
    for start, end in g.edges():
        f.write('%3.0d %3.0d\n' % (mapping[start], mapping[end]))
    f.close()


def get_tour(fname, g):
    mapping = {}
    for i, n in enumerate(g.nodes()):
        mapping[i] = n

    tour = []
    for i, line in enumerate(open(fname)):
        if i == 0:
            continue
        l = line.split()
        for t in l:
            tour.append(int(t))
    nodes = []
    for t in tour:
        nodes.append(mapping[t])
    return nodes


def hamiltonian(g, start):
    count = 0
    mlen = 0
    while True:

        path = random_path(g, start)
        mlen = max(mlen, len(path))
        if len(path) == len(g.nodes()):
            return path
        count += 1
        if count % 100 == 0:
            print count, len(g.nodes()), mlen


def random_path(g, start):
    node_list = [start]
    g = g.copy()
    while True:
        ns = g.neighbors(node_list[-1])
        if len(ns) == 0:
            return node_list
        g.remove_node(node_list[-1])
        nextnode = np.random.randint(0, len(ns), 1)
        node_list.append(ns[nextnode])
