from ACO import ACO
import warnings
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt


warnings.simplefilter('ignore')


def readNetTSP(filename):
    net = {}

    problem = tsplib95.load(filename)

    graph = problem.get_graph()
    mat = nx.to_numpy_matrix(graph)
    n = len(mat)
    net['noNodes'] = n

    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if (mat[i, j] == 1):
                d += 1
            if (j > i):
                noEdges += mat[i, j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    return net

def readNetGML(fileName):
    net = {}

    G = nx.read_gml(fileName)
    n = nx.number_of_nodes(G)
    net['noNodes'] = n

    A = nx.adjacency_matrix(G)
    mat = A.todense()
    net["mat"] = mat
    degrees = []
    noEdges = 0
    for i in range(n):
        d = 0
        for j in range(n):
            if (mat[i, j] == 1):
                d += 1
            if (j > i):
                noEdges += mat[i, j]
        degrees.append(d)
    net["noEdges"] = noEdges
    net["degrees"] = degrees
    return net

def readNet(fileName):
    f = open(fileName, "r")
    net = {}
    n = int(f.readline())
    net['noNodes'] = n
    mat = []
    for i in range(n):
        mat.append([])
        line = f.readline()
        elems = line.split(",")
        for j in range(n):
            mat[-1].append(int(elems[j]))
    net["mat"] = mat
    f.close()
    return net

def TSP(filename):
    acoParam = {'noAnt': 52, 'noGen': 300, 'ro': 0.3}
    #param = readNet(filename)
    param = readNetTSP(filename)
    #param = readNetGML(filename)
    param['alfa'] = 1
    param['beta'] = 1.3
    param['q0'] = 0
    param
    aco = ACO(acoParam, param)
    aco.mmas()

    problem = tsplib95.load(filename)
    G = problem.get_graph()
    #G = nx.read_gml(filename)
    pos = nx.spring_layout(G)
    best = aco.bestAnt().memory
    edges = []
    for i in range (0, len(best) - 1):
        edge = (best[i], best[i + 1])
        edges.append(edge)

    plt.figure(figsize=(30, 30))
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    nx.draw_networkx_nodes(G, pos, node_size=4000, cmap=plt.cm.RdYlBu)
    nx.draw_networkx_edges(G, pos, width=10, alpha=1, edge_color="r", edgelist=edges, nodelist=G.nodes())
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == '__main__':
    TSP("burma14.tsp")
    TSP("berlin52.tsp")
