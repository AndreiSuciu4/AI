import networkx as nx
import matplotlib.pyplot as plt
import warnings
import tsplib95


warnings.simplefilter('ignore')

from Chromosome import Chromosome
from GA import GA

def readNetTSSP(filename):
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

def fcEval(communities, param):
    noNodes = param['noNodes']
    mat = param['mat']
    sum = 0
    for i in range(0, len(communities) - 1):
        if mat[(communities[i] - 1), (communities[i + 1] - 1)] == 0:
            sum += 999999
        sum += mat[(communities[i] - 1), (communities[i + 1] - 1)]
    return sum

def TSP(filename, source, destination):
    gaParam = {'popSize': 40, 'noGen': 1000}
    problParam = readNetTSSP(filename)
    #problParam = readNetGML(filename)
    #problParam = readNet(filename)
    problParam['function'] = fcEval
    problParam['source'] = source
    problParam['destination'] = destination

    ga = GA(gaParam, problParam)
    ga.initialisation()
    ga.evaluation()
    bestSolutions = []
    bestFitness = 9999999

    for g in range(gaParam['noGen']):
        ga.oneGenerationElitism()
        bestChromo = ga.bestChromosome()
        if bestFitness > bestChromo.fitness:
            bestSolutions.clear()
            bestFitness = bestChromo.fitness
        for chr in ga.population:
            if chr.fitness == bestChromo.fitness and chr.repres not in bestSolutions :
                bestSolutions.append(chr.repres)

    print(problParam['noNodes'], '\n', bestSolutions,'\n', bestFitness)

    problem = tsplib95.load(filename)
    G = problem.get_graph()
    #G = nx.read_gml(filename)
    pos = nx.spring_layout(G)

    for nod in bestSolutions:
        edges = []
        for i in range(0, problParam['noNodes'] - 1):
            edge = (nod[i], nod[i + 1])
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
    #TSP("dantzig42.tsp", 1, 3)
    TSP("burma14.tsp", 1, 1)
    #TSP("exemplu4.gml",1, 7)
