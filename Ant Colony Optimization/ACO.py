import random

from Ant import Ant

class ACO():
    def __init__(self, AcoParam=None, AntParam=None):
        self.__AcoParam = AcoParam
        self.__AntParam = AntParam
        self.__population = []
        self.__feromon = []
        for i in range(0, self.__AntParam['noNodes']):
            l = []
            for j in range(0, self.__AntParam['noNodes']):
                l.append(1)
            self.__feromon.append(l)
        self.__AntParam['feromon'] = self.__feromon

    @property
    def population(self):
        return self.__population

    def initialisation(self):
        for i in range(0, self.__AcoParam['noAnt']):
            ant = Ant(self.__AntParam)
            self.__population.append(ant)

    def crossing(self):
        for ant in self.__population:
            ant.crossing()

    def bestAnt(self):
        bestAnt = self.__population[0]
        min = bestAnt.cost
        for ant in self.__population:
            if ant.cost < min:
                min = ant.cost
                bestAnt = ant
        return bestAnt


    def perturbareGraf(self):
        nod1 = random.randint(0, self.__AntParam['noNodes'] - 1)
        nod2 = random.randint(0, self.__AntParam['noNodes'] - 1)
        while nod1 == nod2:
            nod2 = random.randint(0, self.__AntParam['noNodes'] - 1)
        adaus = random.randint(0, 100)
        self.__AntParam['mat'][nod1, nod2] += adaus


    def mmas(self):
        for _ in range(0, self.__AcoParam['noGen']):
            #self.perturbareGraf()
            self.population.clear()
            self.initialisation()
            self.crossing()
            bestAnt = self.bestAnt()
            for i in range (0, self.__AntParam['noNodes']):
                for j in range(0, self.__AntParam['noNodes']):
                    self.__feromon[i][j] = self.__feromon[i][j] * (1 - self.__AcoParam['ro'])
            nodes = bestAnt.memory
            for i in range(0, len(nodes) - 1):
                self.__feromon[nodes[i]][nodes[i + 1]] = self.__feromon[nodes[i]][nodes[i + 1]] + self.__AcoParam['ro'] * (1 / bestAnt.cost)
            self.__AntParam['feromon'] = self.__feromon

        bestAnt = self.bestAnt()
        print(bestAnt.memory)
        print(bestAnt.cost)

