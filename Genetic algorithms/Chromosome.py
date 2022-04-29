from random import randint

def generateARandomPermutation(n):
    perm = [i for i in range(1, n + 1)]
    pos1 = randint(0, n - 1)
    pos2 = randint(0, n - 1)
    perm[pos1], perm[pos2] = perm[pos2], perm[pos1]
    return perm

#vector de comunitati
class Chromosome:
    def __init__(self, problParam = None):
        self.__problParam = problParam
        self.__repres = []
        self.__repres = generateARandomPermutation(problParam['noNodes'])
        #for i in range(0, problParam['noNodes']):
        #    if self.__repres[i] == problParam['source']:
        #        self.__repres[i] = self.__repres[0]
        #        self.__repres[0] = problParam['source']
        #    if self.__repres[i] == problParam['destination']:
        #        self.__repres[i] = self.__repres[problParam['noNodes'] - 1]
        #        self.__repres[problParam['noNodes'] - 1] = problParam['destination']
        for i in range(0, problParam['noNodes']):
            if self.__repres[i] == problParam['source']:
                self.__repres[i] = self.__repres[0]
                self.__repres[0] = problParam['source']
        self.__repres.append(problParam['destination'])
        self.__fitness = 0.0

    @property
    def repres(self):
        return self.__repres

    @property
    def fitness(self):
        return self.__fitness

    @repres.setter
    def repres(self, l=[]):
        self.__repres = l

    @fitness.setter
    def fitness(self, fit=0.0):
        self.__fitness = fit

    #Încrucişare parţial transformată
    def crossover(self, second):
        newrepres = []
        for i in range(len(self.repres)):
            newrepres.append(second.repres[self.repres.index(second.repres[i])])
        offspring = Chromosome(self.__problParam)
        offspring.__repres = newrepres
        return offspring

    def mutation(self):
        pos1 = randint(1, self.__problParam['noNodes'] - 2)
        pos2 = randint(1, self.__problParam['noNodes'] - 2)
        self.__repres[pos1], self.__repres[pos2] = self.__repres[pos2], self.__repres[pos1]

    def __str__(self):
        return '\nChromo: ' + str(self.__repres) + ' has fit: ' + str(self.__fitness)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, c):
        return self.__repres == c.__repres and self.__fitness == c.__fitness