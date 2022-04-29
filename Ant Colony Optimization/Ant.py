import random


class Ant():
    def __init__(self, props):
        self.__props = props;
        self.__current = random.randint(0, (props['noNodes'] - 1))
        self.__first = self.__current
        self.__memory = []
        self.__memory.append(self.__current)
        self.__available = [i for i in range(0, props['noNodes'])]
        self.__available.remove(self.__current)
        self.__cost = 0

    @property
    def cost(self):
        return self.__cost

    @property
    def curret(self):
        return self.__current

    @property
    def memory(self):
        return self.__memory

    def numarator(self, i, j):
        feromon = self.__props['feromon']
        mat = self.__props['mat']
        alfa = self.__props['alfa']
        beta = self.__props['beta']
        fer = feromon[i][j]
        dist = mat[i, j]
        ferPow = pow(fer, alfa)
        distPow = pow((1 / dist), beta)
        return ferPow * distPow

    def suma(self, i):
        suma = 0
        for nod in self.__available:
            suma += self.numarator(i, nod)
        return suma


    def next(self):
        q = random.randint(0, 1)
        feromon = self.__props['feromon']
        mat = self.__props['mat']
        alfa = self.__props['alfa']
        beta = self.__props['beta']
        if q <= self.__props['q0'] and len(self.__available) > 0:
            next = 0
            max = -99999
            for nod in self.__available:
                fer =  feromon[self.__current][nod]
                dist = mat[self.__current, nod]
                ferPow = pow(fer, alfa)
                distPow = pow((1 / dist), beta)
                if ferPow * distPow > max:
                    next = nod
                    max = ferPow * distPow
            self.__memory.append(next)
            self.__cost += mat[self.__current, next]
            self.__current = next
            self.__available.remove(next)

        elif len(self.__available) > 0:
            probs = []
            numitor = self.suma(self.__current)
            for nod in self.__available:
                probs.append((self.numarator(self.__current, nod) / numitor) * 100)
            next = random.choices(self.__available, weights=probs, k = 1)
            self.__memory.append(next[0])
            self.__cost += mat[self.__current, next[0]]
            self.__current = next[0]
            self.__available.remove(next[0])

    def crossing(self):
        for _ in range(0, self.__props['noNodes'] - 1):
            self.next()
        self.__memory.append(self.__first)
        self.__cost += self.__props['mat'][self.__first, self.__current]
