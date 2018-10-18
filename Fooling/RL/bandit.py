import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, upper_limit=0):
        if upper_limit > 0:
            mn = upper_limit
            N = 1
        else:
            mn = 0
            N = 0

        self.m = m
        self.mean = mn
        self.N = N

    def pull(self):
        return np.random.normal() + self.m

    def update(self, x):
        self.N +=1
        self.mean = (1 - 1/self.N)*self.mean + 1/self.N*x

    def pull_and_update(self):

        x = self.pull()
        self.update(x)

        return x

def run_experiment(means, eps, N, upper_limit=0, ucb1_coeff=0):

    n = len(means)
    bandits = []

    data = np.empty(N)

    for mn in means:
        bandits.append(Bandit(mn, upper_limit))


    for i in range(N):
        rnd = np.random.random()

        if rnd < eps:
            j = np.random.choice(n)
        else:
            j = np.argmax([bnd.mean + ucb1_coeff*np.sqrt(2*np.log(i+1)/max(bnd.N, 1e-6)) for bnd in bandits])

        x = bandits[j].pull_and_update()

        data[i] = x

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    for bandit in bandits:
        print(bandit.mean)

    return cumulative_average


run_experiment([1, 2, 3], 0.1, 100000)

plt.plot(run_experiment([1, 2, 3], 0.1, 100000), label="eps == 0.1")
plt.plot(run_experiment([1, 2, 3], 0.2, 100000), label="eps == 0.2")
plt.plot(run_experiment([1, 2, 3], 0.01, 100000), label="eps == 0.01")
plt.legend()
plt.xscale('log')
plt.show()


plt.plot(run_experiment([1, 2, 3], 0, 100000, upper_limit=10), label="optimistic")
plt.plot(run_experiment([1, 2, 3], 0.1, 100000), label="eps == 0.1")
plt.legend()
plt.xscale('log')
plt.show()

plt.plot(run_experiment([1, 2, 3], 0, 100000, upper_limit=10, ucb1_coeff=1), label="ucb1")
plt.plot(run_experiment([1, 2, 3], 0.1, 100000), label="eps == 0.1")
plt.plot(run_experiment([1, 2, 3], 0, 100000, upper_limit=10), label="optimistic")
plt.legend()
plt.xscale('log')
plt.show()
