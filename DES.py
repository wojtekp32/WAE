import sys
import numpy as np
import random


class DES:
    def __init__(self, problem, control: dict, lower, upper, budget):
        self.problem = problem
        self.control = control
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.counteval = 0

    def controlParam(self, name, default):
        if name in self.control.keys():
            return self.control[name]
        else:
            return default

    def sample_from_history(self, history, history_sample, lmbd):
        ret = []
        for i in range(lmbd):
            ret.append(np.random.choice(len(history[history_sample[i]]), 1))
        return ret

    def delete_infs_nans(self, X):
        for i in range(len(X)):
            if X[i] == np.NaN or X[i] == np.Inf:
                X[i] = sys.float_info.max
        return X

    def fn(self, x, problem):
        if (self.lower <= x).all() and (x <= self.upper).all():
            self.counteval += 1
            return problem(x)
        else:
            return sys.float_info.max

    def fn_l(self, X, problem):
        if self.counteval + len(X) <= self.budget:
            return [self.fn(x, problem) for x in X]
        else:
            ret = []
            budget_left = self.budget - self.counteval
            if budget_left > 0:
                ret = [self.fn(X[i], problem) for i in range(max(budget_left,0))]
            return ret + [sys.float_info.max] * (len(X) - budget_left)

    def back_bounce_boundary(self, x):
        if (self.lower <= x).all() and (x <= self.upper).all():
            return x
        elif (x < self.lower).any():
            for i in range(len(x)):
                if x[i] < self.lower[i]:
                    x[i] = self.lower[i] + abs(self.lower[i] - x[i]) % (self.upper[i] - self.lower[i])
        elif (x > self.upper).any():
            for i in range(len(x)):
                if x[i] > self.upper[i]:
                    x[i] = self.upper[i] - abs(self.upper[i] - x[i]) % (self.upper[i] - self.lower[i])

        x = self.delete_infs_nans(x)
        return self.back_bounce_boundary(x)

    def gen_sample(self, l, u):
        sample = []
        for i in range(len(l)):
            sample.append(l[i] + np.random.rand() * (u[i] - l[i]))
        return sample


    def run(self, p0):
        self.counteval = 0

        N = len(p0)
        controlParam = self.controlParam
        Ft = self.controlParam("Ft", 1)  ## Scaling factor of difference vectors (a variable!)
        initFt = self.controlParam("initFt", 1)
        stopfitness = controlParam("stopfitness", -np.Inf)  ## Fitness value after which the convergence is reached
        ## Strategy parameter setting:
        initlambda = controlParam("lambda", 4 * N)  ## Population starting size
        lmbd = initlambda  ## Population size
        mu = controlParam("mu", np.floor(lmbd / 2))  ## Selection size
        weights = controlParam("weights", [np.log(mu+1) - np.log(i+1) for i in range(int(mu))])  ## Weights to calculate mean from selected individuals
        weights = weights / sum(weights)  ##    \-> weights are normalized by the sum
        weightsSumS = sum(weights ** 2)  ## weights sum square
        mueff = controlParam("mueff", sum(weights) ** 2 / sum(weights ** 2))  ## Variance effectiveness factor
        cc = controlParam("ccum", mu / (mu + 2))  ## Evolution Path decay factor
        pathLength = controlParam("pathLength", 6)  ## Size of evolution path
        cp = controlParam("cp", 1 / np.sqrt(N))  ## Evolution Path decay factor
        maxiter = controlParam("maxit", np.floor(self.budget / (lmbd + 1)))  ## Maximum number of iterations after which algorithm stops
        c_Ft = controlParam("c_Ft", 0)
        pathRatio = controlParam("pathRatio", np.sqrt(pathLength))  ## Path Length Control reference value
        histSize = controlParam("history", int(np.ceil(6 + np.ceil(3 * np.sqrt(N)))))  ## Size of the window of history - the step length history
        Ft_scale = controlParam("Ft_scale", ((mueff + 2) / (N + mueff + 3)) / (1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + (mueff + 2) / (N + mueff + 3)))
        tol = controlParam("tol", 10 ^ -12)
        sqrt_N = np.sqrt(N)

        best_fit = np.Inf  ## The best fitness found so far
        best_par = None  ## The best solution found so far
        best_pop = []

        dMean = np.zeros((histSize, N))
        FtHistory = np.zeros(histSize)
        pc = np.zeros((histSize, N))

        msg = None  ## Reason for terminating
        restart_number = -1

        restart_number += 1
        mu = int(np.floor(lmbd / 2))
        weights = [np.log(mu) - np.log(i) for i in range(1, int(mu+1))]
        weights = np.divide(weights, sum(weights))
        weightsPop = [np.log(mu+1) - np.log(i) for i in range(1, lmbd+1)]
        weightsPop = np.divide(weightsPop, sum(weightsPop))

        histHead = -1
        itr = 0
        history = [[]]*histSize
        Ft = initFt

        population = []
        l = np.multiply(self.lower, 0.8)
        u = np.multiply(self.upper, 0.8)
        for i in range(lmbd):
            population.append(self.gen_sample(l, u))
        cumMean = (self.upper + self.lower) / 2
        populationRepaired = [self.back_bounce_boundary(p) for p in population]

        population = populationRepaired

        selection = np.zeros(mu)
        selectedPoints = np.zeros((mu, N))
        fitness = self.fn_l(population, self.problem)
        oldMean = []
        newMean = p0
        limit = 0
        worst_fit = max(fitness)

        popMean = np.dot(weightsPop, population)
        muMean = newMean

        diffs = np.zeros((lmbd, N))
        x1sample = []
        x2sample = []

        chiN = np.sqrt(N)

        histNorm = 1 / np.sqrt(2)
        counterRepaired = 0

        stoptol = False

        while self.counteval < self.budget and not stoptol:
            itr += 1
            histHead = (histHead + 1) % histSize

            mu = int(np.floor(lmbd / 2))
            weights = [np.log(mu + 1) - np.log(i) for i in range(1, mu + 1)]
            weights = np.divide(weights, sum(weights))

            selection = [x for _, x in sorted(zip(fitness, range(len(fitness))))]
            selectedPoints = np.array([population[i] for i in selection[:mu]])

            history[histHead] = np.zeros((N, mu))
            history[histHead] = np.multiply(selectedPoints, histNorm / Ft)

            oldMean = newMean
            newMean = np.dot(weights, selectedPoints)

            muMean = newMean
            dMean[histHead] = np.subtract(muMean, popMean) / Ft

            step = np.subtract(newMean, oldMean) / Ft
            FtHistory[histHead] = Ft
            oldFt = Ft

            if histHead == 0:
                pc[histHead] = (1 - cp) * np.zeros(N) / np.sqrt(N) + np.sqrt(mu * cp * (2 - cp)) * step
            else:
                pc[histHead] = (1 - cp) * pc[histHead - 1] + np.sqrt(mu * cp * (2 - cp)) * step

            limit = histHead + 1 if itr < histSize else histSize

            historySample = np.random.choice(range(limit), lmbd, replace=True)
            historySample2 = np.random.choice(range(limit), lmbd, replace=True)

            x1sample = self.sample_from_history(history, historySample, lmbd)
            x2sample = self.sample_from_history(history, historySample, lmbd)

            for i in range(lmbd):
                x1 = history[historySample[i]][x1sample[i]]
                x2 = history[historySample[i]][x2sample[i]]

                diffs[i] = np.sqrt(cc) * ((x1 - x2) + np.random.randn() * dMean[historySample[i]]) + np.sqrt(1 - cc) \
                           * np.random.randn() * pc[historySample2[i]]

            population = newMean + Ft * diffs + tol * (1 - 2 / N ** 2) ** (itr / 2) * \
                         np.random.randn(diffs.shape[0], diffs.shape[1]) / chiN

            population = [self.delete_infs_nans(p) for p in population]

            populationTemp = population
            populationRepaired = [self.back_bounce_boundary(p) for p in population]

            counterRepaired = 0
            for tt in range(len(populationTemp)):
                if (populationTemp[tt] != populationRepaired[tt]).any():
                    counterRepaired += 1

            population = populationRepaired
            popMean = np.dot(weightsPop, population)

            fitness = self.fn_l(population, self.problem)
            wb = np.argmin(fitness)

            if fitness[wb] == best_fit:
                best_fit = fitness[wb]
                best_par = population[wb]
                best_pop = population
            else:
                best_par = populationRepaired[wb]

            ww = np.argmax(fitness)
            if fitness[ww] > worst_fit:
                worst_fit = fitness[ww]

            cumMean = 0.8 * cumMean + 0.2 * newMean
            cumMeanRepaired = self.back_bounce_boundary(cumMean)

            fn_cum = self.fn_l([cumMeanRepaired], self.problem)[0]

            if fn_cum < best_fit:
                best_fit = fn_cum
                best_par = cumMeanRepaired
                best_pop
            if fitness[0] <= stopfitness:
                msg = "Stop fitness reached."
                break

        return best_par, best_pop










