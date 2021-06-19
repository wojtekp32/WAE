import numpy as np
import sys
import matplotlib.pyplot as plt


class CMA_ES:
    def __init__(self, problem, control: dict, lower, upper, budget, stop):
        self.control = control
        self.problem = problem
        self.lower = lower
        self.upper = upper
        self.budget = budget
        self.stop = stop
        self.counteval = 0

        self.best_par = None
        self.best_fit = sys.float_info.max

    def control_param(self, name, default):
        if name in self.control.keys():
            return self.control[name]
        else:
            return default

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

        return self.back_bounce_boundary(x)

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
                ret = [self.fn(X[i], problem) for i in range(max(budget_left, 0))]
            return ret + [sys.float_info.max] * (len(X) - budget_left)

    def get_selection(self, population, mu):
        fitness = self.fn_l(population, self.problem)
        selection = [x for _, x in sorted(zip(fitness, range(len(fitness))))]
        selectedPoints = np.array([population[i] for i in selection[:mu]])
        if min(fitness) < self.best_fit:
            self.best_par = selectedPoints[0]
            self.best_fit = min(fitness)
        return selectedPoints, min(fitness)

    def run(self, p0):
        self.counteval = 0

        # Params from control
        control_param = self.control_param
        N = len(p0)
        initlambda = control_param("lambda", 4 + np.floor(3 * np.log(N)))
        lmbd = int(initlambda)
        mu = control_param("mu", int(np.floor(lmbd / 2)))
        weights = control_param("weights", [np.log(mu + 1) - np.log(i + 1) for i in range(int(mu))])
        weights = weights / sum(weights)
        ueff = control_param("ueff", np.sum(weights) ** 2 / np.sum(weights ** 2))
        u_cov = control_param("u_cov", ueff)
        c_sigma = control_param("c_sigma", (ueff + 2) / (N + ueff + 3))
        d_sigma = control_param("d_sigma", 1 + 2 * max(0, np.sqrt((ueff - 1) / (N + 1)) - 1) + c_sigma)
        c_c = control_param("c_c", 4 / (N + 4))
        c_cov = control_param("c_cov",
                              1 / u_cov * 2 / (N + np.sqrt(2)) ** 2 + (1 - 1 / u_cov) * min(1, (2 * u_cov - 1) / (
                                      (N + 2) ** 2 + u_cov)))

        # Pre-defined params
        E_norm = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        # Init
        mean = p0
        C = np.identity(N)
        sigma = (self.upper[0] - self.lower[0]) / 3
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        stoptol = False
        g = 0

        TolX = 10e-12 * sigma
        hist = 10 + int(np.ceil(30 * N / lmbd))
        obj_vals = []

        while self.counteval < self.budget:

            population = []
            for_std = []
            for i in range(lmbd):
                normal_vec = np.random.multivariate_normal(np.zeros(N), C)
                population.append(mean + sigma * normal_vec)
                for_std.append(normal_vec)

            population = [self.back_bounce_boundary(p) for p in population]

            eigvalues, B = np.linalg.eig(C)
            if max(abs(eigvalues)) < 10e-9:
                stoptol = True

            D_inv = np.diag([1 / np.sqrt(v) for v in eigvalues])
            C_minus_half = B @ D_inv @ np.transpose(B)

            selected_points, best_fit = self.get_selection(population, mu)
            obj_vals.append(best_fit)
            new_mean = np.dot(weights, selected_points)

            mul = np.matmul(C_minus_half, ((new_mean - mean) / sigma))
            new_p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * ueff) * mul

            new_sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(new_p_sigma) / E_norm - 1))

            H = 1 if np.linalg.norm(new_p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (g + 1))) < (
                    1.4 + 2 / (N + 1) * E_norm) else 0
            new_p_c = (1 - c_c) * p_c + H * np.sqrt(c_c * (2 - c_c) * ueff) * (new_mean - mean) / sigma

            sum_for_C = np.zeros([N, N])
            for i in range(mu):
                sum_for_C = np.add(sum_for_C, weights[i] * np.matmul(np.transpose([selected_points[i] - mean]), [selected_points[i] - mean]))

            new_C = (1 - c_cov) * C + 1 / u_cov * c_cov * np.matmul(np.transpose([new_p_c]), [new_p_c]) + \
                    c_cov * (1 - 1 / u_cov) * 1 / sigma ** 2 * sum_for_C

            mean = new_mean
            p_sigma = new_p_sigma
            sigma = new_sigma
            p_c = new_p_c
            C = new_C
            g += 1

            stop_crit = False

            if len(obj_vals) >= hist and max(obj_vals[-hist:]) - min(obj_vals[-hist:]) < 10e-8:
                stop_crit = True
            if not (10e-14 < np.linalg.cond(C) < 10e+15):
                stop_crit = True
            if (np.std(for_std) < TolX).all() and (p_sigma < TolX).all():
                stop_crit = True

            if stop_crit and self.stop:
                return False

            if self.problem.final_target_hit:
                break

        return True
