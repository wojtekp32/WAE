import numpy as np
import sys


class CMA_ES:
    def __init__(self, problem, control: dict, lower, upper, budget):
        self.control = control
        self.problem = problem
        self.lower = lower
        self.upper = upper
        self.budget = budget

    def control_param(self, name, default):
        if name in self.control.keys():
            return self.control[name]
        else:
            return default

    def fn(self, x, problem):
        if (self.lower <= x).all() and (x <= self.upper).all():
            return problem(x)
        else:
            return sys.float_info.max

    def fn_l(self, X, problem):
        if self.problem.evaluations + len(X) <= self.budget:
            return [self.fn(x, problem) for x in X]
        else:
            ret = []
            budget_left = self.budget - self.problem.evaluations
            if budget_left > 0:
                ret = [self.fn(X[i], problem) for i in range(max(budget_left,0))]
            return ret + [sys.float_info.max] * (len(X) - budget_left)

    def get_selection(self, population, mu):
        fitness = self.fn_l(population, self.problem)
        selection = [x for _, x in sorted(zip(fitness, range(mu)))]
        selectedPoints = np.array([population[i] for i in selection])
        return selectedPoints

    def run(self, p0):
        # Params from control
        control_param = self.control_param
        N = len(p0)
        initlambda = control_param("lambda", 4 + np.floor(3*np.log(N)))
        lmbd = int(initlambda)
        mu = control_param("mu", int(np.floor(lmbd / 2)))
        cm = control_param("cm", 1)
        weights = control_param("weights", [np.log(mu + 1) - np.log(i + 1) for i in range(int(mu))])
        weights = weights / sum(weights)
        ueff = control_param("ueff", np.sum(weights)**2 / np.sum(weights**2))
        u_cov = control_param("u_cov", ueff)
        c_sigma = control_param("c_sigma", (ueff + 2) / (N + ueff + 3))
        d_sigma = control_param("d_sigma", 1 + 2 * max(0, np.sqrt((ueff-1)/(N+1))-1) + c_sigma)
        c_c = control_param("c_c", 4/(N+4))
        c_cov = control_param("c_cov",
                              1/u_cov * 2/(N + np.sqrt(2))**2 + (1 - 1/u_cov) * min(1, (2*u_cov - 1)/((N+2)**2 + u_cov)))

        # Pre-defined params
        E_norm = np.sqrt(N) * (1 - 1 / 4 * N + 1 / 21 * N ** 2)

        # Init
        mean = p0
        C = np.identity(N)
        sigma = 1
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        stoptol = False
        g = 0

        while self.problem.evaluations < self.budget and not stoptol:
            population = []
            for i in range(lmbd):
                population.append(mean + sigma * np.random.multivariate_normal(np.zeros(N), C))

            eigvalues, B = np.linalg.eig(C)
            D_inv = np.diag([1/v for v in eigvalues])
            C_minus_half = (B.dot(D_inv)).dot(np.transpose(B))

            selected_points = self.get_selection(population, mu)
            new_mean = np.dot(weights, selected_points)

            new_p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * ueff) * C_minus_half * \
                          (new_mean - mean) / sigma

            new_sigma = sigma * np.exp(c_sigma/d_sigma * (np.linalg.norm(new_p_sigma)/E_norm - 1))

            H = 1 if np.linalg.norm(new_p_sigma)/np.sqrt(1-(1-c_sigma)**(2*(g+1))) < (1.4 + 2/(N+1) * E_norm) else 0
            new_p_c = (1 - c_c) * p_c + H * np.sqrt(c_c * (2 - c_c) * ueff) * (new_mean - mean) / sigma

            new_C = (1 - c_cov)*C + 1/u_cov * c_cov * np.dot(new_p_c, np.transpose(new_p_c)) + \
                    c_cov * (1 - 1/u_cov) * 1/sigma**2 * \
                    np.sum([weights[i] * np.dot((selected_points[i] - mean), np.transpose(selected_points[i] - mean))
                            for i in range(mu)])

            mean = new_mean
            p_sigma = new_p_sigma
            sigma = new_sigma
            p_c = new_p_c
            C = new_C

            if self.problem.final_target_hit or 10e-15 < np.linalg.cond(C) < 10e+14:
                stoptol = True
























