import numpy as np

from DES import DES
from CMA_ES import CMA_ES


class Hybrid:
    def __init__(self, problem, control_des: dict, control_cma: dict, budget):
        self.problem = problem
        self.control_des = control_des
        self.control_cma = control_cma
        self.lower = problem.lower_bounds
        self.upper = problem.upper_bounds
        self.budget = budget
        self.counteval = 0

    def run(self, x0):
        N = len(x0)
        init_ft = 1/np.sqrt(2)
        best_par = None
        # cma = CMA_ES(self.problem, control=self.control_cma, lower=self.lower, upper=self.upper,
        #              budget=int(self.budget), stop=True)
        # res = cma.run(x0)
        # self.counteval = cma.counteval
        #
        # if not res:
        while self.counteval < self.budget:

            des_budget = int(np.floor(30 * N))
            self.control_des["initFt"] = init_ft
            des = DES(self.problem, control=self.control_des, lower=self.lower, upper=self.upper, budget=des_budget)
            new_x0, des_pop = des.run(
                best_par if best_par is not None else
                x0)
            self.counteval += des.counteval

            budget_left = self.budget - self.counteval
            if budget_left <= 0 or new_x0 is None:
                return
            cma = CMA_ES(self.problem, control=self.control_cma, lower=self.lower, upper=self.upper,
                         budget=budget_left, stop=True)
            res = cma.run(new_x0)
            best_par = cma.best_par
            self.counteval += cma.counteval
            init_ft *= 2
            x0 = self.lower + ((np.random.rand(N) + np.random.rand(N)) * (self.upper - self.problem.lower_bounds) / 2)
            if res:
                return
