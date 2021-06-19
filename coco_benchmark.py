from __future__ import division, print_function
import cocoex, cocopp
import os, webbrowser
import numpy as np
from DES import DES
from CMA_ES import CMA_ES
from Hybrid import Hybrid

MODE = "Hybrid"

suite_name = "bbob"
output_folder = "Hybrid_I"
budget_multiplier = 1000
dimensions = [2, 3, 5, 10, 20]
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

np.random.seed(123)

for problem in suite:
    if problem.dimension not in dimensions:
        continue
    problem.observe_with(observer)
    x0 = problem.initial_solution
    budget = problem.dimension * budget_multiplier

    if MODE == "CMA":
        solver = CMA_ES(problem, control={}, lower=problem.lower_bounds, upper=problem.upper_bounds, budget=int(budget),
                        stop=False)
    elif MODE == "DES":
        solver = DES(problem, control={}, lower=problem.lower_bounds, upper=problem.upper_bounds, budget=int(budget))

    elif MODE == "Hybrid":
        solver = Hybrid(problem, control_cma={}, control_des={}, budget=budget)

    solver.run(x0)
    # while (problem.evaluations < budget
    #        and not problem.final_target_hit):
    #     solver.run(x0)
    #     x0 = problem.lower_bounds + ((np.random.rand(problem.dimension) + np.random.rand(problem.dimension)) *
    #                 (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

cocopp.main(observer.result_folder)
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

