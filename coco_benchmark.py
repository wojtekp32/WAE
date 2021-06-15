from __future__ import division, print_function
import cocoex, cocopp
from numpy.random import rand
import os, webbrowser

from DES import DES
from CMA_ES import CMA_ES

MODE = "CMA"

suite_name = "bbob"
output_folder = "WAE"
budget_multiplier = 1000
dimension_limit = 5
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

for problem in suite:
    if problem.dimension > dimension_limit:
        break
    problem.observe_with(observer)
    x0 = problem.initial_solution
    budget = problem.dimension * budget_multiplier

    if MODE == "CMA":
        solver = CMA_ES(problem, control={}, lower=problem.lower_bounds, upper=problem.upper_bounds, budget=budget)
    elif MODE == "DES":
        solver = DES(problem, control={}, lower=problem.lower_bounds, upper=problem.upper_bounds, budget=budget)

    while (problem.evaluations < budget
           and not problem.final_target_hit):
        solver.run(x0)
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

cocopp.main(observer.result_folder)
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")

