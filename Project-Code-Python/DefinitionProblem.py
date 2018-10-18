class DefinitionProblem:
    def is_twobytwo_problem(self, problem):
        return len(problem.figures) == 9

    def get_problem_figures(self, problem):
        problem_figures = []
        possible_files = ['A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                problem_figures.append(problem.figures[figure_name])
        return problem_figures

    def get_solution_figures(self, problem):
        solution_figures = []
        possible_files = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                solution_figures.append(problem.figures[figure_name])
        return solution_figures
