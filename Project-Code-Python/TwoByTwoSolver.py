from Images import *
from ImageNameFinder import *
from DefinitionProblem import *


class TwoByTwoSolver:

    def __init__(self, problem):
        self.image_analyzer = ImageProcessor()
        self.problem = problem

    def check_if_all_equal(self, problem_images):
        cell_images = []
        for image in problem_images:
            figure_image = Image.open(image.visualFilename)
            cell_images.append(figure_image)
        for i in range(len(cell_images) - 1):
            if not self.image_analyzer.measure_diff(cell_images[i], cell_images[i + 1]) < 965:
                return False
        return True

    def check_if_ab_equal(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            figure_image = Image.open(figure.visualFilename)
            cell_images[figure.visualFilename[-5]] = figure_image
        if self.image_analyzer.measure_diff(cell_images['A'], cell_images['B']) < 965:
            return True
        else:
            return False

    def check_if_ac_equal(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            figure_image = Image.open(figure.visualFilename)
            cell_images[figure.visualFilename[-5]] = figure_image
        if self.image_analyzer.measure_diff(cell_images['A'], cell_images['C']) < 965:
            return True
        else:
            return False

    # Checks if image is flipped on it's y-axis when crossing the y-axis of the matrix
    def check_if_y_axis_flip(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            cell_images[file_name[-5]] = figure_image
        # Check if flipping figure A across the y-axis produces figure B
        if self.image_analyzer.measure_diff(cell_images['A'].transpose(Image.FLIP_LEFT_RIGHT),
                                            cell_images['B']) < 965:
            return True
        else:
            return False

    # Checks if image is flipped on it's x-axis when crossing the x-axis of the matrix
    def check_if_x_axis_flip(self, problem_images):
        cell_images = {}
        for figure in problem_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            cell_images[file_name[-5]] = figure_image
        # Check if flipping figure A across the y-axis produces figure C
        if self.image_analyzer.measure_diff(cell_images['A'].transpose(Image.FLIP_TOP_BOTTOM),
                                            cell_images['C']) < 965:
            return True
        else:
            return False

    # Perform 5 checks for very basic 2x2 patterns
    def do_basic_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)
        figure_finder = ImageNameFinder()

        # All equal
        if self.check_if_all_equal(problem_figures):
            example_figure_image = Image.open(problem_figures[0].visualFilename) # Pass in any figure
            solution_figure_filename = figure_finder.find_matching_image_filename(example_figure_image,
                                                                                   solution_figures)
            if solution_figure_filename:
                solution_number = solution_figure_filename[-5]  # Get image number from filename string
                return int(solution_number)

        # AB equal
        if self.check_if_ab_equal(problem_figures):
            for f in problem_figures:
                if "C.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                           solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # AC equal
        if self.check_if_ac_equal(problem_figures):
            for f in problem_figures:
                if "B.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                           solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # y-axis flip transformation
        if self.check_if_y_axis_flip(problem_figures):
            for f in problem_figures:
                if "C.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename).transpose(Image.FLIP_LEFT_RIGHT)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image, solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # x-axis flip transformation
        if self.check_if_x_axis_flip(problem_figures):
            for f in problem_figures:
                if "B.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename).transpose(Image.FLIP_TOP_BOTTOM)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image, solution_figures)  # Pass in figure B

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        return -1

    # Keep solution cells that do match the above observed patterns
    def sift_solution_cells(self, solution_cells, patterns):
        sifted_solution_cells = solution_cells
        for p_key, pattern in patterns.items():
            for s_key, solution in solution_cells.items():
                solution_attrs = solution.get_attributes()
                valid = False
                # If solution cell attribute is equal to the pattern, add it to sifted_solution_cells
                if round(solution_attrs[p_key], 5) == round(pattern, 5):
                    valid = True
                else:
                    # Otherwise, if solution is a very close int, consider it acceptable
                    if type(solution_attrs[p_key]) == int:
                        if solution_attrs[p_key] == pattern + 1 or solution_attrs[p_key] == pattern + 2:
                            valid = True
                        if solution_attrs[p_key] == pattern - 1 or solution_attrs[p_key] == pattern - 2:
                            valid = True
                    # Otherwise, solution is missing a pattern, so remove it
                    if valid == False:
                        sifted_solution_cells.pop(s_key, None)

        return sifted_solution_cells

    def generate_similarity_score(self, solution_attrs, d_expected_attributes):
        scores = []
        for s_key, s_attr in solution_attrs.items():
            # Ignore cases where values are 0
            if s_attr == 0 or d_expected_attributes[s_key] == 0:
                continue
            attr_score = s_attr / d_expected_attributes[s_key]
            scores.append(attr_score)
        score = sum(scores) / len(scores)

        return score

    def analyze_transformation_patterns(self, problem_cells, attributes):
        transformation_patterns = {}
        abc_pattern = {}
        ab_pattern = {}
        ac_pattern = {}
        for attr in attributes:
            a_value = problem_cells['a'].get_attributes()[attr]
            b_value = problem_cells['b'].get_attributes()[attr]
            c_value = problem_cells['c'].get_attributes()[attr]

            if a_value == b_value == c_value:
                abc_pattern[attr] = a_value
            else:
                if a_value == b_value:
                    ab_pattern[attr] = c_value
                elif a_value == c_value:
                    ac_pattern[attr] = b_value

            # For close enough pixel distances, will also have a check
            if type(a_value) == int:
                if (a_value == b_value + 1 == c_value + 1 or a_value == b_value - 1 == c_value - 1
                        or a_value == b_value + 1 == c_value - 1 or a_value == b_value - 1 == c_value + 1
                        or a_value == b_value + 2 == c_value + 2 or a_value == b_value - 2 == c_value - 2
                        or a_value == b_value + 2 == c_value - 2 or a_value == b_value - 2 == c_value + 2):
                    abc_pattern[attr] = a_value
                else:
                    if (a_value == b_value or a_value == b_value + 1 or a_value == b_value - 1
                            or a_value == b_value + 2 or a_value == b_value - 2):
                        ab_pattern[attr] = c_value  # Expect C value in cell D
                    elif (a_value == c_value or a_value == c_value + 1 or a_value == c_value - 1
                          or a_value == c_value + 2 or a_value == c_value - 2):
                        ac_pattern[attr] = b_value  # Expect B value in cell D
            else:  # for close enough black/white ratios
                if round(a_value, 5) == round(b_value, 5) == round(c_value, 5):
                    abc_pattern[attr] = a_value
                else:
                    if round(a_value, 5) == round(b_value, 5):
                        ab_pattern[attr] = c_value  # Expect C value in cell D
                    elif round(a_value, 5) == round(c_value, 5):
                        ac_pattern[attr] = b_value  # Expect B value in cell D
        transformation_patterns['abc'] = abc_pattern
        transformation_patterns['ab'] = ab_pattern
        transformation_patterns['ac'] = ac_pattern

        return transformation_patterns

    # A detailed transformation analysis on multiple images.
    # Returns solution number if certainty is above the appropriate, else returns -1.
    def do_transformation_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)

        ATTRIBUTES = ['black_white_ratio', 'left_min', 'right_min', 'top_min', 'bottom_min',
                      'left_max', 'right_max', 'top_max', 'bottom_max',
                      'left_avg', 'right_avg', 'top_avg', 'bottom_avg']

        # Init A,B,C cells with their images
        problem_cells = {}
        for figure in problem_figures:
            if "A.png" in figure.visualFilename:
                problem_cells['a'] = Images(Image.open(figure.visualFilename))
            elif "B.png" in figure.visualFilename:
                problem_cells['b'] = Images(Image.open(figure.visualFilename))
            elif "C.png" in figure.visualFilename:
                problem_cells['c'] = Images(Image.open(figure.visualFilename))

        # Instantiate 1-6 cells with their images
        solution_cells = {}
        for figure in solution_figures:
            if "1.png" in figure.visualFilename:
                solution_cells['1'] = Images(Image.open(figure.visualFilename))
            elif "2.png" in figure.visualFilename:
                solution_cells['2'] = Images(Image.open(figure.visualFilename))
            elif "3.png" in figure.visualFilename:
                solution_cells['3'] = Images(Image.open(figure.visualFilename))
            elif "4.png" in figure.visualFilename:
                solution_cells['4'] = Images(Image.open(figure.visualFilename))
            elif "5.png" in figure.visualFilename:
                solution_cells['5'] = Images(Image.open(figure.visualFilename))
            elif "6.png" in figure.visualFilename:
                solution_cells['6'] = Images(Image.open(figure.visualFilename))

        for _, problem_cell in problem_cells.items():
            problem_cell.generate_image_analysis()

        for _, solution_cell in solution_cells.items():
            solution_cell.generate_image_analysis()

        # Find ABC, AB, AC transformations where attributes remain close to the same
        transformation_patterns = self.analyze_transformation_patterns(problem_cells, ATTRIBUTES)
        abc_pattern = transformation_patterns['abc']
        ab_pattern = transformation_patterns['ab']
        ac_pattern = transformation_patterns['ac']

        # If any ABC attribute or AB, AC transormation attribute remains exactly
        # the same, that attributes value should be expected in the solution.
        # Remove possible solutions cells that do not match these patterns.
        unsifted_cells = dict(solution_cells)
        sifted_solution_cells = self.sift_solution_cells(unsifted_cells, abc_pattern)  # Sift for ABC pattern
        sifted_solution_cells = self.sift_solution_cells(sifted_solution_cells, ab_pattern)  # Sift for AB pattern
        sifted_solution_cells = self.sift_solution_cells(sifted_solution_cells, ac_pattern)  # Sift for AC pattern
        if not sifted_solution_cells:
            sifted_solution_cells = solution_cells

        # Generate analyses for AB, AC transformations
        ab_trans_analysis = problem_cells['a'].image_transformation(problem_cells['b'])
        ac_trans_analysis = problem_cells['a'].image_transformation(problem_cells['b'])

        # Predict the attributes of D that AD, BD, CD transformations will produce based on the observed AB, AC transformations
        abac_to_a = problem_cells['a'].predict_transformed_attrs(ab_trans_analysis,ac_trans_analysis)
        ac_to_b = problem_cells['b'].predict_transformed_attrs(ac_trans_analysis)
        ab_to_c = problem_cells['c'].predict_transformed_attrs(ab_trans_analysis)

        d_expected_attributes = {}
        for attr in ATTRIBUTES:
            total = abac_to_a[attr] + ac_to_b[attr] + ab_to_c[
                attr]
            average = total / 3
            d_expected_attributes[attr] = average

        # Generate similarity scores for each sifted solution
        similarity_scores = {}
        for key, solution in sifted_solution_cells.items():
            solution_attrs = solution.get_attributes()
            similarity_score = self.generate_similarity_score(solution_attrs, d_expected_attributes)
            similarity_scores[key] = similarity_score

        # Choose the solution with the highest similarity score
        proposed_solution = int(max(similarity_scores, key=similarity_scores.get))
        similarity = similarity_scores[str(proposed_solution)]

        # If the combined similarity score is above 95% threshold, then return the number of the  solution image,
        # otherwise, if sift out enough solutions, return the solution with the highest similarity score,
        # otherwise, return -1
        if proposed_solution and similarity > 0.95:
            return proposed_solution
        elif len(sifted_solution_cells) < 5:
            return proposed_solution
        else:
            return -1