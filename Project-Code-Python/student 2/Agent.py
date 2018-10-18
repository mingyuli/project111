from PIL import Image
from PIL import ImageChops


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        problem_definition = DefinitionProblem()
        answer = -1
        # Solve 2x2 problems for Project 1
        if problem_definition.is_twobytwo_problem(problem):
            answer = TwoByTwoSolver(problem).do_basic_analysis()
            # Return answer if first layer analysis succeeds
            if answer != -1:
                return answer
            # Else perform a deeper analysis
            answer = TwoByTwoSolver(problem).do_transformation_analysis()
        return answer


class DefinitionProblem:
    def __init__(self):
        pass

    @staticmethod
    def is_twobytwo_problem(problem):
        return len(problem.figures) == 9

    @staticmethod
    def get_problem_figures(problem):
        problem_figures = []
        possible_files = ['A.png', 'B.png', 'C.png', 'D.png', 'E.png', 'F.png', 'G.png', 'H.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                problem_figures.append(problem.figures[figure_name])
        return problem_figures

    @staticmethod
    def get_solution_figures(problem):
        solution_figures = []
        possible_files = ['1.png', '2.png', '3.png', '4.png', '5.png', '6.png']
        for figure_name in problem.figures:
            file_name = problem.figures[figure_name].visualFilename
            if file_name[-5:] in possible_files:
                solution_figures.append(problem.figures[figure_name])
        return solution_figures


class ImageNameFinder:

    def __init__(self):
        self.image_processor = ImageProcessor()

    # Use root_mean_square_diff < 965 as a similarity condition for the following three
    def find_matching_image_filename(self, guess_image, solution_images):
        for figure in solution_images:
            file_name = figure.visualFilename
            figure_image = Image.open(file_name)
            if self.image_processor.measure_diff(guess_image, figure_image) < 965:
                return file_name


class ImageProcessor:

    def __init__(self):
        pass

    @staticmethod
    def same_images(image1, image2):
        return ImageChops.difference(image1, image2).getbbox() is None

    @staticmethod
    def measure_diff(image1, image2):
        histogram_diff = ImageChops.difference(image1, image2).histogram()
        sum_of_squares = sum(value * (idx ** 2) for idx, value in enumerate(histogram_diff))
        root_mean_square_diff = (sum_of_squares / float(image1.size[0] * image2.size[1])) ** 0.5
        return root_mean_square_diff

    @staticmethod
    def measure_black_white_ratio(image):
        pixels = image.getdata()
        n_white = 0
        for pixel in pixels:
            if pixel == (255, 255, 255, 255):
                n_white += 1
        n = len(pixels)
        return n_white / float(n)

    @staticmethod
    def black_pixel_distance(image, direction, metric):
        if direction == "right":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "top":
            image = image.rotate(90)
        elif direction == "bottom":
            image = image.rotate(-90)
        pixels = image.load()
        distances = []
        for i in range(0, image.size[0]):
            for j in range(0, image.size[1]):
                pixel = pixels[i, j]
                # If a non-white pixel is reached, return it
                if pixel != (255, 255, 255, 255):
                    distances.append(j)
                    break
        if len(distances) != 0:
            if metric == "min":
                return min(distances)
            elif metric == "max":
                return max(distances)
            elif metric == "avg":
                return sum(distances) / len(distances)
        else:
            return 0


class Images:

    def __init__(self, image):
        self.image = image
        self.image_processor = ImageProcessor()

    # specify the black_pixel_distance with max and min value for left, right, top, and bottom directions
    def generate_image_analysis(self):
        self.black_white_ratio = self.image_processor.measure_black_white_ratio(self.image)
        self.left_min = self.image_processor.black_pixel_distance(self.image, "left", "min")
        self.right_min = self.image_processor.black_pixel_distance(self.image, "right", "min")
        self.top_min = self.image_processor.black_pixel_distance(self.image, "top", "min")
        self.bottom_min = self.image_processor.black_pixel_distance(self.image, "bottom", "min")

        self.left_max = self.image_processor.black_pixel_distance(self.image, "left", "max")
        self.right_max = self.image_processor.black_pixel_distance(self.image, "right", "max")
        self.top_max = self.image_processor.black_pixel_distance(self.image, "top", "max")
        self.bottom_max = self.image_processor.black_pixel_distance(self.image, "bottom", "max")

        self.left_avg = self.image_processor.black_pixel_distance(self.image, "left", "avg")
        self.right_avg = self.image_processor.black_pixel_distance(self.image, "right", "avg")
        self.top_avg = self.image_processor.black_pixel_distance(self.image, "top", "avg")
        self.bottom_avg = self.image_processor.black_pixel_distance(self.image, "bottom", "avg")

    # set attributes for the key-value set
    def get_attributes(self):
        attributes = {'black_white_ratio': self.black_white_ratio, 'left_min': self.left_min,
                      'right_min': self.right_min, 'top_min': self.top_min, 'bottom_min': self.bottom_min,
                      'left_max': self.left_max, 'right_max': self.right_max, 'top_max': self.top_max,
                      'bottom_max': self.bottom_max, 'left_avg': self.left_avg, 'right_avg': self.right_avg,
                      'top_avg': self.top_avg, 'bottom_avg': self.bottom_avg}

        return attributes

    # Return percentage difference between images for each attribute
    def image_transformation(self, other_image):
        transformation = {}
        current_image_attrs = self.get_attributes()
        for key, value in other_image.get_attributes().items():
            # If pixel distance is within 2 pixels, consider it 100% the same
            if type(value) == int and current_image_attrs[key] - value < 3:
                transformation[key] = 1.0
            else:
                transformation[key] = float(current_image_attrs[key]) / float(value)

        return transformation

    def predict_transformed_attrs(self, transformation_analysis, optional_transformation_analysis=None):
        transformed_attrs = {}
        optional_transformed_attrs = {}
        for key, attr in self.get_attributes().items():
            if optional_transformed_attrs:
                new_value = ((attr * transformation_analysis[key]) + (attr * optional_transformation_analysis[key])) / 2
            else:
                new_value = attr * transformation_analysis[key]
            transformed_attrs[key] = new_value

        return transformed_attrs


class DeeperTransformationAnalysis:

    def __init__(self):
        pass

    @staticmethod
    def generate_similarity_score(solution_attrs, d_expected_attributes):
        scores = []
        for s_key, s_attr in solution_attrs.items():
            # Ignore cases where values are 0
            if s_attr == 0 or d_expected_attributes[s_key] == 0:
                continue
            attr_score = s_attr / d_expected_attributes[s_key]
            scores.append(attr_score)
        score = sum(scores) / len(scores)

        return score

    @staticmethod
    def analyze_transformation_patterns(problem_cells, attributes):
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

    # for basic 2x2 patterns, check the if solution figure match the 5 patterns
    def do_basic_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)
        figure_finder = ImageNameFinder()

        # All equal
        if self.check_if_all_equal(problem_figures):
            example_figure_image = Image.open(problem_figures[0].visualFilename)  # Pass in any figure
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
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure C

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        # x-axis flip transformation
        if self.check_if_x_axis_flip(problem_figures):
            for f in problem_figures:
                if "B.png" in f.visualFilename:
                    f_image = Image.open(f.visualFilename).transpose(Image.FLIP_TOP_BOTTOM)
                    solution_figure_filename = figure_finder.find_matching_image_filename(f_image,
                                                                                          solution_figures)  # Pass in figure B

                    # If proposed solution figure is present, return the number of that image
                    if solution_figure_filename:
                        solution_number = int(solution_figure_filename[-5])  # Get image number from filename string
                        return solution_number

        return -1

    # Keep solution figures that do match the above observed patterns
    def Keep_valid_solution_figures(self, solution_figures, patterns):
        sifted_solution_figures = solution_figures
        for p_key, pattern in patterns.items():
            for s_key, solution in solution_figures.items():
                solution_attrs = solution.get_attributes()
                valid = False
                # If solution cell attribute is equal to the pattern, add it to sifted_solution_figures
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
                        sifted_solution_figures.pop(s_key, None)

        return sifted_solution_figures

    # A detailed transformation analysis on multiple images.
    # Returns solution number if certainty is above the appropriate, else returns -1.
    def do_transformation_analysis(self):
        problems = DefinitionProblem()
        problem_figures = problems.get_problem_figures(self.problem)
        solution_figures = problems.get_solution_figures(self.problem)

        ATTRIBUTES = ['black_white_ratio', 'left_min', 'right_min', 'top_min', 'bottom_min',
                      'left_max', 'right_max', 'top_max', 'bottom_max',
                      'left_avg', 'right_avg', 'top_avg', 'bottom_avg']

        image_problems = {}
        for figure in problem_figures:
            if "A.png" in figure.visualFilename:
                image_problems['a'] = Images(Image.open(figure.visualFilename))
            elif "B.png" in figure.visualFilename:
                image_problems['b'] = Images(Image.open(figure.visualFilename))
            elif "C.png" in figure.visualFilename:
                image_problems['c'] = Images(Image.open(figure.visualFilename))

        image_solutions = {}
        for figure in solution_figures:
            if "1.png" in figure.visualFilename:
                image_solutions['1'] = Images(Image.open(figure.visualFilename))
            elif "2.png" in figure.visualFilename:
                image_solutions['2'] = Images(Image.open(figure.visualFilename))
            elif "3.png" in figure.visualFilename:
                image_solutions['3'] = Images(Image.open(figure.visualFilename))
            elif "4.png" in figure.visualFilename:
                image_solutions['4'] = Images(Image.open(figure.visualFilename))
            elif "5.png" in figure.visualFilename:
                image_solutions['5'] = Images(Image.open(figure.visualFilename))
            elif "6.png" in figure.visualFilename:
                image_solutions['6'] = Images(Image.open(figure.visualFilename))

        for _, problem_cell in image_problems.items():
            problem_cell.generate_image_analysis()

        for _, solution_cell in image_solutions.items():
            solution_cell.generate_image_analysis()

        # Find ABC, AB, AC transformations where attributes remain close to the same
        deeper_analysis = DeeperTransformationAnalysis()
        transformation_patterns = deeper_analysis.analyze_transformation_patterns(image_problems, ATTRIBUTES)
        ABC_pattern = transformation_patterns['abc']
        AB_pattern = transformation_patterns['ab']
        AC_pattern = transformation_patterns['ac']

        # If any ABC attribute or AB, AC transormation attribute remains exactly the same,
        # that attributes value should be expected in the solution.
        # Remove possible solutions cells that do not match these patterns.
        unsifted_images = dict(image_solutions)
        sifted_solution_images = self.Keep_valid_solution_figures(unsifted_images, ABC_pattern)  # Sift for ABC pattern
        sifted_solution_images = self.Keep_valid_solution_figures(sifted_solution_images, AB_pattern)  # Sift for AB pattern
        sifted_solution_images = self.Keep_valid_solution_figures(sifted_solution_images, AC_pattern)  # Sift for AC pattern
        if not sifted_solution_images:
            sifted_solution_images = image_solutions

        # Generate analyses for AB, AC transformations
        ab_trans_analysis = image_problems['a'].image_transformation(image_problems['b'])
        ac_trans_analysis = image_problems['a'].image_transformation(image_problems['b'])

        # Predict the attributes of D that AD, BD, CD transformations will produce based on the observed AB, AC transformations
        abac_to_a = image_problems['a'].predict_transformed_attrs(ab_trans_analysis, ac_trans_analysis)
        ac_to_b = image_problems['b'].predict_transformed_attrs(ac_trans_analysis)
        ab_to_c = image_problems['c'].predict_transformed_attrs(ab_trans_analysis)

        d_expected_attributes = {}
        for attr in ATTRIBUTES:
            total = abac_to_a[attr] + ac_to_b[attr] + ab_to_c[
                attr]
            average = total / 3
            d_expected_attributes[attr] = average

        # Generate similarity scores for each sifted solution
        similarity_scores = {}
        for key, solution in sifted_solution_images.items():
            solution_attrs = solution.get_attributes()
            similarity_score = deeper_analysis.generate_similarity_score(solution_attrs, d_expected_attributes)
            similarity_scores[key] = similarity_score

        # Choose the solution with the highest similarity score
        answer = int(max(similarity_scores, key=similarity_scores.get))
        similarity = similarity_scores[str(answer)]

        # If the combined similarity score is above 95% threshold, then return the number of the  solution image,
        # otherwise, if sift out enough solutions, return the solution with the highest similarity score,
        # otherwise, return -1
        if answer and similarity > 0.95:
            return answer
        elif len(sifted_solution_images) < 5:
            return answer
        else:
            return -1
