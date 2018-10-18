from ImageProcessor import *


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
