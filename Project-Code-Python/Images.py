from ImageProcessor import *

class Images:

    def __init__(self, image):
        self.image = image
        self.image_processor = ImageProcessor()
    
    #specify the black_pixel_distance with max and min value for left, right, top, and bottom directions
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

    #set attributes for the key-value set
    def get_attributes(self):
        attributes = {}
        attributes['black_white_ratio'] = self.black_white_ratio
        attributes['left_min'] = self.left_min
        attributes['right_min'] = self.right_min
        attributes['top_min'] = self.top_min
        attributes['bottom_min'] = self.bottom_min

        attributes['left_max'] = self.left_max
        attributes['right_max'] = self.right_max
        attributes['top_max'] = self.top_max
        attributes['bottom_max'] = self.bottom_max

        attributes['left_avg'] = self.left_avg
        attributes['right_avg'] = self.right_avg
        attributes['top_avg'] = self.top_avg
        attributes['bottom_avg'] = self.bottom_avg

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

    def predict_transformed_attrs(self, transformation_analysis, optional_transformation_analysis = None):
        transformed_attrs = {}
        optional_transformed_attrs = {}
        for key,attr in self.get_attributes().items():
            if optional_transformed_attrs:
                new_value = ((attr * transformation_analysis[key]) + (attr * optional_transformation_analysis[key])) / 2
            else:
                new_value = attr * transformation_analysis[key]
            transformed_attrs[key] = new_value

        return transformed_attrs
