import numpy as np
import torch
from PIL import Image
import string
import random

def resize_bbox(bbox, original_size, new_size):
    """
    Resize bounding box coordinates based on original and new image sizes.
    """
    # Unpack bounding box coordinates
    x_min, y_min, width, height   = bbox[1:]
    # Calculate scaling factors
    scale_x = new_size[1] / original_size[1]
    scale_y = new_size[0] / original_size[0]
    # Resize bounding box coordinates
    resized_bbox = [
        (x_min * scale_x),
        (y_min * scale_y),
        (height * scale_y),
        (width * scale_x)
    ]
    resized_bbox = [int(x) for x in resized_bbox]
    return resized_bbox

def generate_random_string(length):
    """
    Generate a random string of specified length.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def convert_pil_to_opencv(pil_image):
    """
    Convert a PIL Image to an OpenCV image.
    """
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return opencv_image