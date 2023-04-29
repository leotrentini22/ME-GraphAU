import numpy as np
import os

import openpifpaf
from PIL import Image
import json

image_path = '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg'
checkpoint = 'shufflenetv2k30-wholebody'
json_output = '/home/trentini/00262.json'

# Load image
img = Image.open(image_path)

# Create predictor object
predictor = Predictor(checkpoint=checkpoint)

# Run prediction on image
predictions, _ = predictor(img)

# Convert predictions to JSON and save to file
with open(json_output, 'w') as f:
    json.dump(predictions, f)

