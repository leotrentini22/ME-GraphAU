import numpy as np
import os

import openpifpaf
import json

image_path = '/work/vita/datasets/Aff-Wild2/cropped_aligned/430/00262.jpg'
checkpoint = 'shufflenetv2k30-wholebody'
json_output = '/home/trentini/00262.json'

# Load image
img = openpifpaf.datasets.PilImage.open(image_path)

# Create predictor object
predictor = openpifpaf.Predictor(checkpoint=checkpoint)

# Run prediction on image
predictions, _ = predictor(img)

# Convert predictions to JSON and save to file
with open(json_output, 'w') as f:
    json.dump(predictions, f)

