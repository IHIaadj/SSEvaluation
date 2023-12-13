import torchvision.models as models
import torch.nn as nn
from collections import defaultdict
import csv
from utils import *

def extract_specific_blocks(model, block_types):
    block_names = []
    for name, module in model.named_modules():
        # Check if the module is an instance of any of the block types
        if isinstance(module, block_types):
            block_names.append((name, type(module).__name__))
    return block_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Prepare CSV file
with open('model_blocks.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['model_name', 'block_type'])

    for model_name in model_names:
        # Load the model
        print(model_name)
        if "get_" not in model_name and "list_models" not in model_name:
          model = models.__dict__[model_name](pretrained=False)
          
          # Extract specific block names
          layer_sequence = [
              get_layer_type(layer) for _, layer in model.named_modules() 
              if not any(c for c in layer.named_children())  # Include only layers without children
          ]
          blocks, rest = find_pattern(layer_sequence, 7)
          rest= set(rest)
          
          writer.writerow([model_name, blocks])
          for r in rest:
            writer.writerow([model_name, r])

print("CSV file 'model_blocks.csv' has been created.")