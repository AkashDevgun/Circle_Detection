import numpy as np
import torchvision.transforms as transforms
import torch


# Calculate Model Size
def count_parameters_in_mb(model):
    size = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    return size * 4


# Image Transformation
def image_transforms():
    mean = [0.5]
    std = [0.5]

    return transforms.Compose([
        transforms.Normalize(mean, std),
    ])


# Saving the Model
def save(model, model_path):
    torch.save(model.state_dict(), model_path)
