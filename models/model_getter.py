import torch
from models.mobilenet_v2 import make as make_model
def get_model_for_dml(name, num_classes):
    return make_model(name, num_classes)