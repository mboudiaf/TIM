from sacred import Ingredient
from src.models import __dict__

model_ingredient = Ingredient('model')
@model_ingredient.config
def config():
    arch = 'resnet18'
    num_classes = 64


@model_ingredient.capture
def get_model(arch, num_classes):
    return __dict__[arch](num_classes=num_classes)