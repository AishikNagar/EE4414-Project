# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.datasets.folder import make_dataset
from PIL import Image
# import matplotlib.pyplot as plt
import time
import os
import copy
# import cv2

import onnxruntime
import onnx


onnx_model = onnx.load("./onnx_deployment.onnx")
onnx.checker.check_model(onnx_model)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def getPrediction(filename):

    ort_session = onnxruntime.InferenceSession("./onnx_deployment.onnx")
    f= './uploads/'+filename
    # Image processing
    img = Image.open(f).convert('RGB')
    transform_img = transforms.Compose([
            # Define data preparation operations for testing/validation set here.
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet prior
        ])
    input_img = transform_img(img)
    input_img = input_img.unsqueeze_(0)
    input_img =input_img.to('cpu')

    # ONNX Inference
    time_start = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_img)}
    ort_outs = ort_session.run(None, ort_inputs)
    inference_time = time.time()-time_start
    output_tensors = torch.FloatTensor(ort_outs[0])
    _, preds = torch.max(output_tensors, 1)
    preds= preds.numpy()

    class_names = ['Hokkien Prawn Mee', 'Laksa', 'Oyster Omelette', 'Roast Meat Rice', 'Roti Prata']

    return class_names[preds[0]],inference_time
