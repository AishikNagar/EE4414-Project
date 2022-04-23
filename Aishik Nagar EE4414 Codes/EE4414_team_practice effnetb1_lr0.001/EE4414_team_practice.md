EE4414 Team Practice
==============================================

In this team practice, you will design Convolutional Neural Network(s) to classify food images.





```python
%matplotlib inline
```


```python
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.datasets.folder import make_dataset
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import copy


plt.ion()   # interactive mode
```




    <matplotlib.pyplot._IonContext at 0x7f0a9c3a20d0>



## 1. Loading data

Define the dataset, dataloader, and the data augmentation pipeline.

The code below loads 5 classes from all 12 classes in the dataset. You need to modify it to load only the classes that you need.

***Note: For correctly assessing your code, do not change the file structure of the dataset. Use Pytorch data loading utility (`torch.utils.data`) for customizing your dataset.***


```python
# Define the dataset class
class sg_food_dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root, class_id, transform=None):
        self.class_id = class_id
        self.root = root
        all_classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        if not all_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        self.classes = [all_classes[x] for x in class_id]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = make_dataset(self.root, self.class_to_idx, extensions=('jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with open(path, "rb") as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

```


```python
# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        # Define data preparation operations for training set here.
        # Tips: Use torchvision.transforms
        #       https://pytorch.org/vision/stable/transforms.html
        #       Normally this should at least contain resizing (Resize) and data format converting (ToTensor).
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet prior
    ]),
    'val': transforms.Compose([
        # Define data preparation operations for testing/validation set here.
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet prior
    ]),
}

data_dir = os.path.join('./', 'sg_food')
subfolder = {'train': 'train', 'val': 'val'}

# Define the dataset
selected_classes = [3,5,7,8,9]
n_classes = len(selected_classes)
image_datasets = {x: sg_food_dataset(root=os.path.join(data_dir, subfolder[x]),
                                     class_id=selected_classes,
                                     transform=data_transforms[x]) 
                  for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print('selected classes:\n    id: {}\n    name: {}'.format(selected_classes, class_names))

# Define the dataloader
batch_size = 64
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

    selected classes:
        id: [3, 5, 7, 8, 9]
        name: ['Hokkien Prawn Mee', 'Laksa', 'Oyster Omelette', 'Roast Meat Rice', 'Roti Prata']


## 2. Visualizing the dataset
Fetch a batch of training data from the dataset and visualize them. 




```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[:4])

imshow(out, title=[class_names[x] for x in classes[:4]])
```


    
![png](output_7_0.png)
    


## 3. Defining function to train the model

Use a pre-trained CNN model with transfer learning techniques to classify the 5 food categories.

(Note: The provided code is only for reference. You can modify the code whichever way you want.)



```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=24):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    loss_list = []
    acc_list = []
    
    loss_list_val = []
    acc_list_val = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        t1 = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
            
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                ## APPEND LOSS AND ACCURACY
                loss_list.append(epoch_loss)
                acc_list.append(epoch_acc)
            else:
                ## APPEND LOSS AND ACCURACY
                loss_list_val.append(epoch_loss)
                acc_list_val.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        t2=time.time()
        print('Time:'+str(t2-t1))
        print()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,loss_list,acc_list,loss_list_val,acc_list_val
```

## 3.2 Defining Function to Vsualize Model


```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```


```python

```

## 4. Training and validating the model

Train your model for minimum 3 epochs.

### 4.1 Loading pretrained model and defining new classfier layer



```python
# 1. Load the pretrained model and extract the intermediate features.
# Tips:     Use torchvision.models
#           https://pytorch.org/vision/stable/models.html#classification

# (code)

model = models.efficientnet_b1(pretrained=True)

# 2. Modify the pretrain model for your task.


for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, 5)

# # (code)

# # 3. Choose your loss function, optimizer, etc.

criterion = nn.CrossEntropyLoss()

# # Observe that only parameters of final layer are being optimized as
# # opposed to before.
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
# optimizer_conv = optim.Adam(model.classifier.parameters())

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
# (code)
```

### 4.2 Printing and visualizing the modified model


```python
# TODO
print(model)
```

    EfficientNet(
      (features): Sequential(
        (0): ConvNormActivation(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
        (1): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): ConvNormActivation(
                (0): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(16, 4, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(4, 16, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): ConvNormActivation(
                (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.008695652173913044, mode=row)
          )
        )
        (2): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
                (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(96, 4, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(4, 96, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.017391304347826087, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.026086956521739136, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.034782608695652174, mode=row)
          )
        )
        (3): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.043478260869565216, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.05217391304347827, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.06086956521739131, mode=row)
          )
        )
        (4): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
                (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.06956521739130435, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0782608695652174, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.08695652173913043, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.09565217391304348, mode=row)
          )
        )
        (5): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
                (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.10434782608695654, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.11304347826086956, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.12173913043478261, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.13043478260869565, mode=row)
          )
        )
        (6): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1391304347826087, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.14782608695652175, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1565217391304348, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.16521739130434784, mode=row)
          )
          (4): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.17391304347826086, mode=row)
          )
        )
        (7): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
                (1): BatchNorm2d(1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1152, 48, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(48, 1152, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1826086956521739, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(320, 1920, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1920, 1920, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1920, bias=False)
                (1): BatchNorm2d(1920, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1920, 80, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(80, 1920, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1920, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.19130434782608696, mode=row)
          )
        )
        (8): ConvNormActivation(
          (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=1)
      (classifier): Sequential(
        (0): Dropout(p=0.2, inplace=True)
        (1): Linear(in_features=1280, out_features=5, bias=True)
      )
    )



```python
# TODO
from torchsummary import summary
summary(model, input_size=(3,224,224))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 32, 112, 112]             864
           BatchNorm2d-2         [-1, 32, 112, 112]              64
                  SiLU-3         [-1, 32, 112, 112]               0
                Conv2d-4         [-1, 32, 112, 112]             288
           BatchNorm2d-5         [-1, 32, 112, 112]              64
                  SiLU-6         [-1, 32, 112, 112]               0
     AdaptiveAvgPool2d-7             [-1, 32, 1, 1]               0
                Conv2d-8              [-1, 8, 1, 1]             264
                  SiLU-9              [-1, 8, 1, 1]               0
               Conv2d-10             [-1, 32, 1, 1]             288
              Sigmoid-11             [-1, 32, 1, 1]               0
    SqueezeExcitation-12         [-1, 32, 112, 112]               0
               Conv2d-13         [-1, 16, 112, 112]             512
          BatchNorm2d-14         [-1, 16, 112, 112]              32
               MBConv-15         [-1, 16, 112, 112]               0
               Conv2d-16         [-1, 16, 112, 112]             144
          BatchNorm2d-17         [-1, 16, 112, 112]              32
                 SiLU-18         [-1, 16, 112, 112]               0
    AdaptiveAvgPool2d-19             [-1, 16, 1, 1]               0
               Conv2d-20              [-1, 4, 1, 1]              68
                 SiLU-21              [-1, 4, 1, 1]               0
               Conv2d-22             [-1, 16, 1, 1]              80
              Sigmoid-23             [-1, 16, 1, 1]               0
    SqueezeExcitation-24         [-1, 16, 112, 112]               0
               Conv2d-25         [-1, 16, 112, 112]             256
          BatchNorm2d-26         [-1, 16, 112, 112]              32
      StochasticDepth-27         [-1, 16, 112, 112]               0
               MBConv-28         [-1, 16, 112, 112]               0
               Conv2d-29         [-1, 96, 112, 112]           1,536
          BatchNorm2d-30         [-1, 96, 112, 112]             192
                 SiLU-31         [-1, 96, 112, 112]               0
               Conv2d-32           [-1, 96, 56, 56]             864
          BatchNorm2d-33           [-1, 96, 56, 56]             192
                 SiLU-34           [-1, 96, 56, 56]               0
    AdaptiveAvgPool2d-35             [-1, 96, 1, 1]               0
               Conv2d-36              [-1, 4, 1, 1]             388
                 SiLU-37              [-1, 4, 1, 1]               0
               Conv2d-38             [-1, 96, 1, 1]             480
              Sigmoid-39             [-1, 96, 1, 1]               0
    SqueezeExcitation-40           [-1, 96, 56, 56]               0
               Conv2d-41           [-1, 24, 56, 56]           2,304
          BatchNorm2d-42           [-1, 24, 56, 56]              48
               MBConv-43           [-1, 24, 56, 56]               0
               Conv2d-44          [-1, 144, 56, 56]           3,456
          BatchNorm2d-45          [-1, 144, 56, 56]             288
                 SiLU-46          [-1, 144, 56, 56]               0
               Conv2d-47          [-1, 144, 56, 56]           1,296
          BatchNorm2d-48          [-1, 144, 56, 56]             288
                 SiLU-49          [-1, 144, 56, 56]               0
    AdaptiveAvgPool2d-50            [-1, 144, 1, 1]               0
               Conv2d-51              [-1, 6, 1, 1]             870
                 SiLU-52              [-1, 6, 1, 1]               0
               Conv2d-53            [-1, 144, 1, 1]           1,008
              Sigmoid-54            [-1, 144, 1, 1]               0
    SqueezeExcitation-55          [-1, 144, 56, 56]               0
               Conv2d-56           [-1, 24, 56, 56]           3,456
          BatchNorm2d-57           [-1, 24, 56, 56]              48
      StochasticDepth-58           [-1, 24, 56, 56]               0
               MBConv-59           [-1, 24, 56, 56]               0
               Conv2d-60          [-1, 144, 56, 56]           3,456
          BatchNorm2d-61          [-1, 144, 56, 56]             288
                 SiLU-62          [-1, 144, 56, 56]               0
               Conv2d-63          [-1, 144, 56, 56]           1,296
          BatchNorm2d-64          [-1, 144, 56, 56]             288
                 SiLU-65          [-1, 144, 56, 56]               0
    AdaptiveAvgPool2d-66            [-1, 144, 1, 1]               0
               Conv2d-67              [-1, 6, 1, 1]             870
                 SiLU-68              [-1, 6, 1, 1]               0
               Conv2d-69            [-1, 144, 1, 1]           1,008
              Sigmoid-70            [-1, 144, 1, 1]               0
    SqueezeExcitation-71          [-1, 144, 56, 56]               0
               Conv2d-72           [-1, 24, 56, 56]           3,456
          BatchNorm2d-73           [-1, 24, 56, 56]              48
      StochasticDepth-74           [-1, 24, 56, 56]               0
               MBConv-75           [-1, 24, 56, 56]               0
               Conv2d-76          [-1, 144, 56, 56]           3,456
          BatchNorm2d-77          [-1, 144, 56, 56]             288
                 SiLU-78          [-1, 144, 56, 56]               0
               Conv2d-79          [-1, 144, 28, 28]           3,600
          BatchNorm2d-80          [-1, 144, 28, 28]             288
                 SiLU-81          [-1, 144, 28, 28]               0
    AdaptiveAvgPool2d-82            [-1, 144, 1, 1]               0
               Conv2d-83              [-1, 6, 1, 1]             870
                 SiLU-84              [-1, 6, 1, 1]               0
               Conv2d-85            [-1, 144, 1, 1]           1,008
              Sigmoid-86            [-1, 144, 1, 1]               0
    SqueezeExcitation-87          [-1, 144, 28, 28]               0
               Conv2d-88           [-1, 40, 28, 28]           5,760
          BatchNorm2d-89           [-1, 40, 28, 28]              80
               MBConv-90           [-1, 40, 28, 28]               0
               Conv2d-91          [-1, 240, 28, 28]           9,600
          BatchNorm2d-92          [-1, 240, 28, 28]             480
                 SiLU-93          [-1, 240, 28, 28]               0
               Conv2d-94          [-1, 240, 28, 28]           6,000
          BatchNorm2d-95          [-1, 240, 28, 28]             480
                 SiLU-96          [-1, 240, 28, 28]               0
    AdaptiveAvgPool2d-97            [-1, 240, 1, 1]               0
               Conv2d-98             [-1, 10, 1, 1]           2,410
                 SiLU-99             [-1, 10, 1, 1]               0
              Conv2d-100            [-1, 240, 1, 1]           2,640
             Sigmoid-101            [-1, 240, 1, 1]               0
    SqueezeExcitation-102          [-1, 240, 28, 28]               0
              Conv2d-103           [-1, 40, 28, 28]           9,600
         BatchNorm2d-104           [-1, 40, 28, 28]              80
     StochasticDepth-105           [-1, 40, 28, 28]               0
              MBConv-106           [-1, 40, 28, 28]               0
              Conv2d-107          [-1, 240, 28, 28]           9,600
         BatchNorm2d-108          [-1, 240, 28, 28]             480
                SiLU-109          [-1, 240, 28, 28]               0
              Conv2d-110          [-1, 240, 28, 28]           6,000
         BatchNorm2d-111          [-1, 240, 28, 28]             480
                SiLU-112          [-1, 240, 28, 28]               0
    AdaptiveAvgPool2d-113            [-1, 240, 1, 1]               0
              Conv2d-114             [-1, 10, 1, 1]           2,410
                SiLU-115             [-1, 10, 1, 1]               0
              Conv2d-116            [-1, 240, 1, 1]           2,640
             Sigmoid-117            [-1, 240, 1, 1]               0
    SqueezeExcitation-118          [-1, 240, 28, 28]               0
              Conv2d-119           [-1, 40, 28, 28]           9,600
         BatchNorm2d-120           [-1, 40, 28, 28]              80
     StochasticDepth-121           [-1, 40, 28, 28]               0
              MBConv-122           [-1, 40, 28, 28]               0
              Conv2d-123          [-1, 240, 28, 28]           9,600
         BatchNorm2d-124          [-1, 240, 28, 28]             480
                SiLU-125          [-1, 240, 28, 28]               0
              Conv2d-126          [-1, 240, 14, 14]           2,160
         BatchNorm2d-127          [-1, 240, 14, 14]             480
                SiLU-128          [-1, 240, 14, 14]               0
    AdaptiveAvgPool2d-129            [-1, 240, 1, 1]               0
              Conv2d-130             [-1, 10, 1, 1]           2,410
                SiLU-131             [-1, 10, 1, 1]               0
              Conv2d-132            [-1, 240, 1, 1]           2,640
             Sigmoid-133            [-1, 240, 1, 1]               0
    SqueezeExcitation-134          [-1, 240, 14, 14]               0
              Conv2d-135           [-1, 80, 14, 14]          19,200
         BatchNorm2d-136           [-1, 80, 14, 14]             160
              MBConv-137           [-1, 80, 14, 14]               0
              Conv2d-138          [-1, 480, 14, 14]          38,400
         BatchNorm2d-139          [-1, 480, 14, 14]             960
                SiLU-140          [-1, 480, 14, 14]               0
              Conv2d-141          [-1, 480, 14, 14]           4,320
         BatchNorm2d-142          [-1, 480, 14, 14]             960
                SiLU-143          [-1, 480, 14, 14]               0
    AdaptiveAvgPool2d-144            [-1, 480, 1, 1]               0
              Conv2d-145             [-1, 20, 1, 1]           9,620
                SiLU-146             [-1, 20, 1, 1]               0
              Conv2d-147            [-1, 480, 1, 1]          10,080
             Sigmoid-148            [-1, 480, 1, 1]               0
    SqueezeExcitation-149          [-1, 480, 14, 14]               0
              Conv2d-150           [-1, 80, 14, 14]          38,400
         BatchNorm2d-151           [-1, 80, 14, 14]             160
     StochasticDepth-152           [-1, 80, 14, 14]               0
              MBConv-153           [-1, 80, 14, 14]               0
              Conv2d-154          [-1, 480, 14, 14]          38,400
         BatchNorm2d-155          [-1, 480, 14, 14]             960
                SiLU-156          [-1, 480, 14, 14]               0
              Conv2d-157          [-1, 480, 14, 14]           4,320
         BatchNorm2d-158          [-1, 480, 14, 14]             960
                SiLU-159          [-1, 480, 14, 14]               0
    AdaptiveAvgPool2d-160            [-1, 480, 1, 1]               0
              Conv2d-161             [-1, 20, 1, 1]           9,620
                SiLU-162             [-1, 20, 1, 1]               0
              Conv2d-163            [-1, 480, 1, 1]          10,080
             Sigmoid-164            [-1, 480, 1, 1]               0
    SqueezeExcitation-165          [-1, 480, 14, 14]               0
              Conv2d-166           [-1, 80, 14, 14]          38,400
         BatchNorm2d-167           [-1, 80, 14, 14]             160
     StochasticDepth-168           [-1, 80, 14, 14]               0
              MBConv-169           [-1, 80, 14, 14]               0
              Conv2d-170          [-1, 480, 14, 14]          38,400
         BatchNorm2d-171          [-1, 480, 14, 14]             960
                SiLU-172          [-1, 480, 14, 14]               0
              Conv2d-173          [-1, 480, 14, 14]           4,320
         BatchNorm2d-174          [-1, 480, 14, 14]             960
                SiLU-175          [-1, 480, 14, 14]               0
    AdaptiveAvgPool2d-176            [-1, 480, 1, 1]               0
              Conv2d-177             [-1, 20, 1, 1]           9,620
                SiLU-178             [-1, 20, 1, 1]               0
              Conv2d-179            [-1, 480, 1, 1]          10,080
             Sigmoid-180            [-1, 480, 1, 1]               0
    SqueezeExcitation-181          [-1, 480, 14, 14]               0
              Conv2d-182           [-1, 80, 14, 14]          38,400
         BatchNorm2d-183           [-1, 80, 14, 14]             160
     StochasticDepth-184           [-1, 80, 14, 14]               0
              MBConv-185           [-1, 80, 14, 14]               0
              Conv2d-186          [-1, 480, 14, 14]          38,400
         BatchNorm2d-187          [-1, 480, 14, 14]             960
                SiLU-188          [-1, 480, 14, 14]               0
              Conv2d-189          [-1, 480, 14, 14]          12,000
         BatchNorm2d-190          [-1, 480, 14, 14]             960
                SiLU-191          [-1, 480, 14, 14]               0
    AdaptiveAvgPool2d-192            [-1, 480, 1, 1]               0
              Conv2d-193             [-1, 20, 1, 1]           9,620
                SiLU-194             [-1, 20, 1, 1]               0
              Conv2d-195            [-1, 480, 1, 1]          10,080
             Sigmoid-196            [-1, 480, 1, 1]               0
    SqueezeExcitation-197          [-1, 480, 14, 14]               0
              Conv2d-198          [-1, 112, 14, 14]          53,760
         BatchNorm2d-199          [-1, 112, 14, 14]             224
              MBConv-200          [-1, 112, 14, 14]               0
              Conv2d-201          [-1, 672, 14, 14]          75,264
         BatchNorm2d-202          [-1, 672, 14, 14]           1,344
                SiLU-203          [-1, 672, 14, 14]               0
              Conv2d-204          [-1, 672, 14, 14]          16,800
         BatchNorm2d-205          [-1, 672, 14, 14]           1,344
                SiLU-206          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-207            [-1, 672, 1, 1]               0
              Conv2d-208             [-1, 28, 1, 1]          18,844
                SiLU-209             [-1, 28, 1, 1]               0
              Conv2d-210            [-1, 672, 1, 1]          19,488
             Sigmoid-211            [-1, 672, 1, 1]               0
    SqueezeExcitation-212          [-1, 672, 14, 14]               0
              Conv2d-213          [-1, 112, 14, 14]          75,264
         BatchNorm2d-214          [-1, 112, 14, 14]             224
     StochasticDepth-215          [-1, 112, 14, 14]               0
              MBConv-216          [-1, 112, 14, 14]               0
              Conv2d-217          [-1, 672, 14, 14]          75,264
         BatchNorm2d-218          [-1, 672, 14, 14]           1,344
                SiLU-219          [-1, 672, 14, 14]               0
              Conv2d-220          [-1, 672, 14, 14]          16,800
         BatchNorm2d-221          [-1, 672, 14, 14]           1,344
                SiLU-222          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-223            [-1, 672, 1, 1]               0
              Conv2d-224             [-1, 28, 1, 1]          18,844
                SiLU-225             [-1, 28, 1, 1]               0
              Conv2d-226            [-1, 672, 1, 1]          19,488
             Sigmoid-227            [-1, 672, 1, 1]               0
    SqueezeExcitation-228          [-1, 672, 14, 14]               0
              Conv2d-229          [-1, 112, 14, 14]          75,264
         BatchNorm2d-230          [-1, 112, 14, 14]             224
     StochasticDepth-231          [-1, 112, 14, 14]               0
              MBConv-232          [-1, 112, 14, 14]               0
              Conv2d-233          [-1, 672, 14, 14]          75,264
         BatchNorm2d-234          [-1, 672, 14, 14]           1,344
                SiLU-235          [-1, 672, 14, 14]               0
              Conv2d-236          [-1, 672, 14, 14]          16,800
         BatchNorm2d-237          [-1, 672, 14, 14]           1,344
                SiLU-238          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-239            [-1, 672, 1, 1]               0
              Conv2d-240             [-1, 28, 1, 1]          18,844
                SiLU-241             [-1, 28, 1, 1]               0
              Conv2d-242            [-1, 672, 1, 1]          19,488
             Sigmoid-243            [-1, 672, 1, 1]               0
    SqueezeExcitation-244          [-1, 672, 14, 14]               0
              Conv2d-245          [-1, 112, 14, 14]          75,264
         BatchNorm2d-246          [-1, 112, 14, 14]             224
     StochasticDepth-247          [-1, 112, 14, 14]               0
              MBConv-248          [-1, 112, 14, 14]               0
              Conv2d-249          [-1, 672, 14, 14]          75,264
         BatchNorm2d-250          [-1, 672, 14, 14]           1,344
                SiLU-251          [-1, 672, 14, 14]               0
              Conv2d-252            [-1, 672, 7, 7]          16,800
         BatchNorm2d-253            [-1, 672, 7, 7]           1,344
                SiLU-254            [-1, 672, 7, 7]               0
    AdaptiveAvgPool2d-255            [-1, 672, 1, 1]               0
              Conv2d-256             [-1, 28, 1, 1]          18,844
                SiLU-257             [-1, 28, 1, 1]               0
              Conv2d-258            [-1, 672, 1, 1]          19,488
             Sigmoid-259            [-1, 672, 1, 1]               0
    SqueezeExcitation-260            [-1, 672, 7, 7]               0
              Conv2d-261            [-1, 192, 7, 7]         129,024
         BatchNorm2d-262            [-1, 192, 7, 7]             384
              MBConv-263            [-1, 192, 7, 7]               0
              Conv2d-264           [-1, 1152, 7, 7]         221,184
         BatchNorm2d-265           [-1, 1152, 7, 7]           2,304
                SiLU-266           [-1, 1152, 7, 7]               0
              Conv2d-267           [-1, 1152, 7, 7]          28,800
         BatchNorm2d-268           [-1, 1152, 7, 7]           2,304
                SiLU-269           [-1, 1152, 7, 7]               0
    AdaptiveAvgPool2d-270           [-1, 1152, 1, 1]               0
              Conv2d-271             [-1, 48, 1, 1]          55,344
                SiLU-272             [-1, 48, 1, 1]               0
              Conv2d-273           [-1, 1152, 1, 1]          56,448
             Sigmoid-274           [-1, 1152, 1, 1]               0
    SqueezeExcitation-275           [-1, 1152, 7, 7]               0
              Conv2d-276            [-1, 192, 7, 7]         221,184
         BatchNorm2d-277            [-1, 192, 7, 7]             384
     StochasticDepth-278            [-1, 192, 7, 7]               0
              MBConv-279            [-1, 192, 7, 7]               0
              Conv2d-280           [-1, 1152, 7, 7]         221,184
         BatchNorm2d-281           [-1, 1152, 7, 7]           2,304
                SiLU-282           [-1, 1152, 7, 7]               0
              Conv2d-283           [-1, 1152, 7, 7]          28,800
         BatchNorm2d-284           [-1, 1152, 7, 7]           2,304
                SiLU-285           [-1, 1152, 7, 7]               0
    AdaptiveAvgPool2d-286           [-1, 1152, 1, 1]               0
              Conv2d-287             [-1, 48, 1, 1]          55,344
                SiLU-288             [-1, 48, 1, 1]               0
              Conv2d-289           [-1, 1152, 1, 1]          56,448
             Sigmoid-290           [-1, 1152, 1, 1]               0
    SqueezeExcitation-291           [-1, 1152, 7, 7]               0
              Conv2d-292            [-1, 192, 7, 7]         221,184
         BatchNorm2d-293            [-1, 192, 7, 7]             384
     StochasticDepth-294            [-1, 192, 7, 7]               0
              MBConv-295            [-1, 192, 7, 7]               0
              Conv2d-296           [-1, 1152, 7, 7]         221,184
         BatchNorm2d-297           [-1, 1152, 7, 7]           2,304
                SiLU-298           [-1, 1152, 7, 7]               0
              Conv2d-299           [-1, 1152, 7, 7]          28,800
         BatchNorm2d-300           [-1, 1152, 7, 7]           2,304
                SiLU-301           [-1, 1152, 7, 7]               0
    AdaptiveAvgPool2d-302           [-1, 1152, 1, 1]               0
              Conv2d-303             [-1, 48, 1, 1]          55,344
                SiLU-304             [-1, 48, 1, 1]               0
              Conv2d-305           [-1, 1152, 1, 1]          56,448
             Sigmoid-306           [-1, 1152, 1, 1]               0
    SqueezeExcitation-307           [-1, 1152, 7, 7]               0
              Conv2d-308            [-1, 192, 7, 7]         221,184
         BatchNorm2d-309            [-1, 192, 7, 7]             384
     StochasticDepth-310            [-1, 192, 7, 7]               0
              MBConv-311            [-1, 192, 7, 7]               0
              Conv2d-312           [-1, 1152, 7, 7]         221,184
         BatchNorm2d-313           [-1, 1152, 7, 7]           2,304
                SiLU-314           [-1, 1152, 7, 7]               0
              Conv2d-315           [-1, 1152, 7, 7]          28,800
         BatchNorm2d-316           [-1, 1152, 7, 7]           2,304
                SiLU-317           [-1, 1152, 7, 7]               0
    AdaptiveAvgPool2d-318           [-1, 1152, 1, 1]               0
              Conv2d-319             [-1, 48, 1, 1]          55,344
                SiLU-320             [-1, 48, 1, 1]               0
              Conv2d-321           [-1, 1152, 1, 1]          56,448
             Sigmoid-322           [-1, 1152, 1, 1]               0
    SqueezeExcitation-323           [-1, 1152, 7, 7]               0
              Conv2d-324            [-1, 192, 7, 7]         221,184
         BatchNorm2d-325            [-1, 192, 7, 7]             384
     StochasticDepth-326            [-1, 192, 7, 7]               0
              MBConv-327            [-1, 192, 7, 7]               0
              Conv2d-328           [-1, 1152, 7, 7]         221,184
         BatchNorm2d-329           [-1, 1152, 7, 7]           2,304
                SiLU-330           [-1, 1152, 7, 7]               0
              Conv2d-331           [-1, 1152, 7, 7]          10,368
         BatchNorm2d-332           [-1, 1152, 7, 7]           2,304
                SiLU-333           [-1, 1152, 7, 7]               0
    AdaptiveAvgPool2d-334           [-1, 1152, 1, 1]               0
              Conv2d-335             [-1, 48, 1, 1]          55,344
                SiLU-336             [-1, 48, 1, 1]               0
              Conv2d-337           [-1, 1152, 1, 1]          56,448
             Sigmoid-338           [-1, 1152, 1, 1]               0
    SqueezeExcitation-339           [-1, 1152, 7, 7]               0
              Conv2d-340            [-1, 320, 7, 7]         368,640
         BatchNorm2d-341            [-1, 320, 7, 7]             640
              MBConv-342            [-1, 320, 7, 7]               0
              Conv2d-343           [-1, 1920, 7, 7]         614,400
         BatchNorm2d-344           [-1, 1920, 7, 7]           3,840
                SiLU-345           [-1, 1920, 7, 7]               0
              Conv2d-346           [-1, 1920, 7, 7]          17,280
         BatchNorm2d-347           [-1, 1920, 7, 7]           3,840
                SiLU-348           [-1, 1920, 7, 7]               0
    AdaptiveAvgPool2d-349           [-1, 1920, 1, 1]               0
              Conv2d-350             [-1, 80, 1, 1]         153,680
                SiLU-351             [-1, 80, 1, 1]               0
              Conv2d-352           [-1, 1920, 1, 1]         155,520
             Sigmoid-353           [-1, 1920, 1, 1]               0
    SqueezeExcitation-354           [-1, 1920, 7, 7]               0
              Conv2d-355            [-1, 320, 7, 7]         614,400
         BatchNorm2d-356            [-1, 320, 7, 7]             640
     StochasticDepth-357            [-1, 320, 7, 7]               0
              MBConv-358            [-1, 320, 7, 7]               0
              Conv2d-359           [-1, 1280, 7, 7]         409,600
         BatchNorm2d-360           [-1, 1280, 7, 7]           2,560
                SiLU-361           [-1, 1280, 7, 7]               0
    AdaptiveAvgPool2d-362           [-1, 1280, 1, 1]               0
             Dropout-363                 [-1, 1280]               0
              Linear-364                    [-1, 5]           6,405
    ================================================================
    Total params: 6,519,589
    Trainable params: 6,405
    Non-trainable params: 6,513,184
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 245.43
    Params size (MB): 24.87
    Estimated Total Size (MB): 270.88
    ----------------------------------------------------------------


### 4.3 Training using train data and evaluating using validation data

Train your model for minimum 3 epochs.


```python
# TODO 
model, loss_list, acc_list,loss_list_val,acc_list_val = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=10)
```

    Epoch 0/9
    ----------
    train Loss: 1.6091 Acc: 0.2200
    val Loss: 1.5388 Acc: 0.2800
    Time:29.331151008605957
    
    Epoch 1/9
    ----------
    train Loss: 1.5622 Acc: 0.2980
    val Loss: 1.4845 Acc: 0.4733
    Time:31.35258722305298
    
    Epoch 2/9
    ----------
    train Loss: 1.5045 Acc: 0.3900
    val Loss: 1.4290 Acc: 0.5800
    Time:31.710177421569824
    
    Epoch 3/9
    ----------
    train Loss: 1.4441 Acc: 0.4680
    val Loss: 1.3749 Acc: 0.6467
    Time:31.848085641860962
    
    Epoch 4/9
    ----------
    train Loss: 1.3985 Acc: 0.5640
    val Loss: 1.3286 Acc: 0.7067
    Time:31.058417081832886
    
    Epoch 5/9
    ----------
    train Loss: 1.3230 Acc: 0.6140
    val Loss: 1.2852 Acc: 0.6933
    Time:30.749252796173096
    
    Epoch 6/9
    ----------
    train Loss: 1.3007 Acc: 0.6180
    val Loss: 1.2455 Acc: 0.7400
    Time:31.120429515838623
    
    Epoch 7/9
    ----------
    train Loss: 1.2515 Acc: 0.6600
    val Loss: 1.2377 Acc: 0.7533
    Time:31.764952182769775
    
    Epoch 8/9
    ----------
    train Loss: 1.2788 Acc: 0.6220
    val Loss: 1.2361 Acc: 0.7400
    Time:30.777212142944336
    
    Epoch 9/9
    ----------
    train Loss: 1.2683 Acc: 0.6340
    val Loss: 1.2305 Acc: 0.7333
    Time:30.984354734420776
    
    Training complete in 5m 11s
    Best val Acc: 0.753333


## 5. Loading test data

Define the dataset and dataloader for testing.


```python
test_dir = os.path.join('./', 'sg_food', 'test')

# Define the test set.
test_dataset = sg_food_dataset(root=test_dir, class_id=selected_classes, transform=data_transforms['val'])
test_sizes = len(test_dataset)

# Define the dataloader for testing.
test_batch_size = 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=0)
```

## 6. Visualizing the predictions

Predict the label on a few testing samples and visualize the results.


```python
# TODO

# num_images = 4

# (code)

# with torch.no_grad():
    # Predict on the test set

    # (code)

    # Print the output images and labels
    
    # (code)

visualize_model(model)

plt.ioff()
plt.show()

```


    
![png](output_24_0.png)
    



    
![png](output_24_1.png)
    



    
![png](output_24_2.png)
    



    
![png](output_24_3.png)
    



    
![png](output_24_4.png)
    



    
![png](output_24_5.png)
    


## 7. Evaluating on test set

Evaluate your model on the whole test set and compute the accuracy.


```python
model.eval()

test_acc = 0

print('Evaluation')
print('-' * 10)

y_true = []
y_pred = []

wrong_detections = []
correct_detections = []

with torch.no_grad():
    # Iterate over the testing dataset.
    for (inputs, labels) in test_loader:
        inputs = inputs.to(device)
        # Predict on the test set
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
        
        # Confusion Matrix
        
        y_true.extend(preds.numpy())
        y_pred.extend(labels.data.numpy())
        
        test_acc += torch.sum(preds == labels.data)

# Compute the testing accuracy
test_acc = test_acc.double() / test_sizes
print('Testing Acc: {:.4f}'.format(test_acc))

```

    Evaluation
    ----------
    Testing Acc: 0.6897


# Graphing Metrics

### Plotting Loss vs Iteration - Train


```python
iterations = []

for i,loss in enumerate(loss_list):
    iterations.append(i)
```


```python
iterations
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
fig, ax = plt.subplots()
ax.plot(iterations, loss_list)

ax.set(xlabel='Iterations', ylabel='Loss value',
       title='Iteration vs Loss Train')
ax.grid()

plt.show()
```


    
![png](output_31_0.png)
    


### Plotting Loss vs Iteration - Val


```python
iterations = []

for i,loss in enumerate(loss_list_val):
    iterations.append(i)
```


```python
fig, ax = plt.subplots()
ax.plot(iterations, loss_list_val)

ax.set(xlabel='Iterations', ylabel='Loss value',
       title='Iteration vs Loss - Val')
ax.grid()

plt.show()
```


    
![png](output_34_0.png)
    


### Plotting Accuracy vs Iteration - Train


```python
iterations = []

for i,loss in enumerate(acc_list):
    iterations.append(i)
```


```python
fig, ax = plt.subplots()
ax.plot(iterations, acc_list)

ax.set(xlabel='Iterations', ylabel='Accuracy value',
       title='Iteration vs Accuracy - Train')
ax.grid()

plt.show()
```


    
![png](output_37_0.png)
    


### Plotting Accuracy vs Iteration - Val


```python
iterations = []

for i,loss in enumerate(acc_list_val):
    iterations.append(i)
```


```python
fig, ax = plt.subplots()
ax.plot(iterations, acc_list_val)

ax.set(xlabel='Iterations', ylabel='Accuracy value',
       title='Iteration vs Accuracy - Val')
ax.grid()

plt.show()
```


    
![png](output_40_0.png)
    


### Generating Confusion Matrix


```python
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
```


```python
len(y_pred)
```




    999




```python
cf_matrix = confusion_matrix(y_true, y_pred)
```


```python

```


```python
# constant for classes
# classes = ()

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in class_names],
                     columns = [i for i in class_names])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm,fmt='', annot=True)
plt.savefig('effnetb1_lr0.001_output.png')
```


    
![png](output_46_0.png)
    


### Prediction Examples


```python
predictions = pd.DataFrame({'Actual':y_true,'Predicted':y_pred})
```


```python
predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```
