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

model = models.efficientnet_b4(pretrained=True)

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
optimizer_conv = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
# optimizer_conv = optim.Adam(model.classifier.parameters())

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
# (code)
```

    Downloading: "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth" to /home/aishik/.cache/torch/hub/checkpoints/efficientnet_b4_rwightman-7eb33cd5.pth



      0%|          | 0.00/74.5M [00:00<?, ?B/s]


### 4.2 Printing and visualizing the modified model


```python
# TODO
print(model)
```

    EfficientNet(
      (features): Sequential(
        (0): ConvNormActivation(
          (0): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
        (1): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
                (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): ConvNormActivation(
                (0): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (2): ConvNormActivation(
                (0): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.00625, mode=row)
          )
        )
        (2): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
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
                (0): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0125, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.018750000000000003, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.025, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.03125, mode=row)
          )
        )
        (3): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(192, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=192, bias=False)
                (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(192, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(192, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.043750000000000004, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.05, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(336, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.05625, mode=row)
          )
        )
        (4): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(56, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(336, 336, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=336, bias=False)
                (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(336, 14, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(336, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.0625, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
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
            (stochastic_depth): StochasticDepth(p=0.06875, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
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
            (stochastic_depth): StochasticDepth(p=0.07500000000000001, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
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
            (stochastic_depth): StochasticDepth(p=0.08125, mode=row)
          )
          (4): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
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
            (stochastic_depth): StochasticDepth(p=0.08750000000000001, mode=row)
          )
          (5): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
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
            (stochastic_depth): StochasticDepth(p=0.09375, mode=row)
          )
        )
        (5): Sequential(
          (0): MBConv(
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
                (0): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.10625000000000001, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1125, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.11875000000000001, mode=row)
          )
          (4): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.125, mode=row)
          )
          (5): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.13125, mode=row)
          )
        )
        (6): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=960, bias=False)
                (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1375, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.14375000000000002, mode=row)
          )
          (2): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.15000000000000002, mode=row)
          )
          (3): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.15625, mode=row)
          )
          (4): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1625, mode=row)
          )
          (5): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.16875, mode=row)
          )
          (6): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.17500000000000002, mode=row)
          )
          (7): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.18125000000000002, mode=row)
          )
        )
        (7): Sequential(
          (0): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(1632, 1632, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1632, bias=False)
                (1): BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(1632, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.1875, mode=row)
          )
          (1): MBConv(
            (block): Sequential(
              (0): ConvNormActivation(
                (0): Conv2d(448, 2688, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(2688, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (1): ConvNormActivation(
                (0): Conv2d(2688, 2688, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=2688, bias=False)
                (1): BatchNorm2d(2688, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (2): SiLU(inplace=True)
              )
              (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(2688, 112, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(112, 2688, kernel_size=(1, 1), stride=(1, 1))
                (activation): SiLU(inplace=True)
                (scale_activation): Sigmoid()
              )
              (3): ConvNormActivation(
                (0): Conv2d(2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(448, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (stochastic_depth): StochasticDepth(p=0.19375, mode=row)
          )
        )
        (8): ConvNormActivation(
          (0): Conv2d(448, 1792, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): SiLU(inplace=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=1)
      (classifier): Sequential(
        (0): Dropout(p=0.4, inplace=True)
        (1): Linear(in_features=1792, out_features=5, bias=True)
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
                Conv2d-1         [-1, 48, 112, 112]           1,296
           BatchNorm2d-2         [-1, 48, 112, 112]              96
                  SiLU-3         [-1, 48, 112, 112]               0
                Conv2d-4         [-1, 48, 112, 112]             432
           BatchNorm2d-5         [-1, 48, 112, 112]              96
                  SiLU-6         [-1, 48, 112, 112]               0
     AdaptiveAvgPool2d-7             [-1, 48, 1, 1]               0
                Conv2d-8             [-1, 12, 1, 1]             588
                  SiLU-9             [-1, 12, 1, 1]               0
               Conv2d-10             [-1, 48, 1, 1]             624
              Sigmoid-11             [-1, 48, 1, 1]               0
    SqueezeExcitation-12         [-1, 48, 112, 112]               0
               Conv2d-13         [-1, 24, 112, 112]           1,152
          BatchNorm2d-14         [-1, 24, 112, 112]              48
               MBConv-15         [-1, 24, 112, 112]               0
               Conv2d-16         [-1, 24, 112, 112]             216
          BatchNorm2d-17         [-1, 24, 112, 112]              48
                 SiLU-18         [-1, 24, 112, 112]               0
    AdaptiveAvgPool2d-19             [-1, 24, 1, 1]               0
               Conv2d-20              [-1, 6, 1, 1]             150
                 SiLU-21              [-1, 6, 1, 1]               0
               Conv2d-22             [-1, 24, 1, 1]             168
              Sigmoid-23             [-1, 24, 1, 1]               0
    SqueezeExcitation-24         [-1, 24, 112, 112]               0
               Conv2d-25         [-1, 24, 112, 112]             576
          BatchNorm2d-26         [-1, 24, 112, 112]              48
      StochasticDepth-27         [-1, 24, 112, 112]               0
               MBConv-28         [-1, 24, 112, 112]               0
               Conv2d-29        [-1, 144, 112, 112]           3,456
          BatchNorm2d-30        [-1, 144, 112, 112]             288
                 SiLU-31        [-1, 144, 112, 112]               0
               Conv2d-32          [-1, 144, 56, 56]           1,296
          BatchNorm2d-33          [-1, 144, 56, 56]             288
                 SiLU-34          [-1, 144, 56, 56]               0
    AdaptiveAvgPool2d-35            [-1, 144, 1, 1]               0
               Conv2d-36              [-1, 6, 1, 1]             870
                 SiLU-37              [-1, 6, 1, 1]               0
               Conv2d-38            [-1, 144, 1, 1]           1,008
              Sigmoid-39            [-1, 144, 1, 1]               0
    SqueezeExcitation-40          [-1, 144, 56, 56]               0
               Conv2d-41           [-1, 32, 56, 56]           4,608
          BatchNorm2d-42           [-1, 32, 56, 56]              64
               MBConv-43           [-1, 32, 56, 56]               0
               Conv2d-44          [-1, 192, 56, 56]           6,144
          BatchNorm2d-45          [-1, 192, 56, 56]             384
                 SiLU-46          [-1, 192, 56, 56]               0
               Conv2d-47          [-1, 192, 56, 56]           1,728
          BatchNorm2d-48          [-1, 192, 56, 56]             384
                 SiLU-49          [-1, 192, 56, 56]               0
    AdaptiveAvgPool2d-50            [-1, 192, 1, 1]               0
               Conv2d-51              [-1, 8, 1, 1]           1,544
                 SiLU-52              [-1, 8, 1, 1]               0
               Conv2d-53            [-1, 192, 1, 1]           1,728
              Sigmoid-54            [-1, 192, 1, 1]               0
    SqueezeExcitation-55          [-1, 192, 56, 56]               0
               Conv2d-56           [-1, 32, 56, 56]           6,144
          BatchNorm2d-57           [-1, 32, 56, 56]              64
      StochasticDepth-58           [-1, 32, 56, 56]               0
               MBConv-59           [-1, 32, 56, 56]               0
               Conv2d-60          [-1, 192, 56, 56]           6,144
          BatchNorm2d-61          [-1, 192, 56, 56]             384
                 SiLU-62          [-1, 192, 56, 56]               0
               Conv2d-63          [-1, 192, 56, 56]           1,728
          BatchNorm2d-64          [-1, 192, 56, 56]             384
                 SiLU-65          [-1, 192, 56, 56]               0
    AdaptiveAvgPool2d-66            [-1, 192, 1, 1]               0
               Conv2d-67              [-1, 8, 1, 1]           1,544
                 SiLU-68              [-1, 8, 1, 1]               0
               Conv2d-69            [-1, 192, 1, 1]           1,728
              Sigmoid-70            [-1, 192, 1, 1]               0
    SqueezeExcitation-71          [-1, 192, 56, 56]               0
               Conv2d-72           [-1, 32, 56, 56]           6,144
          BatchNorm2d-73           [-1, 32, 56, 56]              64
      StochasticDepth-74           [-1, 32, 56, 56]               0
               MBConv-75           [-1, 32, 56, 56]               0
               Conv2d-76          [-1, 192, 56, 56]           6,144
          BatchNorm2d-77          [-1, 192, 56, 56]             384
                 SiLU-78          [-1, 192, 56, 56]               0
               Conv2d-79          [-1, 192, 56, 56]           1,728
          BatchNorm2d-80          [-1, 192, 56, 56]             384
                 SiLU-81          [-1, 192, 56, 56]               0
    AdaptiveAvgPool2d-82            [-1, 192, 1, 1]               0
               Conv2d-83              [-1, 8, 1, 1]           1,544
                 SiLU-84              [-1, 8, 1, 1]               0
               Conv2d-85            [-1, 192, 1, 1]           1,728
              Sigmoid-86            [-1, 192, 1, 1]               0
    SqueezeExcitation-87          [-1, 192, 56, 56]               0
               Conv2d-88           [-1, 32, 56, 56]           6,144
          BatchNorm2d-89           [-1, 32, 56, 56]              64
      StochasticDepth-90           [-1, 32, 56, 56]               0
               MBConv-91           [-1, 32, 56, 56]               0
               Conv2d-92          [-1, 192, 56, 56]           6,144
          BatchNorm2d-93          [-1, 192, 56, 56]             384
                 SiLU-94          [-1, 192, 56, 56]               0
               Conv2d-95          [-1, 192, 28, 28]           4,800
          BatchNorm2d-96          [-1, 192, 28, 28]             384
                 SiLU-97          [-1, 192, 28, 28]               0
    AdaptiveAvgPool2d-98            [-1, 192, 1, 1]               0
               Conv2d-99              [-1, 8, 1, 1]           1,544
                SiLU-100              [-1, 8, 1, 1]               0
              Conv2d-101            [-1, 192, 1, 1]           1,728
             Sigmoid-102            [-1, 192, 1, 1]               0
    SqueezeExcitation-103          [-1, 192, 28, 28]               0
              Conv2d-104           [-1, 56, 28, 28]          10,752
         BatchNorm2d-105           [-1, 56, 28, 28]             112
              MBConv-106           [-1, 56, 28, 28]               0
              Conv2d-107          [-1, 336, 28, 28]          18,816
         BatchNorm2d-108          [-1, 336, 28, 28]             672
                SiLU-109          [-1, 336, 28, 28]               0
              Conv2d-110          [-1, 336, 28, 28]           8,400
         BatchNorm2d-111          [-1, 336, 28, 28]             672
                SiLU-112          [-1, 336, 28, 28]               0
    AdaptiveAvgPool2d-113            [-1, 336, 1, 1]               0
              Conv2d-114             [-1, 14, 1, 1]           4,718
                SiLU-115             [-1, 14, 1, 1]               0
              Conv2d-116            [-1, 336, 1, 1]           5,040
             Sigmoid-117            [-1, 336, 1, 1]               0
    SqueezeExcitation-118          [-1, 336, 28, 28]               0
              Conv2d-119           [-1, 56, 28, 28]          18,816
         BatchNorm2d-120           [-1, 56, 28, 28]             112
     StochasticDepth-121           [-1, 56, 28, 28]               0
              MBConv-122           [-1, 56, 28, 28]               0
              Conv2d-123          [-1, 336, 28, 28]          18,816
         BatchNorm2d-124          [-1, 336, 28, 28]             672
                SiLU-125          [-1, 336, 28, 28]               0
              Conv2d-126          [-1, 336, 28, 28]           8,400
         BatchNorm2d-127          [-1, 336, 28, 28]             672
                SiLU-128          [-1, 336, 28, 28]               0
    AdaptiveAvgPool2d-129            [-1, 336, 1, 1]               0
              Conv2d-130             [-1, 14, 1, 1]           4,718
                SiLU-131             [-1, 14, 1, 1]               0
              Conv2d-132            [-1, 336, 1, 1]           5,040
             Sigmoid-133            [-1, 336, 1, 1]               0
    SqueezeExcitation-134          [-1, 336, 28, 28]               0
              Conv2d-135           [-1, 56, 28, 28]          18,816
         BatchNorm2d-136           [-1, 56, 28, 28]             112
     StochasticDepth-137           [-1, 56, 28, 28]               0
              MBConv-138           [-1, 56, 28, 28]               0
              Conv2d-139          [-1, 336, 28, 28]          18,816
         BatchNorm2d-140          [-1, 336, 28, 28]             672
                SiLU-141          [-1, 336, 28, 28]               0
              Conv2d-142          [-1, 336, 28, 28]           8,400
         BatchNorm2d-143          [-1, 336, 28, 28]             672
                SiLU-144          [-1, 336, 28, 28]               0
    AdaptiveAvgPool2d-145            [-1, 336, 1, 1]               0
              Conv2d-146             [-1, 14, 1, 1]           4,718
                SiLU-147             [-1, 14, 1, 1]               0
              Conv2d-148            [-1, 336, 1, 1]           5,040
             Sigmoid-149            [-1, 336, 1, 1]               0
    SqueezeExcitation-150          [-1, 336, 28, 28]               0
              Conv2d-151           [-1, 56, 28, 28]          18,816
         BatchNorm2d-152           [-1, 56, 28, 28]             112
     StochasticDepth-153           [-1, 56, 28, 28]               0
              MBConv-154           [-1, 56, 28, 28]               0
              Conv2d-155          [-1, 336, 28, 28]          18,816
         BatchNorm2d-156          [-1, 336, 28, 28]             672
                SiLU-157          [-1, 336, 28, 28]               0
              Conv2d-158          [-1, 336, 14, 14]           3,024
         BatchNorm2d-159          [-1, 336, 14, 14]             672
                SiLU-160          [-1, 336, 14, 14]               0
    AdaptiveAvgPool2d-161            [-1, 336, 1, 1]               0
              Conv2d-162             [-1, 14, 1, 1]           4,718
                SiLU-163             [-1, 14, 1, 1]               0
              Conv2d-164            [-1, 336, 1, 1]           5,040
             Sigmoid-165            [-1, 336, 1, 1]               0
    SqueezeExcitation-166          [-1, 336, 14, 14]               0
              Conv2d-167          [-1, 112, 14, 14]          37,632
         BatchNorm2d-168          [-1, 112, 14, 14]             224
              MBConv-169          [-1, 112, 14, 14]               0
              Conv2d-170          [-1, 672, 14, 14]          75,264
         BatchNorm2d-171          [-1, 672, 14, 14]           1,344
                SiLU-172          [-1, 672, 14, 14]               0
              Conv2d-173          [-1, 672, 14, 14]           6,048
         BatchNorm2d-174          [-1, 672, 14, 14]           1,344
                SiLU-175          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-176            [-1, 672, 1, 1]               0
              Conv2d-177             [-1, 28, 1, 1]          18,844
                SiLU-178             [-1, 28, 1, 1]               0
              Conv2d-179            [-1, 672, 1, 1]          19,488
             Sigmoid-180            [-1, 672, 1, 1]               0
    SqueezeExcitation-181          [-1, 672, 14, 14]               0
              Conv2d-182          [-1, 112, 14, 14]          75,264
         BatchNorm2d-183          [-1, 112, 14, 14]             224
     StochasticDepth-184          [-1, 112, 14, 14]               0
              MBConv-185          [-1, 112, 14, 14]               0
              Conv2d-186          [-1, 672, 14, 14]          75,264
         BatchNorm2d-187          [-1, 672, 14, 14]           1,344
                SiLU-188          [-1, 672, 14, 14]               0
              Conv2d-189          [-1, 672, 14, 14]           6,048
         BatchNorm2d-190          [-1, 672, 14, 14]           1,344
                SiLU-191          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-192            [-1, 672, 1, 1]               0
              Conv2d-193             [-1, 28, 1, 1]          18,844
                SiLU-194             [-1, 28, 1, 1]               0
              Conv2d-195            [-1, 672, 1, 1]          19,488
             Sigmoid-196            [-1, 672, 1, 1]               0
    SqueezeExcitation-197          [-1, 672, 14, 14]               0
              Conv2d-198          [-1, 112, 14, 14]          75,264
         BatchNorm2d-199          [-1, 112, 14, 14]             224
     StochasticDepth-200          [-1, 112, 14, 14]               0
              MBConv-201          [-1, 112, 14, 14]               0
              Conv2d-202          [-1, 672, 14, 14]          75,264
         BatchNorm2d-203          [-1, 672, 14, 14]           1,344
                SiLU-204          [-1, 672, 14, 14]               0
              Conv2d-205          [-1, 672, 14, 14]           6,048
         BatchNorm2d-206          [-1, 672, 14, 14]           1,344
                SiLU-207          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-208            [-1, 672, 1, 1]               0
              Conv2d-209             [-1, 28, 1, 1]          18,844
                SiLU-210             [-1, 28, 1, 1]               0
              Conv2d-211            [-1, 672, 1, 1]          19,488
             Sigmoid-212            [-1, 672, 1, 1]               0
    SqueezeExcitation-213          [-1, 672, 14, 14]               0
              Conv2d-214          [-1, 112, 14, 14]          75,264
         BatchNorm2d-215          [-1, 112, 14, 14]             224
     StochasticDepth-216          [-1, 112, 14, 14]               0
              MBConv-217          [-1, 112, 14, 14]               0
              Conv2d-218          [-1, 672, 14, 14]          75,264
         BatchNorm2d-219          [-1, 672, 14, 14]           1,344
                SiLU-220          [-1, 672, 14, 14]               0
              Conv2d-221          [-1, 672, 14, 14]           6,048
         BatchNorm2d-222          [-1, 672, 14, 14]           1,344
                SiLU-223          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-224            [-1, 672, 1, 1]               0
              Conv2d-225             [-1, 28, 1, 1]          18,844
                SiLU-226             [-1, 28, 1, 1]               0
              Conv2d-227            [-1, 672, 1, 1]          19,488
             Sigmoid-228            [-1, 672, 1, 1]               0
    SqueezeExcitation-229          [-1, 672, 14, 14]               0
              Conv2d-230          [-1, 112, 14, 14]          75,264
         BatchNorm2d-231          [-1, 112, 14, 14]             224
     StochasticDepth-232          [-1, 112, 14, 14]               0
              MBConv-233          [-1, 112, 14, 14]               0
              Conv2d-234          [-1, 672, 14, 14]          75,264
         BatchNorm2d-235          [-1, 672, 14, 14]           1,344
                SiLU-236          [-1, 672, 14, 14]               0
              Conv2d-237          [-1, 672, 14, 14]           6,048
         BatchNorm2d-238          [-1, 672, 14, 14]           1,344
                SiLU-239          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-240            [-1, 672, 1, 1]               0
              Conv2d-241             [-1, 28, 1, 1]          18,844
                SiLU-242             [-1, 28, 1, 1]               0
              Conv2d-243            [-1, 672, 1, 1]          19,488
             Sigmoid-244            [-1, 672, 1, 1]               0
    SqueezeExcitation-245          [-1, 672, 14, 14]               0
              Conv2d-246          [-1, 112, 14, 14]          75,264
         BatchNorm2d-247          [-1, 112, 14, 14]             224
     StochasticDepth-248          [-1, 112, 14, 14]               0
              MBConv-249          [-1, 112, 14, 14]               0
              Conv2d-250          [-1, 672, 14, 14]          75,264
         BatchNorm2d-251          [-1, 672, 14, 14]           1,344
                SiLU-252          [-1, 672, 14, 14]               0
              Conv2d-253          [-1, 672, 14, 14]          16,800
         BatchNorm2d-254          [-1, 672, 14, 14]           1,344
                SiLU-255          [-1, 672, 14, 14]               0
    AdaptiveAvgPool2d-256            [-1, 672, 1, 1]               0
              Conv2d-257             [-1, 28, 1, 1]          18,844
                SiLU-258             [-1, 28, 1, 1]               0
              Conv2d-259            [-1, 672, 1, 1]          19,488
             Sigmoid-260            [-1, 672, 1, 1]               0
    SqueezeExcitation-261          [-1, 672, 14, 14]               0
              Conv2d-262          [-1, 160, 14, 14]         107,520
         BatchNorm2d-263          [-1, 160, 14, 14]             320
              MBConv-264          [-1, 160, 14, 14]               0
              Conv2d-265          [-1, 960, 14, 14]         153,600
         BatchNorm2d-266          [-1, 960, 14, 14]           1,920
                SiLU-267          [-1, 960, 14, 14]               0
              Conv2d-268          [-1, 960, 14, 14]          24,000
         BatchNorm2d-269          [-1, 960, 14, 14]           1,920
                SiLU-270          [-1, 960, 14, 14]               0
    AdaptiveAvgPool2d-271            [-1, 960, 1, 1]               0
              Conv2d-272             [-1, 40, 1, 1]          38,440
                SiLU-273             [-1, 40, 1, 1]               0
              Conv2d-274            [-1, 960, 1, 1]          39,360
             Sigmoid-275            [-1, 960, 1, 1]               0
    SqueezeExcitation-276          [-1, 960, 14, 14]               0
              Conv2d-277          [-1, 160, 14, 14]         153,600
         BatchNorm2d-278          [-1, 160, 14, 14]             320
     StochasticDepth-279          [-1, 160, 14, 14]               0
              MBConv-280          [-1, 160, 14, 14]               0
              Conv2d-281          [-1, 960, 14, 14]         153,600
         BatchNorm2d-282          [-1, 960, 14, 14]           1,920
                SiLU-283          [-1, 960, 14, 14]               0
              Conv2d-284          [-1, 960, 14, 14]          24,000
         BatchNorm2d-285          [-1, 960, 14, 14]           1,920
                SiLU-286          [-1, 960, 14, 14]               0
    AdaptiveAvgPool2d-287            [-1, 960, 1, 1]               0
              Conv2d-288             [-1, 40, 1, 1]          38,440
                SiLU-289             [-1, 40, 1, 1]               0
              Conv2d-290            [-1, 960, 1, 1]          39,360
             Sigmoid-291            [-1, 960, 1, 1]               0
    SqueezeExcitation-292          [-1, 960, 14, 14]               0
              Conv2d-293          [-1, 160, 14, 14]         153,600
         BatchNorm2d-294          [-1, 160, 14, 14]             320
     StochasticDepth-295          [-1, 160, 14, 14]               0
              MBConv-296          [-1, 160, 14, 14]               0
              Conv2d-297          [-1, 960, 14, 14]         153,600
         BatchNorm2d-298          [-1, 960, 14, 14]           1,920
                SiLU-299          [-1, 960, 14, 14]               0
              Conv2d-300          [-1, 960, 14, 14]          24,000
         BatchNorm2d-301          [-1, 960, 14, 14]           1,920
                SiLU-302          [-1, 960, 14, 14]               0
    AdaptiveAvgPool2d-303            [-1, 960, 1, 1]               0
              Conv2d-304             [-1, 40, 1, 1]          38,440
                SiLU-305             [-1, 40, 1, 1]               0
              Conv2d-306            [-1, 960, 1, 1]          39,360
             Sigmoid-307            [-1, 960, 1, 1]               0
    SqueezeExcitation-308          [-1, 960, 14, 14]               0
              Conv2d-309          [-1, 160, 14, 14]         153,600
         BatchNorm2d-310          [-1, 160, 14, 14]             320
     StochasticDepth-311          [-1, 160, 14, 14]               0
              MBConv-312          [-1, 160, 14, 14]               0
              Conv2d-313          [-1, 960, 14, 14]         153,600
         BatchNorm2d-314          [-1, 960, 14, 14]           1,920
                SiLU-315          [-1, 960, 14, 14]               0
              Conv2d-316          [-1, 960, 14, 14]          24,000
         BatchNorm2d-317          [-1, 960, 14, 14]           1,920
                SiLU-318          [-1, 960, 14, 14]               0
    AdaptiveAvgPool2d-319            [-1, 960, 1, 1]               0
              Conv2d-320             [-1, 40, 1, 1]          38,440
                SiLU-321             [-1, 40, 1, 1]               0
              Conv2d-322            [-1, 960, 1, 1]          39,360
             Sigmoid-323            [-1, 960, 1, 1]               0
    SqueezeExcitation-324          [-1, 960, 14, 14]               0
              Conv2d-325          [-1, 160, 14, 14]         153,600
         BatchNorm2d-326          [-1, 160, 14, 14]             320
     StochasticDepth-327          [-1, 160, 14, 14]               0
              MBConv-328          [-1, 160, 14, 14]               0
              Conv2d-329          [-1, 960, 14, 14]         153,600
         BatchNorm2d-330          [-1, 960, 14, 14]           1,920
                SiLU-331          [-1, 960, 14, 14]               0
              Conv2d-332          [-1, 960, 14, 14]          24,000
         BatchNorm2d-333          [-1, 960, 14, 14]           1,920
                SiLU-334          [-1, 960, 14, 14]               0
    AdaptiveAvgPool2d-335            [-1, 960, 1, 1]               0
              Conv2d-336             [-1, 40, 1, 1]          38,440
                SiLU-337             [-1, 40, 1, 1]               0
              Conv2d-338            [-1, 960, 1, 1]          39,360
             Sigmoid-339            [-1, 960, 1, 1]               0
    SqueezeExcitation-340          [-1, 960, 14, 14]               0
              Conv2d-341          [-1, 160, 14, 14]         153,600
         BatchNorm2d-342          [-1, 160, 14, 14]             320
     StochasticDepth-343          [-1, 160, 14, 14]               0
              MBConv-344          [-1, 160, 14, 14]               0
              Conv2d-345          [-1, 960, 14, 14]         153,600
         BatchNorm2d-346          [-1, 960, 14, 14]           1,920
                SiLU-347          [-1, 960, 14, 14]               0
              Conv2d-348            [-1, 960, 7, 7]          24,000
         BatchNorm2d-349            [-1, 960, 7, 7]           1,920
                SiLU-350            [-1, 960, 7, 7]               0
    AdaptiveAvgPool2d-351            [-1, 960, 1, 1]               0
              Conv2d-352             [-1, 40, 1, 1]          38,440
                SiLU-353             [-1, 40, 1, 1]               0
              Conv2d-354            [-1, 960, 1, 1]          39,360
             Sigmoid-355            [-1, 960, 1, 1]               0
    SqueezeExcitation-356            [-1, 960, 7, 7]               0
              Conv2d-357            [-1, 272, 7, 7]         261,120
         BatchNorm2d-358            [-1, 272, 7, 7]             544
              MBConv-359            [-1, 272, 7, 7]               0
              Conv2d-360           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-361           [-1, 1632, 7, 7]           3,264
                SiLU-362           [-1, 1632, 7, 7]               0
              Conv2d-363           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-364           [-1, 1632, 7, 7]           3,264
                SiLU-365           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-366           [-1, 1632, 1, 1]               0
              Conv2d-367             [-1, 68, 1, 1]         111,044
                SiLU-368             [-1, 68, 1, 1]               0
              Conv2d-369           [-1, 1632, 1, 1]         112,608
             Sigmoid-370           [-1, 1632, 1, 1]               0
    SqueezeExcitation-371           [-1, 1632, 7, 7]               0
              Conv2d-372            [-1, 272, 7, 7]         443,904
         BatchNorm2d-373            [-1, 272, 7, 7]             544
     StochasticDepth-374            [-1, 272, 7, 7]               0
              MBConv-375            [-1, 272, 7, 7]               0
              Conv2d-376           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-377           [-1, 1632, 7, 7]           3,264
                SiLU-378           [-1, 1632, 7, 7]               0
              Conv2d-379           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-380           [-1, 1632, 7, 7]           3,264
                SiLU-381           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-382           [-1, 1632, 1, 1]               0
              Conv2d-383             [-1, 68, 1, 1]         111,044
                SiLU-384             [-1, 68, 1, 1]               0
              Conv2d-385           [-1, 1632, 1, 1]         112,608
             Sigmoid-386           [-1, 1632, 1, 1]               0
    SqueezeExcitation-387           [-1, 1632, 7, 7]               0
              Conv2d-388            [-1, 272, 7, 7]         443,904
         BatchNorm2d-389            [-1, 272, 7, 7]             544
     StochasticDepth-390            [-1, 272, 7, 7]               0
              MBConv-391            [-1, 272, 7, 7]               0
              Conv2d-392           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-393           [-1, 1632, 7, 7]           3,264
                SiLU-394           [-1, 1632, 7, 7]               0
              Conv2d-395           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-396           [-1, 1632, 7, 7]           3,264
                SiLU-397           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-398           [-1, 1632, 1, 1]               0
              Conv2d-399             [-1, 68, 1, 1]         111,044
                SiLU-400             [-1, 68, 1, 1]               0
              Conv2d-401           [-1, 1632, 1, 1]         112,608
             Sigmoid-402           [-1, 1632, 1, 1]               0
    SqueezeExcitation-403           [-1, 1632, 7, 7]               0
              Conv2d-404            [-1, 272, 7, 7]         443,904
         BatchNorm2d-405            [-1, 272, 7, 7]             544
     StochasticDepth-406            [-1, 272, 7, 7]               0
              MBConv-407            [-1, 272, 7, 7]               0
              Conv2d-408           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-409           [-1, 1632, 7, 7]           3,264
                SiLU-410           [-1, 1632, 7, 7]               0
              Conv2d-411           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-412           [-1, 1632, 7, 7]           3,264
                SiLU-413           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-414           [-1, 1632, 1, 1]               0
              Conv2d-415             [-1, 68, 1, 1]         111,044
                SiLU-416             [-1, 68, 1, 1]               0
              Conv2d-417           [-1, 1632, 1, 1]         112,608
             Sigmoid-418           [-1, 1632, 1, 1]               0
    SqueezeExcitation-419           [-1, 1632, 7, 7]               0
              Conv2d-420            [-1, 272, 7, 7]         443,904
         BatchNorm2d-421            [-1, 272, 7, 7]             544
     StochasticDepth-422            [-1, 272, 7, 7]               0
              MBConv-423            [-1, 272, 7, 7]               0
              Conv2d-424           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-425           [-1, 1632, 7, 7]           3,264
                SiLU-426           [-1, 1632, 7, 7]               0
              Conv2d-427           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-428           [-1, 1632, 7, 7]           3,264
                SiLU-429           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-430           [-1, 1632, 1, 1]               0
              Conv2d-431             [-1, 68, 1, 1]         111,044
                SiLU-432             [-1, 68, 1, 1]               0
              Conv2d-433           [-1, 1632, 1, 1]         112,608
             Sigmoid-434           [-1, 1632, 1, 1]               0
    SqueezeExcitation-435           [-1, 1632, 7, 7]               0
              Conv2d-436            [-1, 272, 7, 7]         443,904
         BatchNorm2d-437            [-1, 272, 7, 7]             544
     StochasticDepth-438            [-1, 272, 7, 7]               0
              MBConv-439            [-1, 272, 7, 7]               0
              Conv2d-440           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-441           [-1, 1632, 7, 7]           3,264
                SiLU-442           [-1, 1632, 7, 7]               0
              Conv2d-443           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-444           [-1, 1632, 7, 7]           3,264
                SiLU-445           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-446           [-1, 1632, 1, 1]               0
              Conv2d-447             [-1, 68, 1, 1]         111,044
                SiLU-448             [-1, 68, 1, 1]               0
              Conv2d-449           [-1, 1632, 1, 1]         112,608
             Sigmoid-450           [-1, 1632, 1, 1]               0
    SqueezeExcitation-451           [-1, 1632, 7, 7]               0
              Conv2d-452            [-1, 272, 7, 7]         443,904
         BatchNorm2d-453            [-1, 272, 7, 7]             544
     StochasticDepth-454            [-1, 272, 7, 7]               0
              MBConv-455            [-1, 272, 7, 7]               0
              Conv2d-456           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-457           [-1, 1632, 7, 7]           3,264
                SiLU-458           [-1, 1632, 7, 7]               0
              Conv2d-459           [-1, 1632, 7, 7]          40,800
         BatchNorm2d-460           [-1, 1632, 7, 7]           3,264
                SiLU-461           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-462           [-1, 1632, 1, 1]               0
              Conv2d-463             [-1, 68, 1, 1]         111,044
                SiLU-464             [-1, 68, 1, 1]               0
              Conv2d-465           [-1, 1632, 1, 1]         112,608
             Sigmoid-466           [-1, 1632, 1, 1]               0
    SqueezeExcitation-467           [-1, 1632, 7, 7]               0
              Conv2d-468            [-1, 272, 7, 7]         443,904
         BatchNorm2d-469            [-1, 272, 7, 7]             544
     StochasticDepth-470            [-1, 272, 7, 7]               0
              MBConv-471            [-1, 272, 7, 7]               0
              Conv2d-472           [-1, 1632, 7, 7]         443,904
         BatchNorm2d-473           [-1, 1632, 7, 7]           3,264
                SiLU-474           [-1, 1632, 7, 7]               0
              Conv2d-475           [-1, 1632, 7, 7]          14,688
         BatchNorm2d-476           [-1, 1632, 7, 7]           3,264
                SiLU-477           [-1, 1632, 7, 7]               0
    AdaptiveAvgPool2d-478           [-1, 1632, 1, 1]               0
              Conv2d-479             [-1, 68, 1, 1]         111,044
                SiLU-480             [-1, 68, 1, 1]               0
              Conv2d-481           [-1, 1632, 1, 1]         112,608
             Sigmoid-482           [-1, 1632, 1, 1]               0
    SqueezeExcitation-483           [-1, 1632, 7, 7]               0
              Conv2d-484            [-1, 448, 7, 7]         731,136
         BatchNorm2d-485            [-1, 448, 7, 7]             896
              MBConv-486            [-1, 448, 7, 7]               0
              Conv2d-487           [-1, 2688, 7, 7]       1,204,224
         BatchNorm2d-488           [-1, 2688, 7, 7]           5,376
                SiLU-489           [-1, 2688, 7, 7]               0
              Conv2d-490           [-1, 2688, 7, 7]          24,192
         BatchNorm2d-491           [-1, 2688, 7, 7]           5,376
                SiLU-492           [-1, 2688, 7, 7]               0
    AdaptiveAvgPool2d-493           [-1, 2688, 1, 1]               0
              Conv2d-494            [-1, 112, 1, 1]         301,168
                SiLU-495            [-1, 112, 1, 1]               0
              Conv2d-496           [-1, 2688, 1, 1]         303,744
             Sigmoid-497           [-1, 2688, 1, 1]               0
    SqueezeExcitation-498           [-1, 2688, 7, 7]               0
              Conv2d-499            [-1, 448, 7, 7]       1,204,224
         BatchNorm2d-500            [-1, 448, 7, 7]             896
     StochasticDepth-501            [-1, 448, 7, 7]               0
              MBConv-502            [-1, 448, 7, 7]               0
              Conv2d-503           [-1, 1792, 7, 7]         802,816
         BatchNorm2d-504           [-1, 1792, 7, 7]           3,584
                SiLU-505           [-1, 1792, 7, 7]               0
    AdaptiveAvgPool2d-506           [-1, 1792, 1, 1]               0
             Dropout-507                 [-1, 1792]               0
              Linear-508                    [-1, 5]           8,965
    ================================================================
    Total params: 17,557,581
    Trainable params: 8,965
    Non-trainable params: 17,548,616
    ----------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 449.91
    Params size (MB): 66.98
    Estimated Total Size (MB): 517.46
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
    train Loss: 1.5981 Acc: 0.2400
    val Loss: 1.5791 Acc: 0.3867
    Time:56.17032241821289
    
    Epoch 1/9
    ----------
    train Loss: 1.5350 Acc: 0.4460
    val Loss: 1.5089 Acc: 0.6333
    Time:59.566983222961426
    
    Epoch 2/9
    ----------
    train Loss: 1.4481 Acc: 0.6080
    val Loss: 1.4176 Acc: 0.7467
    Time:61.92120027542114
    
    Epoch 3/9
    ----------
    train Loss: 1.3761 Acc: 0.6840
    val Loss: 1.3400 Acc: 0.7267
    Time:60.17346239089966
    
    Epoch 4/9
    ----------
    train Loss: 1.3059 Acc: 0.7080
    val Loss: 1.2547 Acc: 0.7733
    Time:61.1920382976532
    
    Epoch 5/9
    ----------
    train Loss: 1.2431 Acc: 0.7300
    val Loss: 1.1849 Acc: 0.8133
    Time:70.23015642166138
    
    Epoch 6/9
    ----------
    train Loss: 1.1810 Acc: 0.7280
    val Loss: 1.1372 Acc: 0.8000
    Time:65.69802761077881
    
    Epoch 7/9
    ----------
    train Loss: 1.1503 Acc: 0.7480
    val Loss: 1.1328 Acc: 0.8067
    Time:60.124581813812256
    
    Epoch 8/9
    ----------
    train Loss: 1.1251 Acc: 0.7740
    val Loss: 1.1221 Acc: 0.8000
    Time:58.424713134765625
    
    Epoch 9/9
    ----------
    train Loss: 1.1405 Acc: 0.7460
    val Loss: 1.1220 Acc: 0.7933
    Time:58.93634796142578
    
    Training complete in 10m 13s
    Best val Acc: 0.813333


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
    Testing Acc: 0.7778


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
plt.savefig('effnetb4_lr0.01_output.png')
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
      <td>1</td>
      <td>1</td>
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
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2</td>
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
