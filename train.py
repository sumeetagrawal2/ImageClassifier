# Imports here
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network based on input parameters')
    parser.add_argument('data_dir', action='store', default='flowers', help='training based on data in this folder')
    parser.add_argument('--gpu', action='store', default='gpu', help='uses GPU for training as default')
    parser.add_argument('--save_dir', dest="save_dir", default="checkpoint.pth", help='folder where checkpoint to be saved')
    parser.add_argument('--arch', dest="arch", default="vgg11", choices=['vgg11', 'densenet121'], help='the pre-trained networ to use. vg11 is default. densenet121 is other choice')
    parser.add_argument('--learning_rate', dest="learning_rate", default="0.003", help='The learning rate to use, 0.003 is default')
    parser.add_argument('--hidden_units', dest="hidden_units", default="5000", help='sets default hidden units, 5000 is default')
    parser.add_argument('--epochs', dest='epochs', default='3', help='Number of training iterations. 3 is default')
    return parser.parse_args()

#args = parser.parse_args()

def main(): 
    args = parse_args()
    save_dir = args.save_dir
    learning_rate = float(args.learning_rate)
    arch = args.arch
    hidden_units = int(args.hidden_units)
    if arch=='vgg11':
        input_size =  25088
    else:
        input_size = 1024

    epochs = int(args.epochs)
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = valid_transforms


    # TODO: Load the datasets with ImageFolder
    #image_datasets = 
    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    model = getattr(models, arch)(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('dropout',nn.Dropout(p=0.2)),  
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    #print (model)
    # Use GPU if it's available
    device = torch.device("cuda" if args.gpu=='gpu' else "cpu")
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    #print()
    model.to(device)
    
    criterion = nn.NLLLoss()
    steps = 0
    running_loss = 0
    print_every = 20
    train_losses, validate_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            #inputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                validate_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validate_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validate_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                train_losses.append(running_loss/print_every)
                validate_losses.append(validate_loss/len(validloader))

                model.train()

    checkpoint={
        'architecture': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'classifier': model.classifier,
        'class_label': train_data.class_to_idx,
        'input_size': input_size,
        'output_size': 102,
        'learning_rate': learning_rate,
        'epochs': epochs
                }
    torch.save(checkpoint, save_dir)
    model.to("cpu")   #so it can be loaded from CPU for testing
    
                           
if __name__ == "__main__":
    main()

    #python train.py flowers --arch densenet121 --learning_rate 0.004 --epochs 4 --hidden_units 512 --save_dir checkpoint_densenet121.pth
    #python train.py flowers --arch vgg11 --learning_rate 0.003 --epochs 5 --hidden_units 5000 --save_dir checkpoint_vgg11.pth