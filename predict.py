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
import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description='Prediction: Predict flower name from an image along with the probability of that name.')
    parser.add_argument('path_to_image', action='store', help='point to image location') 
    parser.add_argument('checkpoint', action='store', help='select trained checkpoint file')
    parser.add_argument('--top_k', dest="top_k", default="3", help='returns the most likely classes, 3 is default')
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json', help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store', default='cpu', help='use GPU for inference as default')
    return parser.parse_args()

def main():
    args = parse_args()
    path_to_image = args.path_to_image
    checkpoint = args.checkpoint
    top_k = int(args.top_k)
    category_names = args.category_names
    model= load_checkpoint(checkpoint)
    #model
    probs, classes = predict(path_to_image, model, top_k)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Converting classes to names
    names = []
    for i in classes:
        names += [cat_to_name[i]]

    print (names)
    print (probs)

def load_checkpoint(file):
    checkpoint= torch.load(file)
    model= getattr(torchvision.models, checkpoint['architecture'])(pretrained=True) #it returns the value of the named attribute of an object
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_label']
#    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.thumbnail((256, 256))
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    im = im.crop((left, top, right, bottom))

    np_image = np.array(im) /255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    image = torch.from_numpy(np_image)  #.type(torch.cuda.FloatTensor)
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Calculate the class probabilities (softmax) for img

    model.eval()
    pImage = process_image(image_path)
    pImage.unsqueeze_(0)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #pImage = pImage.to(device)
    #model = model.to(device) ###############
    model = model.to('cpu')
    
    
    with torch.no_grad():
        output = model.forward(pImage.type(torch.FloatTensor)) #.type(torch.cuda.FloatTensor)

    ps = torch.exp(output)

    probs = ps.topk(topk)[0].numpy()[0]
    classes = ps.topk(topk)[1].numpy()[0]
    #probs, classes = ps.topk(5, dim=1)

    # use a dictionary to convert class integers to names
    class_to_idx= model.class_to_idx
    idx_to_class= {x: y for y, x in class_to_idx.items()}
    list_of_top_classes= []
    for clas in classes:
        list_of_top_classes += [idx_to_class[clas]]

    return probs, list_of_top_classes
if __name__ == "__main__":
    main()

    #python predict.py 'flowers/test/1/image_06743.jpg' checkpoint_densenet121.pth --gpu cpu --top_k 7