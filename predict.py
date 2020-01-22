import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import PIL
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Checkpoint of the model",
                    type=str)
parser.add_argument("image_loc", help="Location of the image to tested",
                    type=str)
parser.add_argument("--top_k", help="Number of top classes needed",
                    type=int)
parser.add_argument("--gpu", help="select cpu or gpu for prediction",
                    type=str)
parser.add_argument("--category_names", help="select category names",
                    type=str)
args = parser.parse_args() 



checkpoint_load=torch.load(args.checkpoint)
model=models.__dict__[checkpoint_load['architecture']](pretrained=True)
model.classifier=checkpoint_load['model_classifier']
model.load_state_dict(checkpoint_load['model_state_dict'])
class_to_index=checkpoint_load['model_class_to_index']


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model 
    im = Image.open(image)
    ratio=min(im.size)/256
    width=(im.size[0]/ratio)
    height=(im.size[1]/ratio)
    im.thumbnail((width,height),Image.ANTIALIAS)
    left=(im.size[0]-224)/2
    right=left+224
    top=(im.size[1]-224)/2
    bottom=top+224
    im2=im.crop((left, top, right, bottom))
    np_image=np.array(im2)

    np_image_std=np.copy(np_image)
    dim0=((np_image_std[:,:,0]-np.mean(np_image_std[:,:,0]))/np.std(np_image_std[:,:,0])*(.229))+.485
    dim1=((np_image_std[:,:,1]-np.mean(np_image_std[:,:,1]))/np.std(np_image_std[:,:,1])*(.224))+.456
    dim2=((np_image_std[:,:,2]-np.mean(np_image_std[:,:,2]))/np.std(np_image_std[:,:,2])*(.225))+.406
    
    np_image_norm=np.stack((dim0,dim1,dim2),axis=-1)
    np_image_norm_tp=np.transpose(np_image_norm,(2,0,1))
    
    return np_image_norm_tp

#test_image= "/home/workspace/aipnd-project/flowers/test/36/image_04334.jpg"   
test_image=args.image_loc


np_image=process_image(test_image)

image_tensor=torch.from_numpy(np_image)
image_tensor=image_tensor.view((1,3,224,224))
image_tensor=image_tensor.type(torch.float32)

if args.top_k:
    topk=args.top_k
else:
    topk=5
if args.gpu=='gpu':
    device='cuda'
else:
    device='cpu'   
    
model.to(device)
inputs=image_tensor.to(device)
model.eval()
with torch.no_grad():
    logps=model.forward(inputs)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)
    
top_p=top_p.view((topk,)).cpu().tolist()
top_class=top_class.view((topk,)).cpu().tolist()

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)      
    class_to_label={}
    for clas in class_to_index.keys():
        class_to_label[clas]=cat_to_name[class_to_index[clas]]
        
    top_class_index=[class_to_index[i] for i in top_class]
    top_class_name=[class_to_label[i] for i in top_class]
    top_df=pd.DataFrame({'prob':top_p,'index':top_class_index,'label':top_class_name},columns=['label','index','prob'])
else:
    top_class_index=[class_to_index[i] for i in top_class]
    top_df=pd.DataFrame({'prob':top_p,'index':top_class_index},columns=['index','prob'])
   
print(top_df)