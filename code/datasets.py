import os
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import utils

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class PartImageDataset(Dataset):
    def __init__(self,
                 part_image_list:list[Image],
                 not_part_image_list:list[Image],
                 transform=transforms.Compose([
                    transforms.Resize((224, 224),antialias=True),
                    transforms.ToTensor()]),
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data_and_labels = None
        self.PART_LABEL = 1
        self.NOT_PART_LABEL = 0
        self._create_data_and_labels(part_image_list,not_part_image_list)
        
#     def __init__(self,
#                  data:list[Image],
#                  labels:list[int],
#                  transform=transforms.Compose([
#                     transforms.Resize((224, 224)),
#                     transforms.ToTensor()]),
#                  target_transform=None):
#         self.transform = transform
#         self.target_transform = target_transform
#         self.data_and_labels = None
#         self.PART_LABEL = 1
#         self.NOT_PART_LABEL = 0
#         self._join_data_and_labels(data,labels)
          
    
    def _create_data_and_labels(self,part_images,not_part_images):
        
        image_and_labels = []
        for part_image in part_images:
            part_image = utils.convert_to_rgb(part_image)
            image_and_labels.append((part_image,self.PART_LABEL))
        for not_part_image in not_part_images:
            not_part_image = utils.convert_to_rgb(not_part_image)
            image_and_labels.append((not_part_image,self.NOT_PART_LABEL))
        
        self.data_and_labels = image_and_labels
        
    def _join_data_and_labels(self,data,labels):
        self.data_and_labels = [(utils.convert_to_rgb(image),label) for image,label in zip(data,labels)] 
            
    
    def __len__(self):
        return len(self.data_and_labels)
    
    def __getitem__(self,idx):
        image = self.data_and_labels[idx][0]
        label = self.data_and_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
class CustomTorchvisionDataset(Dataset):
    def __init__(self,
                 data_dir,
                 data_file_name,
                 select_labels = None,
                 transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224),antialias=True)]),
                 target_transform=lambda x: x[0]):
        
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = None
        self.data_labels = None
        self.select_labels = select_labels
        self.data = None
        self._load_data()
    
    def _load_data(self):
        file = os.path.join(self.data_dir,self.data_file_name)
        with open(file, 'rb') as fo:
            dictionary = pickle.load(fo,encoding='bytes')
        self.data, self.data_labels = self._convert_to_images_and_labels(dictionary)
        self.data, self.data_labels = self._filter_labels()
        self.num_classes = len(np.unique(self.data_labels))
        
    def _filter_labels(self):
        if self.select_labels == None:
            return self.data, self.data_labels
        
        new_images = []
        new_labels = []
        for i in range(0,len(self.data)):
            img = self.data[i]
            fine_label, coarse_label = self.data_labels[i]
            if fine_label in self.select_labels:
                new_images.append(img)
                new_labels.append((fine_label,coarse_label))
        return new_images, new_labels
                
    
    def _convert_to_images_and_labels(self,dictionary):
        data_arrays = dictionary[b'data']
        data_labels = dictionary[b'fine_labels']
        
        images = []
        labels = []
        for i in range(0,len(data_arrays)):
            array = data_arrays[i]
            label = data_labels[i]
            images.append(self._convert_to_image(array))
            fine_label = label
            coarse_label = (int)(label/5)
            labels.append((fine_label,coarse_label))
        return images, labels
            
        
    def _convert_to_image(self,array):
        red = np.array(array[0:1024])
        green = np.array(array[1024:2048])
        blue = np.array(array[2048:])
    
        red = red.reshape(32,32)
        green = green.reshape(32,32)
        blue = blue.reshape(32,32)
    
        combined = np.stack((red,green,blue),axis=-1)
        return combined
    
    def __len__(self):
        return len(self.data_labels)
    
    def __getitem__(self,idx):
        image = self.data[idx]
        label = self.data_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def get_fine_label_name(self,fine_label_num):
        meta_file = os.path.join(self.data_dir,"meta")
        try:
            with open(meta_file,'rb')as fo:
                meta_dictionary = pickle.load(fo,encoding="bytes")
        except FileNotFoundError:
            raise FileNotFoundError(f"The meta file was not found in {self.data_dir}")
            
        fine_label_names = meta_dictionary[b'fine_label_names']
        fine_label = fine_label_names[fine_label_num].decode('utf-8')
        
        return fine_label
    
        
    def get_coarse_label_name(self,coarse_label_num):
        meta_file = os.path.join(self.data_dir,"meta")
        try:
            with open(meta_file,'rb')as fo:
                meta_dictionary = pickle.load(fo,encoding="bytes")
        except FileNotFoundError:
            raise FileNotFoundError(f"The meta file was not found in {self.data_dir}")
            
        coarse_label_names = meta_dictionary[b'coarse_label_names']
        coarse_label = coarse_label_names[coarse_label_num].decode('utf-8')
        
        return coarse_label
    


def get_basic_torchvision_dataset():
    training_data = datasets.CIFAR100(
                            root="data",
                            train=True,
                            download=True,
                            transform=ToTensor())
    
    data_dir = "./data/cifar-100-python/"
    train_dataset = CustomTorchvisionDataset(data_dir,"train",select_labels=[0,1,2,3,4,5])
    return train_dataset
    