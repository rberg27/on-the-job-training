import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import datasets
from typing import Callable
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def view_sample_images_from_list(image_list:[Image],num_samples:int=10,is_tensors=False):
    num_images =  len(image_list)
    if num_images == 0: return
    fig, axes = plt.subplots(1, min(num_images,num_samples))
    samples = random.sample(range(num_images),min(num_images,num_samples))
    for i in range(0,len(samples)):
        image_index = samples[i]
        image = image_list[image_index]

        # Convert tensor image to numpy array and transpose for plotting
        if is_tensors:
            image = image.numpy().transpose((1, 2, 0))
            image = np.clip(image, 0, 1)

        # Display image with true and predicted labels
        axes[i].imshow(image)
        axes[i].axis("off")
    plt.axis("off")
    plt.show()
    
def view_sample_labeled_images_from_list(labeled_image_list:[(int,int,Image)],
                                         num_samples:int=10):
    
    num_images =  len(labeled_image_list)
    if num_images == 0: return 0
    fig, axes = plt.subplots(1, min(num_images,num_samples))
    samples = random.sample(range(num_images),min(num_images,num_samples))
    for i in range(0,len(samples)):
        item_index = samples[i]
        item = labeled_image_list[item_index]
        
        prediction = get_prediction(item)
        label = get_label(item)
        image = get_image(item)
        
        # Display image with true and predicted labels
        axes[i].imshow(image)
        axes[i].set_title(f"Pred:{prediction}\n Label:{label}",fontsize=6)
        axes[i].axis("off")
    plt.axis("off")
    plt.show()
    

    
def get_random_image(image_list:[Image]) -> Image:
    return random.choice(image_list)

def get_int_from_ratio(ratio:float,ref:int):
    value = (int)(ratio*ref)
    return value

def crop_annotations_from_image(image,annotations):
    # Open the image
    cropped_images = []
    # Convert JSON bounding box data to coordinates
    for bounding_box in annotations:
        # Define the bounding box coordinates
        x1, y1, x2, y2 = bounding_box["x1"], bounding_box["y1"], bounding_box["x2"], bounding_box["y2"]
        # Crop the image to the adjusted bounding box
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_images.append(cropped_image)
    return cropped_images



def contains(box1:dict,
             box2:dict,
             offset:int = 0):
    test1 = (box1["x1"] - box2["x1"]) <= offset and (box1["y1"] - box2["y1"]) <= offset
    test2 = (box2["x2"] - box1["x2"]) <= offset and (box2["y2"] - box1["y2"]) <= offset
    return test1 and test2

import random
def generate_random_box(image,min_box_dimension,max_box_dimension):
    x1 = random.randint(0,(int)(image.width-min_box_dimension))
    y1 = random.randint(0,(int)(image.height-min_box_dimension))
    
    remaining_width = image.width - x1
    remaining_height = image.height - y1
    
    width_limit = min(remaining_width,max_box_dimension)
    height_limit = min(remaining_height,max_box_dimension)
    
    img_width = random.randint(min_box_dimension,width_limit)
    img_height = random.randint(min_box_dimension,height_limit)
    
    x2 = x1 + img_width
    y2 = y1 + img_width
    
    return [x1, y1, x2, y2]


def crop_image(image,box):
    if box is dict:
        cropped_image = image.crop((box["x1"],box["y1"],box["x2"],box["y2"]))
        return cropped_image
    else:
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        return cropped_image

def convert_bbox_list_to_dict(bbox_list:list) -> dict:
    bbox = {}
    bbox["x1"] = bbox_list[0]
    bbox["y1"] = bbox_list[1]
    bbox["x2"] = bbox_list[2]
    bbox["y2"] = bbox_list[3]
    
    return bbox

def does_crop_contain_annotation(crop,annotations,offset=0):
    crop_dict = crop
    if crop is list:
        crop_dict = convert_bbox_list_to_dict(crop)
    for annotation in annotations:
        if contains(crop_dict,annotation,offset):
            #print(f"OVERLAP: {crop_dict} contains {annotation}")
            return True
    return False

def get_height(bbox_dict):
    height = bbox_dict["y2"] - bbox_dict["y1"]
    return height
    
def get_width(bbox_dict):
    width = bbox_dict["x2"] - bbox_dict["x1"]
    return width

def get_min_and_max_height_and_width(image,annotations):
    min_height = image.height
    min_width = image.width
    max_height = 0
    max_width = 0
    for annotation in annotations:
        h = get_height(annotation)
        w = get_width(annotation)
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w
        if h < min_height:
            min_height = h
        if w < min_width:
            min_width = w
    return min_height, min_width, max_height, max_width


def get_crop_bounds(image,annotations):
    min_h,min_w,max_h,max_w = get_min_and_max_height_and_width(image,annotations)
    min_bound = min(min_h,min_w)
    max_bound = max(max_h,max_w)
    return min_bound, max_bound

def generate_not_part_images(image,annotations,num_data):
    min_bound, max_bound = get_crop_bounds(image,annotations)
    count = 0
    images = []
    while count < num_data:
        crop = generate_random_box(image,min_bound,max_bound)
        crop_dict = convert_bbox_list_to_dict(crop)
        if not does_crop_contain_annotation(crop_dict,annotations,offset=5):
            cropped_image = crop_image(image,crop)
            images.append(cropped_image)
            count = count + 1
    return images

def generate_part_images(image,annotations,num_data):
    min_bound, max_bound = get_crop_bounds(image,annotations)
    count = 0
    images = []
    while count < num_data:
        crop = generate_random_box(image,min_bound,max_bound)
        crop_dict = convert_bbox_list_to_dict(crop)
        if does_crop_contain_annotation(crop_dict,annotations,offset=5):
            cropped_image = crop_image(image,crop)
            images.append(cropped_image)
            count = count + 1
    return images

def generate_random_samples(image:Image,
                            num_data:int,
                            min_bound:int=-1,
                            max_bound:int=-1,
                            scale:int=5) -> list[Image]:
    width,height = image.size
    if min_bound == -1:
        min_bound = (int)(min(width,height)/scale)
    if max_bound == -1:
        max_bound = (int)(max(width,height)/scale)
        
    count = 0
    images = []
    while count < num_data:
        crop = generate_random_box(image,min_bound,max_bound)
        crop_dict = convert_bbox_list_to_dict(crop)
        cropped_image = crop_image(image,crop)
        images.append(cropped_image)
        count = count + 1
    return images

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
def train_model(model:nn.Module,
                train_loader:DataLoader,
                criterion, 
                optimizer, 
                num_epochs=10, 
                device=DEVICE):
    
    model.to(device)  # Move the model to the specified device
    model.train()     # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0

        # Loop over batches in the training DataLoader
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss for this batch
            running_loss += loss.item()
            
            # Print loss every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

    print("Training complete")
    

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

THRESHOLD = 0.8
def test_model(model:nn.Module, 
               test_loader:DataLoader,
               calculate_result:Callable[[list[int],list[int]],...]=accuracy_score,
               device=DEVICE,
               threshold = THRESHOLD,
               return_images=False):
    
    model.to(device)        # Move model to device
    model.eval()            # Set model to evaluation mode
    all_labels = []
    all_predictions = []
    
    incorrect_images = []
    correct_images = []

    with torch.no_grad():   # Disable gradient computation for testing
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = None
            if isinstance(test_loader.dataset,datasets.PartImageDataset):
                predicted = (outputs >= THRESHOLD).float()
            else:
                _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

            # Collect all labels and predictions
            labels = labels.cpu().numpy()
            predictions = predicted.cpu().numpy()
            all_labels.extend(labels)
            all_predictions.extend(predictions)
            
            # Log the incorrect and correct images
            if return_images:
                inputs = convert_images_tensor_to_list(inputs.cpu())
                right, wrong = collect_images(inputs,predictions,labels)
                for item in right:
                    correct_images.append(item)
                for item in wrong:
                    incorrect_images.append(item)

    # Calculate accuracy
    result = calculate_result(all_labels, all_predictions)
    if return_images:
        return result, incorrect_images, correct_images
    else:
        return result

from torchvision.transforms import ToPILImage
def convert_images_tensor_to_list(image_tensor:torch.Tensor) -> list[Image]:
    return [ToPILImage()(img) for img in image_tensor]

def get_labeled_image(prediction:int,
                      label:int,
                      image:Image) -> (int,int,Image):
    return (prediction,label,image)

def get_prediction(labeled_image:(int,int,Image)) -> int:
    return (int)(labeled_image[0])

def get_label(labeled_image:(int,int,Image)) -> int:
    return (int)(labeled_image[1])

def get_image(labeled_image:(int,int,Image)) -> Image:
    return labeled_image[2]

def copy_model(model:nn.Module):
    new_model = copy.deepcopy(model)
    return new_model

def collect_images(inputs:list[Image],
                   predictions:list[int],
                   labels:list[int]) -> (list[(int,int,Image)],list[(int,int,Image)]):
    right = []
    wrong = []
    for i in range(len(labels)):
        label = labels[i]
        prediction = predictions[i]
        image = inputs[i]
        item = get_labeled_image(prediction=prediction,label=label,image=image)
        if label == prediction:
            right.append(item)
        else:
            wrong.append(item)
    return right, wrong

def convert_to_rgb(image: Image) -> Image:
    return image.convert("RGB")

import os
def save_images_to_folder(folder_path:str,
                          new_folder_name:str,
                          image_list:list[Image]):
    new_folder_path = os.path.join(folder_path,new_folder_name)
    os.makedirs(new_folder_path,exist_ok=True)
    for i in range(0,len(image_list)):
        img = image_list[i]
        img_name = f"{i}.jpg"
        image_path = os.path.join(new_folder_path,img_name)
        img = img.convert("RGB")
        img.save(image_path)
    return new_folder_path

def save_test_images_to_folder(new_folder_name:str,
                               image_list:list[Image]):
    
    SAVE_FOLDER = "../images/test_image_sets/"
    return save_images_to_folder(SAVE_FOLDER,new_folder_name,image_list)

def load_images_from_path(folder_path:str)-> list[Image]:
    images = []
    file_list = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for file in file_list:
        image = Image.open(file)
        images.append(image)
    return images

from itertools import combinations
from itertools import permutations
# This method was developed with recommendations from Chat-GPT
def get_all_pairs(full_list:list,
                  order_matters:bool=False):
    
    if order_matters:
        pair_combinations = list(permuntations(full_list,2))
    else:
        pair_combinations = list(combinations(full_list,2))
    return pair_combinations
        
            
                             

