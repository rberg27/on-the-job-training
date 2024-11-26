# TODO
    # Implement PartImageDataset - create from list of images
    # Create a main method to run the file from
    # 


train_image_collections = {}
model_collection = {}
from PIL import Image
import torch.nn as nn
import copy

def run_trial(train_images:Image,
              augment_method,
              dataset_size:int,
              model:nn.Module):
    
    init_part_list, init_not_part_list = create_initial_lists(train_images)
    full_part_list, full_not_part_list = augment_data(init_part_list,init_not_part_list,augment_method)
    accuracies = evaluate_model(model,full_part_list,full_not_part_list)
    return accuracies
    
FOLDS = 5
def evaluate_model(model:nn.Module,
                   full_part_list:list[Image],
                   full_not_part_list:list[Image]):
    
    X, y, splits = generate_folds(full_part_list, full_not_part_list)
    folds = extract_X_and_y(X,y,splits)
    accuracies = []
    for train_list, test_list in folds:
        copy_model = copy(model)
        train(copy_model,train_list)#Train
        accuracy = test(copy_model,test_list)#Test
        accuracies.append(accuracy)
    return accuracies

PART_LABEL = 1
NOT_PART_LABEL = 0
from sklearn.model_selection import StratifiedKFold
def generate_folds(full_part_list:list[Image],
                   full_not_part_list:list[Image],
                   folds:int):
    
    X, y = create_X_and_y(part_list,PART_LABEL,not_part_list,NOT_PART_LABEL)
    kf = StratifiedKFold(n_splits=folds,shuffle=True)
    splits = list(kf.split(X,y))
    
    return X, y, splits
    
def extract_X_and_y(X,y,indicies):
    train_idx, test_idx = indicies
    train_list = []
    test_list = []
    for idx in train_idx:
        train_list.append((X[idx],y[idx]))
    for idx in test_idx:
        test_list.append((X[idx],y[idx]))
    return (train_list,test_list)
           
    
def create_X_and_y(part_list,PART_LABEL,not_part_list,NOT_PART_LABEL):
    X = []
    y = []
    
    for img in part_list:
        X.append(img)
        y.append(PART_LABEL)
    for img in not_part_list:
        X.append(img)
        y.append(NOT_PART_LABEL)
 
    return X,y

    
def add_label(image_list,label):
    labeled_list = [(img,label) for img in image_list]
    return labeled_list

def copy(model):
    copy_model = copy.deepcopy(model)
    return copy_model


import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, train_list: list[(Image,int)]):
    train_dataset = PartImageDataset(train_list)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = SimpleCNN(num_classes=10)  # Instantiate your model with the number of classes
    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Choose an optimizer

    
    
# This code is based on the PyTorch tutorial available at https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def train_loop(dataloader, model, loss_fn, optimizer):
    batch_size = 64
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            
            
import sys

def main():
    
    images
    run_trial(
    
if __name__ == "__main__":
    sys.exit(main())
