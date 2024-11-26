# In: part_list, not_part_list, model
# Out: accuracy, performance metric

import torch.nn as nn
from PIL import Image
import copy
import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import utils

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 1
THRESHOLD = 0.8

def evaluate(model:nn.Module,
             train_list:(list[Image],list[Image]),
             test_list:(list[Image],list[Image]),
             num_epochs:int=NUM_EPOCHS,
             batch_size:int=BATCH_SIZE,
             threshold=THRESHOLD):
    
    all_results = []
    X_train, y_train = train_list
    X_test, y_test = test_list
    test_model = utils.copy_model(model)
    train(test_model,X_train,y_train,num_epochs=num_epochs,batch_size=batch_size)
    result,wrong,right = test(test_model,X_test,y_test)
    
    return result, wrong, right


def train(model:nn.Module,
          part_image_list:list[Image],
          not_part_image_list:list[Image],
          num_epochs:int,
          batch_size:int=BATCH_SIZE) -> nn.Module:
    
    train_dataset = datasets.PartImageDataset(part_image_list=part_image_list,
                                              not_part_image_list=not_part_image_list)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    utils.train_model(model,train_loader,criterion,optimizer,num_epochs=num_epochs)

def test(model:nn.Module,
         part_image_list:list[Image],
         not_part_image_list:list[Image],
         threshold:float=0.8) -> nn.Module:
    
    test_dataset = datasets.PartImageDataset(part_image_list=part_image_list,
                                             not_part_image_list=not_part_image_list)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
    
    result, wrong, right = utils.test_model(model,test_loader,return_images=True,threshold=threshold)
    return result, wrong, right
