# In: Model to copy, Data to train on, epochs, dataset training size
# Out: model, model weights

import torch.nn as nn
import numpy as np
import models
import utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import evaluate_data
import datasets
import torch
import torch.optim as optim

# If on # of data training samples that can be seen: Total/dataset size = epochs
# If on the same number of epochs: epochs

# Q: do I take in a dataset or create a data loader?
BATCH_SIZE = evaluate_data.BATCH_SIZE

def pretrain_model(base_model:nn.Module,
                   dataset:Dataset,
                   model_name:str,
                   epochs:int = -1,
                   total_train_data:int=1000,
                   batch_size:int = BATCH_SIZE) -> nn.Module:
    
    # Copy base model
    model = utils.copy_model(base_model)
    # Train base model
    calculated_epochs = calculate_epochs(epochs=epochs,
                                         total_train_data=total_train_data,
                                         dataset_size=len(dataset))
    
    trained_model = train(model,
                          dataset,
                          epochs = calculated_epochs)
    
    pretrained_model = models.replace_fc_layer(trained_model)
    
    models.save_model(model_name,pretrained_model)
    
    return pretrained_model

LEARNING_RATE = evaluate_data.LEARNING_RATE
def train(model:nn.Module,
          dataset:Dataset,
          epochs:int,
          batch_size:int=BATCH_SIZE):
    
    train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    criterion = nn.BCELoss() if dataset.num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    utils.train_model(model,train_loader,criterion,optimizer,num_epochs=epochs)
    
    return model
    
def calculate_epochs(epochs:int,
                     total_train_data:int,
                     dataset_size:int):
    calculated_epochs = 1
    if epochs <= 0:
        if total_train_data > 0:
            calculated_epochs = np.ceil(total_train_data/dataset_size)
        return calculated_epochs
    else:
        return epochs

import sys   
def main():
    # Create Torchvision Dataset
    dataset = datasets.get_basic_torchvision_dataset()
    # test pretrain model
    model = models.build_model()
    pretrained_model = pretrain_model(base_model=model,
                                      dataset=dataset,
                                      model_name="test-from-pretrain_model-main",
                                      epochs=1)                                    
    
if __name__ == "__main__":
    sys.exit(main())
    
                      
