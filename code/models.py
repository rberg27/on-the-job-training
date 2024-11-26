import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        final_output = 1 if num_classes == 2 else num_classes
        self.num_classes = num_classes
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # (32, H, W)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # (64, H, W)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # (128, H, W)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions by half

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=512)  # Flattened feature map size (depends on input image size)
        self.fc2 = nn.Linear(in_features=512, out_features=final_output)

    def forward(self, x):
        # Apply first convolutional block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional block
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 128 * 28 * 28)  # Adjust dimensions based on input image size

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.num_classes == 2:
            x = nn.Flatten(start_dim=0)(x)
            x = nn.Sigmoid()(x)
        return x

def build_model():
    model = SimpleCNN(num_classes=2)
    return model

import os
MODEL_PATH = "../models/"
def save_model(model_name:str,
               model:nn.Module):
    
    model_path = os.path.join(MODEL_PATH,f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    

def load_model(model_name:str) -> nn.Module:
    
    model = build_model()
    model_path = os.path.join(MODEL_PATH,model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def replace_fc_layer(trained_model:nn.Module):
    
    copy_of_model = utils.copy_model(trained_model)
    copy_of_model.fc2 = nn.Linear(in_features=512, out_features=1)
    
    return copy_of_model

