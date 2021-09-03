import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = timm.create_model('tf_efficientnet_b4', pretrained=True)
        self.model.classifier = nn.Linear(1792, 1024)
        self.fc1 = nn.Linear(1024, 3)
        self.fc2 = nn.Linear(1024, 2)
        self.fc3 = nn.Linear(1024, 3)
    
    def forward(self, x):
        fc_output = self.model(x)
        mask = self.fc1(fc_output)
        gender = self.fc2(fc_output)
        age = self.fc3(fc_output)
        
        return mask, gender, age      


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)