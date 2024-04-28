import numpy as np

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from utils import hyper_parameters as hp




class orientationClassifier:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = ["Back","Front","Side"]
        self.model = models.resnet50(pretrained=False).to(self.device)
        self.model.fc = nn.Sequential(
                       nn.Linear(2048, 512),
                       nn.ReLU(inplace=True),
                       nn.Linear(512, len(self.classes))).to(self.device)
        self.model.load_state_dict(torch.load(hp.classifierModelWeights,map_location=self.device))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        self.data_transforms = {
            'train':
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'validation':
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize
            ]),
        }
        
        
    def detect(self,images):
        classOutputs = []
        imageBatch = torch.stack([self.data_transforms['validation'](img).to(self.device)
                                for img in images])
        pred_logits_tensor = self.model(imageBatch)
        pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
        for classPred in np.argmax(pred_probs,axis=-1):
            classOutputs.append(self.classes[classPred])
        return classOutputs,np.max(pred_probs,axis=-1)
