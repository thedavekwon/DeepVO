import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optims
from torchsummary import summary
from model import DeepVO
from utils import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = DeepVO()
model.load_pretrained()
model.to(device)

optimizer = optims.Adagrad(model.parameters(), lr=0.001)

dl = DataLoader(KittiOdometryRandomSequenceDataset("00"), batch_size=1, shuffle=True, num_workers=0)

train(model, dl, optimizer, device)

