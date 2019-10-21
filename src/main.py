import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optims
from torchsummary import summary
from model import DeepVO
from utils import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = DeepVO()
model.load_pretrained(device)
model.to(device)

optimizer = optims.Adagrad(model.parameters(), lr=0.001)

trian_dl = DataLoader(KittiPredefinedDataset(), batch_size=24, shuffle=True, num_workers=6)
vali_dl  = DataLoader(KittiPredefinedDataset(["01"]), batch_size=24, shuffle=False, num_workers=6)

EPOCH = 10
for e in range(1, EPOCH+1):
    train(model, trian_dl, optimizer, device, e)
    validate(model,vali_dl, optimizer, device, e)