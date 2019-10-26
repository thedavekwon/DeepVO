import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optims
from torchsummary import summary
from model import DeepVO
from utils import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

cur, model, optimizer = load_model(device, "../weights/20.weights")

trian_dl = DataLoader(KittiPredefinedDataset(), batch_size=25, shuffle=True, num_workers=8)
vali_dl  = DataLoader(KittiPredefinedValidationDataset(), batch_size=25, shuffle=False, num_workers=8)

EPOCH = 200
for e in range(cur+1, cur+EPOCH+1):
    train(model, trian_dl, optimizer, device, e)
    validate(model,vali_dl, optimizer, device, e)