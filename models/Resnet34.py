import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
import torch.nn.init
import os
os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')
import sys
sys.path.insert(0, '../')

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

transform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)


I = Image.open('49.bmp')
# I.show()
img1 = transform1(I)
I = np.array(I)
# print (I_array.shape)

# torch.load("netD_A_epoch_27.pth")
# a = torch.load('netD_A_epoch_27.pth', map_location=torch.device('cpu'))

resnet34 = models.resnet34(pretrained=True)
resnet34.fc = nn.Linear(512, 2048)

for param in resnet34.parameters():
  param.requires_grad = False
#resnet152 = models.resnet152(pretrained = True)
#densenet201 = models.densenet201(pretrained = True)
  x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
#y1 = resnet18(x)
  y = resnet34(x)
  y = y.data.numpy()
print(y.shape)

class PoseNet(nn.Module):
  def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
      feat_dim=2048, filter_nans=False):
    super(PoseNet, self).__init__()
    self.droprate = droprate

    # replace the last FC layer in feature extractor
    self.feature_extractor = feature_extractor
    self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    fe_out_planes = self.feature_extractor.fc.in_features
    self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 3)
    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, x):
    x = self.feature_extractor(x)
    #x = F.relu(x)
    if self.droprate > 0:
      x = F.dropout(x, p=self.droprate)

    xyz  = self.fc_xyz(x)
    wpqr = self.fc_wpqr(x)
    return torch.cat((xyz, wpqr), 1)