import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn

import torch.optim as optim
from datetime import datetime

# from EMOTI_AUDIO import EMOTI

import math

import os
import visdom
vis = visdom.Visdom()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

batchsize = 1
weight__decay = 1e-4
featurename = '_Mels'
modelname = '29_13h_16m_resnet50_rhf_r45__Mels'
modelnum = ['040','070','080','090','100','150'] #'040','050','060','130','140','060','070','080','090',
#modelnum = ['080']

## RESNET
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


trans = transforms.Compose([
    transforms.Resize((224,int(800*1.25))),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    # transforms.RandomSizedCrop(),
    transforms.ToTensor()])


testset = torchvision.datasets.ImageFolder(root='./_Features%s/Val'%(featurename), transform=trans)

testloader = DataLoader(testset, batch_size=batchsize,shuffle=False)

classes = ('Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise')

#print net



######################################################################################

net = resnet50(pretrained=False)

net.cuda()
cudnn.benchmark = True

# print net


for num in modelnum :
    net.load_state_dict(torch.load('_LOG/%s/%s_%s.pt'%(modelname,modelname,num)))
    print('\n####################################\n')
    print('\n_LOG/%s/%s_%s.pt Loaded\n'%(modelname,modelname,num))
    #net.load_state_dict(torch.load('_LOG/05_18h_19m_densenet169__Mels_RESUMED/05_18h_19m_densenet169__Mels_RESUMED_300.pt'))

    # print(net)
#    net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
    print('Network Done!')

    total = 0
    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))

    net.eval()
    test_loss = 0
    correct = 0

    collect_preds = []

    with torch.no_grad():
        for data in testloader:

            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            output = net(inputs)

            test_loss += F.nll_loss(output, labels, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #            print pred # my predictions

            collect_preds.append(output.cpu().data.numpy())  # SAVE OUPUTS

            correct += pred.eq(labels.view_as(pred)).sum().item()

            # for i in xrange(len(classes)) :
            #     class_correct[i] += pred.eq(labels.view_as(pred)).sum().item()


            ## Each Classes
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            #            print predicted
            c = (predicted == labels.data).squeeze()
            #            print c

            for i in xrange(batchsize):
                label = labels.data[i]
                class_correct[label] += c[i]
                class_total[label] += 1

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    for i in range(7):
        print('Accuracy of {} : {} / {} = {:.4f} %'.format(classes[i],
                                                           class_correct[i], class_total[i],
                                                           100 * class_correct[i].item() /
                                                           class_total[i]))

    prediction = np.vstack(collect_preds)
    #    print prediction
    np.savez(os.path.join('_MODELS/%s_%s_val.npz' % (modelname,num)), prediction)