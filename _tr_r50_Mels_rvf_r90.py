
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch.backends.cudnn as cudnn

import torch.optim as optim
from datetime import datetime

# from EMOTI_AUDIO import EMOTI

import os
import visdom
vis = visdom.Visdom()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batchsize = 64
epochnum = 150
weight__decay = 1e-4
featurename = '_Mels'
lr1 = 1e-2
lr2 = 1e-3
lr3 = 1e-4
modelname = 'resnet50_rvf_r90'

trans = transforms.Compose([
    transforms.Resize((224,int(800*1.25))),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    # transforms.RandomSizedCrop(),
    transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(root='./_Features%s/Train_Val_KDH_LHW'%(featurename), transform=trans)
testset = torchvision.datasets.ImageFolder(root='./_Features%s/Test_2017'%(featurename), transform=trans)

trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=True, num_workers=2)

# 7 class
classes = ('Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise')

dataiter = iter(trainloader)
inputs, labels = dataiter.next()

net = torchvision.models.resnet50(pretrained= False)
net.fc = torch.nn.Linear(512*4, 7)
net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# print net

net.cuda()
#net = torch.nn.DataParallel(net, device_ids=[0,1,2,3])
cudnn.benchmark = True

print('Network Done!')

optimizer = optim.SGD(net.parameters(), lr = lr1, momentum = 0.9, weight_decay=weight__decay)
criterion = nn.CrossEntropyLoss()


def EVAL() :

    total = 0
    correct = 0
    for data in trainloader :
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Trained Accuracy : %f'%(100*correct/total))
    f.write('Trained Accuracy : %f\n' %(100*correct/total))

    total = 0
    correct = 0
    for data in testloader :
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy : %f'%(100*correct/total))
    f.write('Test Accuracy : %f\n' %(100*correct/total))

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in testloader:

            data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
            output = net(data)

            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

print('Training Start')

startime = datetime.today().strftime('%d_%Hh_%Mm')
logfilename = '%s_%s_%s'%(startime,modelname,featurename)

for epoch in xrange(epochnum) :  # loop over the dataset multiple times
    # Make Log FILE
    if epoch == 8 :
        if not os.path.exists('_LOG/%s' % (logfilename)):
            os.mkdir('_LOG/%s' % (logfilename))

        f = open('_LOG/%s/log.txt' % (logfilename), 'w')
        f.write('----- lr = %s applied \n' % (lr1))
        f.write('Batch Size : %d \n Weight Decay : %s \n Epoch : %d \n\n' % (batchsize, weight__decay, epochnum))


    # Learning Rates
    if epoch == 50 :
        optimizer = optim.SGD(net.parameters(), lr = lr2, momentum=0.9, weight_decay=weight__decay)
        f.write('----- lr = %s applied \n'%(lr2))
        print('----- lr = %s applied \n'%(lr2))
    elif epoch == 100 :
        optimizer = optim.SGD(net.parameters(), lr = lr3, momentum=0.9, weight_decay=weight__decay)
        f.write('----- lr = %s applied \n'%(lr3))
        print('----- lr = %s applied \n'%(lr3))

    # Model Saver
    if epoch % 10 == 9 :
        # save model
        if epoch > 10 :
            torch.save(net.state_dict(), '_LOG/%s/%s_%03d.pt' % (logfilename, logfilename, epoch + 1))
            print ('%s_%03d%s.pt saved \n' % (logfilename, epoch + 1, featurename))
            f.write('%s_%03d%s.pt saved \n' % (logfilename, epoch + 1, featurename))
            EVAL()

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9 :    # print
            print('[%d, %5d] loss: %.5f' %(epoch + 1, i + 1, running_loss))
            if epoch > 10:
                recordloss = ('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss))
                time = datetime.today().strftime('%Y/%m/%d %H:%M:%S')
                f.write('%s | %s \n' % (time, recordloss))
            vis.images(inputs[:9])
            running_loss = 0.0

print('Finished Training')


f.close()

