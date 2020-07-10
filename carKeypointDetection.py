# Project KNN, FIT VUT 2019/2020 - Car keypoint detection
# Stacked hourglass model for car kp detection implementation with trainer and dataloader
## Vladislav Halva, Martin Dvořák
## xhalva04, xdvora2l


"""
CUSTUM DATASET CLASS
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import sys
import pandas as pd
from tqdm import tqdm

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
import random


class GenerateHeatmap():
  def __init__(self, outputRes, numParts):
    self.outputRes = outputRes
    self.numParts = numParts
    sigma = 1
    self.sigma = sigma
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

  def __call__(self, keypoints):
    hms = np.zeros(shape=(self.numParts, self.outputRes, self.outputRes), dtype=np.float32)
    sigma = self.sigma

    for p in np.array([keypoints]):
      for idx, pt in enumerate(p):
        if pt[0] > 0:
          x, y = int(pt[0]), int(pt[1])
          if x<0 or y<0 or x>=self.outputRes or y>=self.outputRes:
            continue
          ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
          br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

          c,d = max(0, -ul[0]), min(br[0], self.outputRes) - ul[0]
          a,b = max(0, -ul[1]), min(br[1], self.outputRes) - ul[1]

          cc,dd = max(0, ul[0]), min(br[0], self.outputRes)
          aa,bb = max(0, ul[1]), min(br[1], self.outputRes)
          hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
    
    return hms


class CarKeyPointsDataset(Dataset):
  def __init__(self, csvFile, rootDir):
    """
    csvFile = file with annotations - labels
    rootDir = directory with all the images
    """
    self.imgSize = 64
    self.keypointsFrame = pd.read_csv(csvFile, sep=' ')
    self.rootDir = rootDir
    self.generateHeatmap = GenerateHeatmap(self.imgSize, 20)

  def __len__(self):
    return len(self.keypointsFrame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    imgName = os.path.join(self.rootDir,
                            self.keypointsFrame.iloc[idx, 0])

    try:
      image = io.imread(imgName)
    except OSError as e:
      len = self.__len__()
      idx = random.randrange(0, len)
      return self.__getitem__(idx)

    keypoints = self.keypointsFrame.iloc[idx, 1:]
    keypoints = np.array([keypoints])
    keypoints = keypoints.astype('double').reshape(-1, 2)

    sample = {'image': image, 'keypoints': keypoints}

    rescale = Rescale((self.imgSize,self.imgSize))
    sample = rescale(sample)

    hms = self.generateHeatmap(sample['keypoints'])
    sample['heatmaps'] = hms

    toTensor = ToTensor()
    sample = toTensor(sample)
    return sample

class Rescale(object):
  def __init__(self, outputSize):
    assert isinstance(outputSize, tuple)
    self.outputSize = outputSize

  def __call__(self, sample):
    image, keypoints = sample['image'], sample['keypoints']

    h, w = image.shape[:2]
    newH, newW = self.outputSize

    img = transform.resize(image, (newH, newW))

    # h a w swapper for keypoints - images swapped axes
    keypoints = keypoints * [newW / w, newH / h]

    return {'image': img, 'keypoints': keypoints}

class ToTensor(object):
  """Convert ndarrays in sample to Tensors"""

  def __call__(self, sample):
    image, keypoints, heatmaps = sample['image'], sample['keypoints'], sample['heatmaps']

    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {'image': torch.from_numpy(image),
            'keypoints': torch.from_numpy(keypoints),
            'heatmaps': torch.from_numpy(heatmaps)}

"""
  MODEL
"""

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim batchSize
    

class Conv(nn.Module):
  def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, bn=False, relu=False):
    super(Conv, self).__init__()
    self.inChannels = inChannels
    self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride, padding=(kernelSize - 1) // 2, bias=True)
    self.relu = relu
    self.bn = bn
    if relu:
      self.reluF = nn.ReLU()
    if bn:
      self.bnF = nn.BatchNorm2d(outChannels)

  def forward(self, x):
    x = self.conv(x)
    if self.bn:
      x = self.bnF(x)
    if self.relu:
      x = self.reluF(x)
    return x


class Residual(nn.Module):
  def __init__(self, inChannels, outChannels):
    super(Residual, self).__init__()
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(inChannels)
    self.conv1 = Conv(inChannels, int(outChannels / 2), 1)
    self.bn2 = nn.BatchNorm2d(int(outChannels / 2))
    self.conv2 = Conv(int(outChannels / 2), int(outChannels / 2), 3)
    self.bn3 = nn.BatchNorm2d(int(outChannels / 2))
    self.conv3 = Conv(int(outChannels / 2), outChannels, 1)
    self.skip_layer = Conv(inChannels, outChannels, 1)

  def forward(self, x):
    residual = self.skip_layer(x)

    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)
    out += residual
    return out


class Hourglass(nn.Module):
  def __init__(self, levels, features):
    super(Hourglass, self).__init__()
    self.levels = levels

    self.skip = Residual(features, features)

    self.pool = nn.MaxPool2d(2, 2)
    self.res1 = Residual(features, features)

    # recursive level hourglass
    if self.levels > 1:
      self.res2 = Hourglass(levels - 1, features)
    else:
      self.res2 = Residual(features, features)

    self.res3 = Residual(features, features)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')

  def forward(self, x):
    skip = self.skip(x)
    x = self.pool(x)
    x = self.res1(x)
    x = self.res2(x)
    x = self.res3(x)
    x = self.up(x)
    return skip + x


class Merge(nn.Module):
  def __init__(self, x_dim, y_dim):
    super(Merge, self).__init__()
    self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

  def forward(self, x):
    return self.conv(x)


class StackedHourglass(nn.Module):
  def __init__(self, nstack, inChannels, outChannels, **kwargs):
    super(StackedHourglass, self).__init__()
    self.nstack = nstack

    # preprocessing before hourglass modules
    self.preprocess = nn.Sequential(
      Conv(3, 64, kernelSize=3, stride=1, bn=True, relu=True),
      Residual(64, 128),
      Residual(128, 128),
      Residual(128, inChannels)
    )

    # subsequent hourglass modules
    self.hourglasses = nn.ModuleList([
      nn.Sequential(
        Hourglass(4, inChannels),
      ) for i in range(nstack)])

    # reintegration of intermediate predictions
    self.features = nn.ModuleList([
      nn.Sequential(
        Residual(inChannels, inChannels),
        Conv(inChannels, inChannels, 1, bn=True, relu=True)
      ) for i in range(nstack)])

    # outputs of single hourglass modules converted to desired number of channels - heatmaps
    self.outs = nn.ModuleList([Conv(inChannels, outChannels, 1, relu=False, bn=False) for i in range(nstack)])
    self.mergeFeatures = nn.ModuleList([Merge(inChannels, inChannels) for i in range(nstack - 1)])
    self.mergePreds = nn.ModuleList([Merge(outChannels, inChannels) for i in range(nstack - 1)])
    self.nstack = nstack
    self.heatmapLoss = HeatmapLoss()

  def forward(self, x):
    # preprocessing
    x = self.preprocess(x)

    # contains predictions from all hourglass modules
    combinedHeatmapPredictions = []

    # pass through hourglass modules
    for i in range(self.nstack):
      hourglassOut = self.hourglasses[i](x)
      feature = self.features[i](hourglassOut)
      predictions = self.outs[i](feature)
      combinedHeatmapPredictions.append(predictions)

      # if not last hourglass module
      if i < self.nstack - 1:
        x = x + self.mergePreds[i](predictions) + self.mergeFeatures[i](feature)

        # stack concatenates tensors
    return torch.stack(combinedHeatmapPredictions, dim=1)

  def calcLoss(self, combinedHeatmapPredictions, heatmaps, device):
    combinedLoss = torch.from_numpy(np.zeros((20))).to(device)
    combinedHeatmapPredictions = combinedHeatmapPredictions.permute(1,2,0,3,4)
    heatmaps = heatmaps.permute(1,0,2,3)
    for i in range(self.nstack):
      combinedLoss += (self.heatmapLoss(combinedHeatmapPredictions[i], heatmaps))
    return combinedLoss

"""
  TRAINER & TESTER
"""

class CarKeypointDetector():
  def __init__(self, trainDir, testDir, trainLabels, testLabels, 
               nstack, inChannels, pretrained=False, pretrainedPath="/", epochsPretrained=0):
    self.trainDir = trainDir
    self.testDir = testDir
    self.trainLabels = trainLabels
    self.testLabels = testLabels
    self.nstack = nstack
    self.inChannels = inChannels
    self.imgSize = 64
    if pretrained:
      self.pretrainedPath = pretrainedPath
    self.epochsPretrained = epochsPretrained

    # define device (cpu/gpu)
    if(torch.cuda.is_available()):
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

    # define network
    self.net = StackedHourglass(nstack, inChannels, 20, bn=False)
    self.net = self.net.double()
    
    # load pretrained parameters if given
    if(pretrained):
      self.net.load_state_dict(torch.load(pretrainedPath))

    self.net.to(self.device)

  def train(  self, epochs, batchSize=16, trainDir=False, trainLabels=False, dataLoaderProcesses=4,
              storeNetwork=False, storeEach=5, networkPath="/", log=True, logEachIters=1000, testOnStore=False):
      if trainDir != False:
        self.trainDir = trainDir
      if trainLabels != False:
        self.trainLabels = trainLabels

      # load dataset
      dataset = CarKeyPointsDataset(csvFile=self.trainLabels, rootDir=self.trainDir)

      # set dataloader
      dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=dataLoaderProcesses)

      # set optimizer
      optimizer = optim.Adam(self.net.parameters(), lr=0.001)

      # training
      for epoch in tqdm(range(epochs)):
        for i, data in enumerate(dataloader, 0):
          inputs, gt = data['image'].to(self.device), data['heatmaps'].to(self.device)
          
          optimizer.zero_grad()
          outputs = self.net(inputs.double())
          loss = self.net.calcLoss(outputs, gt, self.device)  
          loss = torch.sum(loss)
          loss.backward()
          optimizer.step()

          if log:
            if i % logEachIters == 0:
              print("Epoch: ", epoch+1+self.epochsPretrained, ", iteration: ", i, ", loss: ", loss.item())
          
        if storeNetwork:
          if (epoch+self.epochsPretrained) % storeEach == storeEach-1:
            netName = networkPath[:-4] + "_" + str(epoch+1+self.epochsPretrained) + ".pth"
            torch.save(self.net.state_dict(), netName)

            if testOnStore != False:
              print("####### Test after epoch ", epoch+1+self.epochsPretrained)
              hgAcc, kpsAcc = self.test(log=True)


  def test(self, testDir=False, testLabels=False, log=True, dataLoaderProcesses=4):
      if testDir != False:
        self.testDir = testDir
      if testLabels != False:
        self.testLabels = testLabels

      distanceTreshold = 2

      # load dataset
      dataset = CarKeyPointsDataset(csvFile=self.testLabels, rootDir=self.testDir)

      # set dataloader
      dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=dataLoaderProcesses)

      # count accuracy for each hourglass model and keypoint
      incorrect = torch.zeros([self.nstack, 20]) 
      total = torch.zeros([self.nstack, 20]) 
      
      with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
          inputs, gt, gtKps = data['image'].to(self.device), data['heatmaps'].to(self.device), data['keypoints']
          
          outputs = self.net(inputs.double())
          loss = self.net.calcLoss(outputs, gt, self.device)
          sumLoss = torch.sum(loss)

          # remove batchSize dimension
          output = outputs[0].cpu()
          gtKps = gtKps[0].cpu()

          # iterate over hourglass models outputs
          for hgIdx, hgOutput in enumerate(output):
            # iterate over heatmap predictions of current level hg model
            for kpIdx, prediction in enumerate(hgOutput):
              #skip unlabelled keypoints
              if gtKps[kpIdx][0] < 0 or gtKps[kpIdx][0] < 0:
                continue

              # count total labelled keypoints
              total[hgIdx][kpIdx] += 1

              # find maximum value coordinations in predicted heatmap
              maximumIdx = torch.argmax(prediction) # flattened index
              maxX = (maximumIdx % prediction.shape[0]).item()
              maxY = (maximumIdx // prediction.shape[0]).item()

              # get ground-truth keypoint coordinations
              gtX = gtKps[kpIdx][0].item()
              gtY = gtKps[kpIdx][1].item()

              # calculate euclidean distance between prediction and gt 
              dist = math.sqrt((gtX - maxX)**2 + (gtY - maxY)**2)

              # incorrect keypoint estimation
              if dist > distanceTreshold:
                incorrect[hgIdx][kpIdx] += 1

      hgModelsAccuracy = []
      keypointsAccuracy = []

      for i in range(self.nstack):
        accuracy = ((1 - (incorrect[i].sum() / total[i].sum())) * 100).item()
        hgModelsAccuracy.append(accuracy)
        if log:
          print("HG model ", i+1, "total accuracy: ", accuracy, "%")
        
        kpAccuracyPerModel = []
        for kpIdx in range(incorrect[i].shape[0]):
          kpAccuracy = ((1 - (incorrect[i][kpIdx] / total[i][kpIdx])) * 100).item()
          kpAccuracyPerModel.append(kpAccuracy)
          if log:
            print("  kp ", kpIdx+1, " accuracy: ", kpAccuracy, "%")    
        keypointsAccuracy.append(kpAccuracyPerModel)

      return hgModelsAccuracy, keypointsAccuracy


'''
  Aditional methods, running training and tests, storing results etc.
'''

def storeResults(hgResults, kpsResults, filePath):
  fd = open(filePath, 'w') 
  for i, hgResult in enumerate(hgResults):
    print("HG model ", i+1, ", total accuracy: ", hgResult, "%", file=fd)
    for j, kpAcc in enumerate(kpsResults[i]):
      print("  kp ", j+1, " accuracy: ", kpAcc, "%", file=fd)
  fd.close()


def trainModel():
  # directory with train dataset
  trainDir = 'imgs_train/image_train/'
  #directory with test dataset
  testDir = 'imgs_test/image_test/'
  # file with train data labels
  trainLabels = 'imgs_train/keypoint_train.txt'
  # file with test data labels
  testLabels = 'imgs_test/keypoint_test.txt'
  # path to directory, where result networks should be stored
  netPath = '/content/drive/My Drive/ModelsKNN/'
  # name of network parameters file
  netName = 'test_4_128.pth'  # do not include number of epochs if pretrained
  
  # init network
  keypointDetector = CarKeypointDetector( trainDir, testDir, trainLabels, testLabels, 
                                          nstack=4, inChannels=128, pretrained=False)

  # train                                    
  keypointDetector.train(epochs=100, batchSize=16, 
                        storeNetwork=True, storeEach=5, 
                        networkPath= netPath + netName,
                        log=True, logEachIters=200, testOnStore=False)


def testModel():
  trainDir = ''
  trainLabels = ''
  #directory with test dataset
  testDir = 'imgs_test/image_test/'
  # file with test data labels
  testLabels = 'imgs_test/keypoint_test.txt'
  # path to directory where network which should be tested are located
  netsPath = '/content/drive/My Drive/ModelsKNN/'
  # path to directory where results should be stored
  resultsPath = '/content/drive/My Drive/ModelsKNN/results/' # kam vysledky

  # number of hg models
  nstack = 4
  # in channels to hg models
  inChannels = 128

  for filename in os.listdir(netsPath):
      if filename.endswith(".pth"):
        print("Testing: ", filename)
        epochs = filename[:-4]
        epochs = epochs.split('_')
        epochs = int(epochs[-1])

        keypointDetector = CarKeypointDetector( trainDir, testDir, trainLabels, testLabels, 
                                        nstack=nstack, inChannels=inChannels, pretrained=True,
                                        pretrainedPath=netsPath + filename, epochsPretrained=epochs)
          
        hg, kps = keypointDetector.test(log=True)
        storeResults(hg, kps, resultsPath + filename[:-4] + ".res")
      else:
        continue