import torch.optim as optim
import torch
import os
from src.layerHelper import LayerHelper
import dill as pickle
import random
import numpy as np
from collections import OrderedDict
from src.models.graphEmbCrf import GraphEmbCrf

class ModelHelper:
    def __init__(self, Config, layerHelper):
        self.Config = Config
        self.layerHelper = layerHelper

    def getModel(self):
        if self.Config.model.model_type == 'graph':
            return GraphEmbCrf(self.Config, self.layerHelper)

    def lrDecay(self, trainer, epoch):
        lr = self.Config.train.learning_rate / (1 + self.Config.train.lr_decay * (epoch - 1))
        if len(self.Config.gpu_num) > 1:
            for param_group in trainer.module.param_groups:
                param_group['lr'] = lr
        else:
            for param_group in trainer.param_groups:
                param_group['lr'] = lr
        print('learning rate is set to: ', lr)
        return trainer

    def getTrainer(self, model):
        if self.Config.train.optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.Config.train.learning_rate, momentum=self.Config.train.momentum, weight_decay=float(self.Config.train.l2))
        elif self.Config.train.optimizer.lower() == "adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=self.Config.train.learning_rate, weight_decay=float(self.Config.train.l2))
        elif self.Config.train.optimizer.lower() == "adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=self.Config.train.learning_rate, weight_decay=float(self.Config.train.l2))
        elif self.Config.train.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=self.Config.train.learning_rate, weight_decay=float(self.Config.train.l2))
        elif self.Config.train.optimizer.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.Config.train.learning_rate, weight_decay=float(self.Config.train.l2))
        else:
            print("Optimizer illegal: %s" % self.Config.train.optimizer)
            exit(1)
        return optimizer

    def saveModel(self, model, epoch):
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name)
        if not os.path.exists(modelFolder):
            os.makedirs(modelFolder)
        torch.save(model.state_dict(), os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch)))

    def saveIterModel(self, model, iter, epoch):
        iter = str(iter)
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name, iter)
        if not os.path.exists(modelFolder):
            os.makedirs(modelFolder)
        torch.save(model.state_dict(), os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch)))

    def saveIncompleteModel(self, model, iter, fold, epoch):
        iter = str(iter)
        fold = str(fold)
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name, iter, fold)
        if not os.path.exists(modelFolder):
            os.makedirs(modelFolder)
        torch.save(model.state_dict(), os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch)))

    def loadModel(self, epoch):
        model = self.getModel()
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name)
        modelPath = os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch))
        if not os.path.exists(modelPath):
            print("Can't find model in {}.".format(modelFolder))
            exit(1)
        if len(self.Config.gpu_num) > 1:
            state_dict = torch.load(modelPath)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = k[7:]
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(torch.load(modelPath))
        return model

    def loadIncompleteModel(self, iter, fold, epoch):
        iter = str(iter)
        fold = str(fold)
        model = self.getModel()
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name, iter, fold)
        modelPath = os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch))
        if not os.path.exists(modelPath):
            print("Can't find model in {}.".format(modelFolder))
            exit(1)
        model.load_state_dict(torch.load(modelPath))
        return model

    def loadIterModel(self, iter, epoch):
        iter = str(iter)
        model = self.getModel()
        modelFolder = os.path.join(self.Config.model.base_path, self.Config.model.model_name, iter)
        modelPath = os.path.join(modelFolder, 'model.pkl' + '-' + str(epoch))
        if not os.path.exists(modelPath):
            print("Can't find model in {}.".format(modelFolder))
            exit(1)
        model.load_state_dict(torch.load(modelPath))
        return model
