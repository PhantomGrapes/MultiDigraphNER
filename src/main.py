# coding: utf-8

import argparse
from hbconfig import Config
from src.data.corpus import Corpus
from src.data.corpusMeta import CorpusMeta
from src.modelHelper import ModelHelper
from src.layerHelper import LayerUtils, LayerHelper
import os
import sys
import time
import random
from src.utils.gazetteer import Gazetteer
import torch
import numpy as np
import random
import torch.nn as nn
from src.utils.gpu_tools import move2cuda
from src.utils.gpu_tools import move2cpu
from src.utils.globalVariable import GLOBAL_VARIABLE
import sys

def setSeed(seed, useGpu):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if useGpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parseTag(s):
    if s != '<UNK>' and s != '<START>' and s != '<EOS>' and s != '<PAD>':
        if s == 'O':
            return ('O', '<UNK>')
        return (s.split('-'))
    else:
        return ('<UNK>', '<UNK>')

def getEntities(tags, tag2Idx, idx2Tag):
    entityStart = -1
    entities = []
    current = 0
    while current < len(tags):
        currentLabel, currentType = parseTag(idx2Tag[tags[current]])
        if currentLabel.startswith('O'):
            if entityStart != -1:
                entities.append((entityStart, current - 1, idx2Tag[tags[entityStart]].split('-')[1]))
                entityStart = -1
        elif currentLabel.startswith('B'):
            if entityStart != -1:
                entities.append((entityStart, current - 1, idx2Tag[tags[entityStart]].split('-')[1]))
            entityStart = current
        elif currentLabel.startswith('S'):
            if entityStart != -1:
                entities.append((entityStart, current - 1, idx2Tag[tags[entityStart]].split('-')[1]))
            entities.append((current, current, currentType))
            entityStart = -1
        elif currentLabel.startswith('E'):
            if entityStart != -1:
                entities.append((entityStart, current, currentType))
            entityStart = -1
        current += 1

    if entityStart != -1:
        entities.append((entityStart, len(tags) - 1, idx2Tag[tags[entityStart]].split('-')[1]))
    return entities

def evaluate(model, batchInput, tag2Idx, idx2Tag, dumpFile=None, id2Word=None, rawSentenceBatch=None, useGpu=False):
    model.eval()
    s = 0
    g = 0
    sIgStrict = 0
    sIgLoose = 0
    batchId = -1
    for batchItem in batchInput:
        batchId += 1
        if useGpu:
            batchItem = move2cuda(batchItem)
        wordSeqTensor =  batchItem[0]
        tagSeqTensor = batchItem[1]
        wordSeqLengths = batchItem[2]
        bestScores, decodeIdx = model(batchItem)
        tagSeq = tagSeqTensor.data
        predSeq = decodeIdx.data
        for batchPos in range(len(wordSeqTensor)):
            sentLength = wordSeqLengths[batchPos]
            gold = tagSeq[batchPos][: sentLength]
            pred = reversed(predSeq[batchPos][: sentLength])

            sentenceString = ' '.join(rawSentenceBatch[batchId][batchPos])

            if dumpFile is not None:
                for wordIdx in range(len(pred)):
                    dumpFile[2].write(rawSentenceBatch[batchId][batchPos][wordIdx] + '\t' + idx2Tag[pred[wordIdx]] + '\n')
                dumpFile[2].write('\n')
                for wordIdx in range(len(gold)):
                    dumpFile[5].write(rawSentenceBatch[batchId][batchPos][wordIdx] + '\t' + idx2Tag[gold[wordIdx]] + '\n')
                dumpFile[5].write('\n')

            goldEntities = getEntities(gold, tag2Idx, idx2Tag)
            predEntities = getEntities(pred, tag2Idx, idx2Tag)

            if dumpFile is not None:
                dumpFile[3].write(sentenceString + '\n')
                dumpFile[4].write(sentenceString + '\n')
                for entity in goldEntities:
                    dumpFile[3].write(' '.join(rawSentenceBatch[batchId][batchPos][entity[0]: entity[1] + 1]) + '\t' + str(entity[0]) + '\t'
                                      + str(entity[1]) + '\t' + entity[2] + '\n')
                for entity in predEntities:
                    dumpFile[4].write(' '.join(rawSentenceBatch[batchId][batchPos][entity[0]: entity[1] + 1]) + '\t' + str(entity[0]) + '\t'
                                      + str(entity[1]) + '\t' + entity[2] + '\n')
                dumpFile[3].write('\n')
                dumpFile[4].write('\n')

            s += len(predEntities)
            g += len(goldEntities)
            if len(predEntities) == 0:
                if dumpFile is not None:
                    dumpFile[0].write(sentenceString)
                    for entity in goldEntities:
                        dumpFile[0].write('\t'.join([' '.join(rawSentenceBatch[batchId][batchPos][entity[0]: entity[1] + 1]), str(entity[0]), str(entity[1]), entity[2]]) + '\n')
                continue
            predIdx = 0
            strictFalse = []
            looseFalse = []
            for entityIdx in range(len(goldEntities)):
                losseMatch = False
                strictMath = False
                entityStart, entityEnd, entityType = goldEntities[entityIdx]
                while predIdx < len(predEntities) - 1 and predEntities[predIdx][1] < entityStart:
                    predIdx += 1
                if entityType == predEntities[predIdx][2]:
                    if entityStart == predEntities[predIdx][0] and entityEnd == predEntities[predIdx][1]:
                        sIgStrict += 1
                        sIgLoose += 1
                        strictMath = True
                    elif max(predEntities[predIdx][0], entityStart) <= min(entityEnd, predEntities[predIdx][1]):
                        sIgLoose += 1
                        losseMatch = True
                if dumpFile is not None and not strictMath:
                    if not losseMatch:
                        strictFalse.append(goldEntities[entityIdx])
                    else:
                        looseFalse.append([goldEntities[entityIdx], predEntities[predIdx]])
            if dumpFile is not None:
                if len(strictFalse) > 0:
                    dumpFile[0].write(sentenceString + '\n')
                    for entity in strictFalse:
                        dumpFile[0].write('\t'.join([''.join(rawSentenceBatch[batchId][batchPos][entity[0]: entity[1] + 1]), str(entity[0]), str(entity[1]), entity[2]]) + '\n')
                if len(looseFalse) > 0:
                    dumpFile[1].write(sentenceString + '\n')
                    for entity in looseFalse:
                        dumpFile[1].write('\t'.join([''.join(rawSentenceBatch[batchId][batchPos][entity[0][0]: entity[0][1] + 1]), str(entity[0][0]), str(entity[0][1]), entity[0][2], ''.join(rawSentenceBatch[batchId][batchPos][entity[1][0]: entity[1][1] + 1]), str(entity[1][0]), str(entity[1][1]), entity[1][2]]) + '\n')
    ps = sIgStrict / s if s > 0 else 0
    pl = sIgLoose / s if s > 0 else 0
    rs = sIgStrict / g if g > 0 else 0
    rl = sIgLoose / g if g > 0 else 0
    fs = 2 * ps * rs / (ps + rs) if ps + rs > 0 else 0
    fl = 2 * pl * rl / (pl + rl) if pl + rl > 0 else 0
    return ps * 100, pl * 100, rs * 100, rl * 100, fs * 100, fl * 100


def train(Config):
    setSeed(Config.train.seed, Config.use_gpu)

    # load data
    dataPath = os.path.join(Config.data.data_base_path)
    if Config.eval.do_eval:
        testData = Corpus(os.path.join(dataPath, 'test.txt'), Config.data.use_normalized_word)
    trainData = Corpus(os.path.join(dataPath, 'train.txt'), Config.data.use_normalized_word)
    if Config.use_dev:
        devData = Corpus(os.path.join(dataPath, 'dev.txt'), Config.data.use_normalized_word)
    if Config.data.get('word_embedding', False):
        trainData.words.update(testData.words)
        trainData.words.update(devData.words)
        trainData.bwbigrams.update(testData.bwbigrams)
        trainData.bwbigrams.update(devData.bwbigrams)
        trainData.fwbigrams.update(testData.fwbigrams)
        trainData.fwbigrams.update(devData.fwbigrams)

    # gazetter
    gazetters = []
    if Config.model.get("graph_emb", None) is not None and Config.model.graph_emb.get('gazetter', None) is not None:
        for name in Config.model.graph_emb.gazetter.to_dict():
            if name != 'get_tag':
                gaItem = Config.model.graph_emb.gazetter.get(name)
                gazetters.append(Gazetteer(name, gaItem['path'], Config.data.use_normalized_word, gaItem['emb_dim'], gaItem['method'], gaItem['space'], gaItem['match_ignore_case'], embedding=gaItem.get('embedding', None)))
                gazetters[-1].matchCorpus(testData)
                gazetters[-1].matchCorpus(devData)
                gazetters[-1].matchCorpus(trainData)

    # generate train data meta
    print("Generating corpus meta...")
    trainMeta = CorpusMeta(trainData, Config)
    trainMeta.updateMaxSentLength(devData)
    trainMeta.updateMaxSentLength(testData)
    trainMeta.gazetters = gazetters

    # initialize model
    print("Initializing model...")
    layerUtils = LayerUtils(Config, trainMeta)
    layerHelper = LayerHelper(Config, layerUtils)

    modelHelper = ModelHelper(Config, layerHelper)
    if Config.model.get('load_from_pretrain', False):
        model = modelHelper.loadModel()
    else:
        model = modelHelper.getModel()
    if Config.use_gpu:
        model.cuda()
    trainer = modelHelper.getTrainer(model)
    if len(Config.gpu_num) > 1:
        trainer = nn.DataParallel(trainer, device_ids=device_ids)

    # genenerate batch input
    print("Generating batch input...")

    if Config.use_dev:
        if len(Config.gpu_num) > 1:
            devRawSentenceBatch = model.module.getRawSentenceBatches(devData, trainMeta, Config.train.batch_size)
        else:
            devRawSentenceBatch = model.getRawSentenceBatches(devData, trainMeta, Config.train.batch_size)
    if Config.eval.do_eval:
        if len(Config.gpu_num) > 1:
            testRawSentenceBatch = model.module.getRawSentenceBatches(testData, trainMeta, Config.train.batch_size)
        else:
            testRawSentenceBatch = model.getRawSentenceBatches(testData, trainMeta, Config.train.batch_size)
    logFolder = os.path.join(Config.log_folder, Config.model.model_name)
    if not os.path.exists(logFolder):
        os.makedirs(logFolder)

    # train
    trainLog = open(os.path.join(logFolder, "train.log"), 'w', encoding='utf-8')
    trainLog.write(Config.__str__()+'\n')
    print(Config.__str__())
    trainStart = time.time()
    bestF1 = -1
    bestEpoch = -1
    for epoch in range(1, Config.train.epoch + 1):
        random.shuffle(trainData.utterances)
        if len(Config.gpu_num) > 1:
            batchInput = model.module.generateBatchInput(trainData, trainMeta, Config.train.batch_size)
        else:
            batchInput = model.generateBatchInput(trainData, trainMeta, Config.train.batch_size)
        if Config.train.optimizer == "sgd":
            trainer = modelHelper.lrDecay(trainer, epoch)
        model.train()
        model.zero_grad()
        sampleLoss = 0
        sampleCount = 0
        tempStart = time.time()
        epochStart = time.time()
        epochLoss = 0
        for batchItem in batchInput:
            if Config.use_gpu:
                batchItem = move2cuda(batchItem)
            if len(Config.gpu_num) > 1:
                loss = model(batchItem, negMode=True).sum()
            else:
                loss = model.negLogLikelihoodLoss(batchItem)
            sampleLoss += loss.data
            epochLoss += float(loss.data)
            sampleCount += len(batchItem[0])
            loss.backward()
            trainer.step()
            model.zero_grad()
            if sampleCount >= Config.train.report_frequence:
                tempTime = time.time()
                tempCost = tempTime - tempStart
                tempStart = tempTime
                print("Process {} sentences. Loss: {:.2f}. Time: {:.2f}".format(sampleCount, loss/sampleCount, tempCost))
                trainLog.write("Process {} sentences. Loss: {:.2f}. Time: {:.2f}".format(sampleCount, loss/sampleCount, tempCost) + '\n')
                if sampleLoss > 1e8 or str(sampleLoss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    trainLog.write("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT...." + '\n')
                    exit(1)
                sys.stdout.flush()
                sampleLoss = 0
                sampleCount = 0
        epochCost = time.time() - epochStart
        if Config.use_dev:
            if len(Config.gpu_num) > 1:
                devBatchInput = model.module.generateBatchInput(devData, trainMeta, Config.train.batch_size)
            else:
                devBatchInput = model.generateBatchInput(devData, trainMeta, Config.train.batch_size)
        if Config.eval.do_eval:
            if len(Config.gpu_num) > 1:
                testBatchInpit = model.module.generateBatchInput(testData, trainMeta, Config.train.batch_size)
            else:
                testBatchInpit = model.generateBatchInput(testData, trainMeta, Config.train.batch_size)
        if Config.use_dev and epoch % Config.train.dev_epoch_frequence == 0:
            model.eval()
            ps, pl, rs, rl, fs, fl = evaluate(model, devBatchInput, trainMeta.tag2Idx, trainMeta.idx2Tag, rawSentenceBatch=devRawSentenceBatch, useGpu=Config.use_gpu)
            if fs >= bestF1:
                bestF1 = fs
                bestEpoch = epoch
                modelHelper.saveModel(model, epoch)
            print("Epoch {}. Loss: {:.2f}. Time: {:.2f}".format(epoch, epochLoss, epochCost))
            print('Dev P: {:.2f} R: {:.2f} F1: {:.2f}'.format(ps, rs, fs))
            ps, pl, rs, rl, fs, fl = evaluate(model, testBatchInpit, trainMeta.tag2Idx, trainMeta.idx2Tag, rawSentenceBatch=testRawSentenceBatch, useGpu=Config.use_gpu)
            print('Test P: {:.2f} R: {:.2f} F1: {:.2f}'.format(ps, rs, fs))
            trainLog.write("Epoch {}. Loss: {:.2f}. Dev P: {:.2f} R: {:.2f} F1: {:.2f}. Time: {:.2f}".format(
                    epoch, epochLoss, ps, rs, fs, epochCost) + '\n')
        else:
            print("Epoch {}. Loss: {:.2f}. Time: {:.2f}".format(epoch, epochLoss, epochCost))
            trainLog.write("Epoch {}. Loss: {:.2f}. Time: {:.2f}".format(epoch, epochLoss, epochCost) + '\n')
        sys.stdout.flush()


    if bestF1 == 0:
        modelHelper.saveModel(model, 0)

    print("Finish training. Best epoch {}. F1 {:.2f}. Time {:.2f}".format(bestEpoch, bestF1, time.time() - trainStart))
    trainLog.write("Finish training. Time {:.2f}".format(time.time() - trainStart) + '\n')
    trainLog.close()

    if Config.eval.do_eval:
        model = modelHelper.loadModel(bestEpoch)
        if Config.use_gpu:
            model.cuda()
        testBatchInpit = model.generateBatchInput(testData, trainMeta, 20)
        testRawSentenceBatch = model.getRawSentenceBatches(testData, trainMeta, 20)
        testLog = open(os.path.join(logFolder, 'test.log'), 'w', encoding='utf-8')
        strictFalse = open(os.path.join(logFolder, 'strictFalse'), 'w', encoding='utf-8')
        looseFalse = open(os.path.join(logFolder, 'looseFalse'), 'w', encoding='utf-8')
        rawOutput = open(os.path.join(logFolder, 'testOutput'), 'w', encoding='utf-8')
        goldOutput = open(os.path.join(logFolder, 'testGold'), 'w', encoding='utf-8')
        goldEntityOutput = open(os.path.join(logFolder, 'testGoldEntities'), 'w', encoding='utf-8')
        predEntityOutput = open(os.path.join(logFolder, 'testPredEntities'), 'w', encoding='utf-8')
        ps, pl, rs, rl, fs, fl = evaluate(model, testBatchInpit, trainMeta.tag2Idx, trainMeta.idx2Tag, (
        strictFalse, looseFalse, rawOutput, goldEntityOutput, predEntityOutput, goldOutput), trainMeta.idx2Word,
                                          testRawSentenceBatch, useGpu=Config.use_gpu)
        print("Test P: {:.2f} R: {:.2f} F1: {:.2f}".format(ps, rs, fs))
        testLog.write("Test P: {:.2f} R: {:.2f} F1: {:.2f}".format(ps, rs, fs) + '\n')
        strictFalse.close()
        looseFalse.close()
        testLog.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='baseline')
    args = parser.parse_args()

    Config(args.config)
    if Config.mode == 'train':
        train(Config)
