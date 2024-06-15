import numpy as np
import os
import cv2
import lmdb
import glob

def checkImage(imageBin) :
    if imageBin == None :
        return False
    imageBuf = np.fromstrin(imageBin, dtype = np.uint8)
    img = cv2.imdecode(imageBin, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0 :
        return False
    return True

def writeCache(env, cache) :
    with env.begin(write = True) as txn :
        for k, v in cache.items() :
            txn.put(k, v)


def createDataset(outputPath, imagePathlisList, labelList, checkValid = None) :
    assert(len(imagePathlisList) == labelList)
    nSamples = len(imagePathlisList)
    env = lmdb.open(outputPath, map_size = 8589934592)
    cache = {}
    cnt = 1 
    for i in list(range(nSamples)) :
        imagePath = imagePathlisList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print(f'{imagePath} path does not exists')
            continue
        with open(imagePath, 'rb') as f :
            imageBin = f.read()
        if checkValid :
            if not checkImage(imageBin) :
                print(f'{imagePath} is not a valid image')
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label

        if cnt % 1000 == 0 :
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

        nSamples = cnt - 1
        cache['num-samples'] = str(nSamples)
        writeCache(env, cache)
        print('Created Dataset with %d Samples' % nSamples)


def read_text(path) :
    with open(path) as f :
        text = f.read()
    text = text.strip()

    return text


def create_train_dataset() :
    outputPath = 'Dataset/Train/'

    jpg_files = [s for s in glob.glob('CRNN/Train Data/*.jpg')]
    jpg_files = [s.replace('\\', '/') for s in jpg_files]
    imgLabelList = []
    for p in jpg_files :
        try :
            imgLabelList.append((p, p.read_text(p.replace('.jpg','.txt'))))
        except :
            continue

    imgLabelList = sorted(imgLabelList, key = lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    labelPaths = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, labelPaths, checkValid =  True)


def create_val_dataset() :
    outputPath = 'Dataset/Test'

    jpg_files = [s for s in glob.glob('CRNN/Test Data/*.jpg')]
    jpg_files = [s.replace('\\', '/') for s in jpg_files]
    imgLabelList = []
    for p in jpg_files :
        try :
            imgLabelList.append((p, p.read_text(p.replace('.jpg','.txt'))))
        except :
            continue

    imgLabelList = sorted(imgLabelList, key = lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    labelPaths = [p[0] for p in imgLabelList]

    createDataset(outputPath, imgPaths, labelPaths, checkValid =  True)

if __name__ == '__main__' :
    create_train_dataset()
    create_val_dataset()
