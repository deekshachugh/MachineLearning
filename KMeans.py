from time import time
import numpy as np
import pylab as pl
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import csv
from sklearn.preprocessing import scale



def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def cvrt_to_num_if_can(str):
    ''' If str is really a number,
    convert it to same, preferring ints
    '''
    if is_int(str):
        return int(str)
    elif is_float(str):
        return float(str)
    return str



def extractData(fileName, targetName, delim =','):
    '''Given the name of a file, the name of the target variable column,
    and optionally the column deliminator (',' is the default),
    Return a sklearn-style dictionary
    '''
    try:
        in_file = open(fileName, 'rUb')
        reader = csv.reader(in_file, delimiter=delim, quotechar='"')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        raise
    except ValueError:
        print "Could not convert data."
        raise
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    #initialization
    dataDict = {}
    dataDict['feature_names'] = []
    dataDict['target_names'] = []
    dataDict['target'] =[ ]
    dataDict['data'] = []
    fieldNames = []
    #read the header row
    for row in reader:
        for field in row:
            if field != '':
                fieldNames.append(field)
        break

    #find the index of the target value, if exists
    try:
        targetIdx = fieldNames.index(targetName)
    except ValueError:
        print "Target %s not in fields %s" %(targetName, fieldNames)
        raise
    fieldNames = fieldNames[:targetIdx] + fieldNames[targetIdx+1:]
    dataDict['feature_names'] = fieldNames

    #read the data
    for row in reader:
        #We may want to later have more sophistication if values are missing,
        # but for now we fill the example with "None"
        rowData = [None for i in range(len(fieldNames))]
        #add one to length because the target is also there
        if len(row) != len(fieldNames)+1:
            print "found a bad row? ",row
        dataIdx = 0
        for colIdx in range(len(row)):
            if colIdx == targetIdx:
                tVal = cvrt_to_num_if_can(row[colIdx])
                dataDict['target'].append(tVal)
            elif row[colIdx] != r'\N' and row[colIdx] != "":
                rowData[dataIdx] = cvrt_to_num_if_can(row[colIdx])
                dataIdx += 1
        dataDict['data'].append(rowData)

    #get unique targets
    dataDict['target_names'] = list(set(dataDict['target']))
    return dataDict


#function to calculate score
def bench_k_means(estimator, name, data, labels):
    estimator.fit(data)
    return metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')


def question1b():
    np.random.seed(42)
    digits = load_digits()
    data = scale(digits.data)
    labels = digits.target
    Kset = list(range(2,13))
    Kset.extend([15, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    scores = []
    for k in Kset:
        score = bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
                  name="random", data=data,labels=labels)
        scores.append(score)
        print k,score
    pl.plot(Kset,scores)
    pl.show()


def question1c(fileName, target):
    dataDict = extractData(fileName, target, delim =',')
    scaled_data = scale(np.array(dataDict['data']))
    labels = np.array(dataDict['target'])
    Kset = list(range(2,21))
    scores = []
    for k in Kset:
        score = bench_k_means(KMeans(init='random', n_clusters=k, n_init=10),
                  name="random", data=scaled_data, labels=labels)
        scores.append(score)
        print k,score
    pl.plot(Kset,scores)
    pl.show()

question1b()
question1c("hw4-cluster.csv","ID")


