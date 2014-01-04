__author__ = 'deeksha'

from sklearn import cross_validation
import csv
import sys
from sklearn import preprocessing
import numpy as np



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


#extract data from a CSV file
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


#split the data into training and testing splits
def split(inData, TrainProp=0.9):
    ''' Given an sklearn dataset, return two lists:
    1. the first TrainProp proportion of the data, [X_train, Y_Train]
    2. the second (1-TrainProp) proportion of the data, [X_test, Y_test]
    '''
    x_data = inData['data']
    y_data = inData['target']
    n_samples = len(x_data)
    slice_size = int(TrainProp * n_samples)
    x_train = x_data[:slice_size]
    y_train = y_data[:slice_size]
    x_test = x_data[slice_size:]
    y_test = y_data[slice_size:]
    return [x_train, y_train], [x_test, y_test]


#function to calculate the euclidean distance
def euclidean_distance(querypoint,data_array):
    distance = []
    querypoint = np.array(querypoint)

    for array in data_array:
        distance.append(np.sqrt(np.sum((querypoint-array)**2)))
    return distance


#function which return the majority class of the k nearest neighbour
def k_nearest_neighbours(n_neighbours, training_data, queryPoint, rescale):
    x_train = training_data[0]
    x_train = np.array(x_train)
    if rescale == "TRUE":
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        scaler = preprocessing.StandardScaler().fit(queryPoint)
        queryPoint = scaler.transform(queryPoint)
    distance = euclidean_distance(queryPoint,x_train)
    y_train = training_data[1]
    y_train= np.array(y_train)
    tuple = zip(distance,y_train)
    sorted_tuple = sorted(tuple)
    knn = sorted_tuple[:n_neighbours]
    knn = [x[1] for x in knn]
    knn = max(set(knn), key=knn.count)
    return knn


#function which return the k classifier for fixed querypoint
def question3a(fileName, target):
    dataDict = extractData(fileName, target, delim =',')
    x_train = dataDict['data']
    y_train = dataDict['target']
    data = [x_train, y_train]
    n_neighbours = [1, 3, 5]
    rescale = "TRUE"
    queryPoint = [2.5, 2.5]
    for k in n_neighbours:
        knn = k_nearest_neighbours(k, data, queryPoint, rescale)
        print "For k:", k, "Classifier:", knn


def question3b(fileName, target):
    dataDict = extractData(fileName, target, delim =',')
    [x_train, y_train], [x_test, y_test] = split(dataDict, TrainProp=0.9)
    training_data = [x_train, y_train]
    n_neighbours = 1
    rescale = "TRUE"
    error = 0
    for i in range(len(x_test)):
        query = x_test[i]
        target = y_test[i]

        knn = k_nearest_neighbours(n_neighbours, training_data, query, rescale)
        if knn <> target:
            error += 1
    score = float(len(x_test)-error)/len(x_test)
    print score


def kfold(fileName, target):
    rescale = "FALSE"
    dataDict = extractData(fileName, target, delim =',')
    perm = list(np.random.permutation(len(dataDict['data'])))
    x_train = [dataDict['data'][p] for p in perm]
    y_train = [dataDict['target'][p] for p in perm]
    data = [x_train, y_train]
    n_neighbours = filter(lambda x: x % 2, range(0, 13))
    k_fold = cross_validation.KFold(n=len(x_train), n_folds=5, indices=True)
    for k in n_neighbours:
        scores = []
        for train, test in k_fold:
            error = 0
            training_data_x = [data[0][t] for t in list(train)]
            training_data_y = [data[1][t] for t in list(train)]
            training_data = [training_data_x, training_data_y]
            test_data_x = [data[0][t] for t in list(test)]
            test_data_y = [data[1][t] for t in list(test)]
            for i in range(len(test_data_x)):
                query = test_data_x[i]
                target = test_data_y[i]
                knn = k_nearest_neighbours(k, training_data, query, rescale)
                if knn <> target:
                    error += 1
            score = float(len(test_data_x)-error)/len(test_data_x)
            scores.append(score)
        print "For k:", k, "Score:", np.average(scores)

#function that returns result to question 3a
question3a( "MLHW3.csv" ,"y")
# function that returns result to question 4
kfold( "wineHeaders.csv" ,"Target")
#function which returns the answer to question 3b
#question3b( "BioResponseKaggleTrain.csv","Activity")