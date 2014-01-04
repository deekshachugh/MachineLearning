__author__ = 'deeksha'

import csv
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO



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


#function which evaluates the accuracy of the test data
def test_evaluate(model, test_data):
    '''
    assumes test data is a list of [x_test, y_test]
    '''
    if type(model)== np.int64:
        predictions = [model]*len(test_data[0])
        num_wrong = len([i for i, j in zip(predictions, test_data[1]) if i != j])
    else:
        predictions = model.predict(test_data[0])
        num_wrong = (predictions != test_data[1]).sum()
    num_exs = float(len(test_data[0]))
    score = (num_exs-num_wrong)/num_exs

    return score


#function which creates the decision tree classifier using Gini splitting criteria
def decision_tree_gini(training_data):
    clf = DecisionTreeClassifier(criterion="gini",random_state=0)
    clf.fit(training_data[0], training_data[1])
    #with open("/media/deeksha/e/Deeksha/Dropbox/Coursework/MachineLearning/HW3/gini.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, out_file=f)
    print "gini:Number of Nodes", clf.tree_.node_count
    return clf


#function which creates the decision tree classifier using Entropy splitting criteria
def decision_tree_entropy(training_data):
    clf = DecisionTreeClassifier(criterion="entropy",random_state=0)
    clf.fit(training_data[0], training_data[1])
    #with open("/media/deeksha/e/Deeksha/Dropbox/Coursework/MachineLearning/HW3/entropy.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, out_file=f)
    print "entropy:Number of Nodes", clf.tree_.node_count
    return clf


#function which evaluates the accuracy of the training data
def training_evaluate(model, training_data):
    '''
    assumes training data is a list of [x_training, y_training]
    '''
    predictions = model.predict(training_data[0])
    num_wrong = (predictions != training_data[1]).sum()
    num_exs = float(len(training_data[0]))
    score = (num_exs-num_wrong)/num_exs
    return score


#ZeroR is a function which returns the majority class of the database
def zeroR(Target):
    count_target_dict = {}
    target_variables = np.unique(Target)
    for variable in target_variables:
        count_target_dict[variable] = 0
    for value in Target:
        count_target_dict[value] += 1
    maxProb = max(count_target_dict, key=count_target_dict.get)

    return maxProb


#Function which takes input as filename and target
def model(fileName, target):
    dataDict = extractData(fileName, target, delim =',')
    [x_train, y_train], [x_test, y_test] = split(dataDict, TrainProp=0.9)
    test_data = [x_test, y_test]
    training_data = [x_train, y_train]
    print "Training data size", len(x_train)
    #Decision tree classifier functions
    model_gini = decision_tree_gini(training_data)
    model_entropy = decision_tree_entropy(training_data)
    #Zero R function
    label_zeroR = zeroR(y_train)
    #Function evaluate calculates the accuracy of test data using ZeroR
    score = test_evaluate(label_zeroR, test_data)
    print 'Test set accuracy of ZeroR: %f' %score
    #Function training_evaluate calculates the accuracy of the training data
    training_evaluate(model_gini, training_data)
    print 'Training set accuracy using Gini: %f' %score,
    training_evaluate(model_entropy, training_data)
    print 'Training set accuracy using Entropy: %f' %score,
    #Function evaluate calculates the accuracy of test data
    score = test_evaluate(model_gini, test_data)
    print 'Test set accuracy using Gini: %f' %score
    score = test_evaluate(model_entropy, test_data)
    print 'Test set accuracy using Entropy: %f' %score


#Just call this function model with the required filename and target column
model("BioResponseKaggleTrain.csv","Activity")

