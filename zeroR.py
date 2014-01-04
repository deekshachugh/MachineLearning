from sklearn import datasets
from sklearn.datasets import fetch_mldata
import numpy as np
import argparse
import csv
import sys


#function to read csv file and return a dictionary
def readCSV(filename):
    try:
        #opening the file
        in_file = open(filename, 'rUb')
        reader = csv.reader(in_file, delimiter=';', quotechar='"')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        raise
    except ValueError:
        print "Could not convert data to an integer."
        raise
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
    fieldnames = []
    #reading the file
    for row in reader:
        #checking if the length of row is not equal to zero
        if len(row) <> 0:
            for field in row:
                #checking the field is not blank
                if field <> '':
                    #appending the column headings in fieldnames
                    fieldnames.append(field)
            break
    dataDict = {'feature_names': fieldnames[1:]}
    target_list = []
    data = []
    for row in reader:
        target_list.append(row[0])
        data.append(row[1:])
    dataDict["target"] = target_list
    dataDict["data"] = data
    in_file.close()
    return dataDict


#function to read dat file and return a dictionary
def read_dat(filename):
    try:
        #reading the file
        in_file = open(filename, 'rUb')
        #opening the file
        reader = csv.reader(in_file, delimiter=';', quotechar='"')
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        raise
    except ValueError:
        print "Could not convert data to an integer."
        raise
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise
    fieldNames = []
    for row in reader:
        #reading the first meaningful row which contains column headings
        if len(row) <> 0 and row[0]=='user':
            for field in row:
                if field <> '':
                    fieldNames.append(field)
            break
    dataDict = {'feature_names': fieldNames[0:len(row)-1]}
    target_list = []
    data = []
    for row in reader:
        target_list.append(row[len(row)-1])
        data.append(row[0:len(row)-2])
    dataDict["target"] = target_list
    dataDict["data"] = data
    in_file.close()
    return dataDict


#function to detect the file type and execute the functions corresponding to those file type
def info_extract(dataDict, data_set, file_type):
    if  file_type == "csv" or data_set == "wearable-accelerometers-activity.dat":
        Number_of_Examples, Number_of_Features, Target_variables = info_csv(dataDict)
    else:
        Number_of_Examples, Number_of_Features, Target_variables = info_data(dataDict)
    return Number_of_Examples, Number_of_Features, Target_variables


#function to extract information from a dictionary made by a csv file
def info_csv(dataDict):
    Number_of_Examples = len(dataDict['data'])
    Number_of_Features = len(dataDict['feature_names'])
    Target_variables = np.unique(dataDict["target"])
    return Number_of_Examples, Number_of_Features, Target_variables


#function to extract relevant information from the data set dictionary
def info_data(dataDict):
    Number_of_Examples = dataDict.data.shape[0]
    Number_of_Features = dataDict.data.shape[1]
    Target_variables = np.unique(dataDict["target"])
    return Number_of_Examples, Number_of_Features, Target_variables


#function that classifies data based on the most common classification in the training set.
def zeroR(dataDict):
    count_target_dict = {}
    target_variables = np.unique(dataDict["target"])
    for variable in target_variables:
        count_target_dict[variable] = 0
    for variable in dataDict["target"]:
        count_target_dict[variable] += 1
    maxProb = max(count_target_dict, key=count_target_dict.get)
    return maxProb


#main program for zeroR
#takes arguments -f=['csv' | 'toy' | 'mldata'] dataset_name
# the f argument specifies which type of data set we are dealing with,
# and dataset_name names which dataset if toy or mldata and filename if csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-f", type=str, required=True)
    args = parser.parse_args()
    file_type = args.f
    data_set = args.dataset_name
    if args.f == 'csv':
        dataDict = readCSV(data_set)
    elif args.f == 'toy':
        dataDict = eval("datasets.load_"+data_set+"()")
    elif args.f == 'mldata' and data_set <> "wearable-accelerometers-activity.dat":
        to_eval = "datasets.fetch_mldata(\""+data_set+"\")"
        dataDict = eval(to_eval)
    elif args.f == 'mldata' and  data_set == "wearable-accelerometers-activity.dat":
        dataDict = read_dat(data_set)
    Number_of_Examples, Number_of_Features, Target_variables = info_extract(dataDict,data_set, file_type)
    zeroR(dataDict)
    print "# of examples:  ", Number_of_Examples
    print "# of features:  ", Number_of_Features
    print "Target variables: ", Target_variables
    print "ZeroR class label: ", zeroR(dataDict)