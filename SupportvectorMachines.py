__author__ = 'deeksha'


from sklearn import metrics
import csv
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split


def read_data(filename):
    #reads csv file and
    #returns features and target variable
    X = []
    Y = []
    # Load the data from the csv file
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            i += 1
            if i == 1 and filename == '2d_example.csv':
            #ignoring the first line if the file is '2d_example.csv'
                continue
            if filename == '2d_example.csv':
                X.append(row[1:len(row)-1])
            else:
                X.append(row[0:len(row)-1])
            Y.append(row[len(row)-1])
    return X, Y


def graphs_parameters_svm(filename):
    #reading the features and target variables from the file
    features, target = read_data(filename)
    # to normalize the data by subtracting mean and dividing by standard deviation
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    #setting the ranges of gamma and C
    C_2d_range = [1, 1e2, 1e4, 1e5]
    gamma_2d_range = [1e-1, 1, 1e1, 1e2]
    #classifiers will contain list of all the models for various ranges of C and gamma
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(features, target)
            classifiers.append((C, gamma, clf))
    target = [int(y) for y in target]

    pl.figure(figsize=(12, 10))
    # construct a mesh
    h = .02
    x = np.array(features, dtype=float)
    y = np.array(target, dtype= int)
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #Plotting Support vectors
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        pl.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        pl.title("gamma 10^%d, C 10^%d" % (np.log10(gamma), np.log10(C)),size='medium')
        pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        pl.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1], c=y[clf.support_], cmap=pl.cm.Paired)
        pl.xticks(())
        pl.yticks(())
        pl.axis('tight')
    pl.show()

    pl.figure(figsize=(12, 10))
    #plotting decision boundary
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # visualize decision function for these parameters
        pl.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        pl.title("gamma 10^%d, C 10^%d" % (np.log10(gamma), np.log10(C)),
                 size='medium')
        # visualize parameter's effect on decision function
        pl.pcolormesh(xx, yy, -Z, cmap=pl.cm.jet)
        pl.scatter(features[:, 0], features[:, 1], c=target, cmap=pl.cm.jet)
        pl.xticks(())
        pl.yticks(())
        pl.axis('tight')
    pl.show()
    optimal_model = grid_search(clf, x, y)
    print " the optimal parameters are (C gamma):(", optimal_model.C,optimal_model._gamma, ")"


def grid_search(model, x_train, y_train):
    #input parameters are the model and data and
    #it returns the best model by searching over common values of C and gamma
    #set the C and gamma range over which grid search will be applied
    C_range = 10.0 ** np.arange(-1, 6)
    gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #stratified k folds of data will be created
    cv = StratifiedKFold(y=y_train, n_folds=3)
    #below function searches over the entire space of param_grid
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    return grid.best_estimator_


def svm_model(X,Y):
    #input parameters are features and target variables.
    #returns the rbf model and linear model
    col = len(X)
    X1 = []
    for i in range(col-1):
        X1.append([float(x) for x in X[i]])
    Y = [float(y) for y in Y if y]
    X1 = np.asarray(X1)
    Y1 = np.asarray(Y)
    # divide the dataset into training and testing set
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X1, Y1, test_size=0.15, random_state = 0)
    #creating two SVM classifiers
    clf_rbf = svm.SVC(kernel='rbf')
    clf_linear = svm.SVC(kernel='linear')
    best_model_rbf = grid_search(clf_rbf, x_train, y_train)
    best_model_linear = grid_search(clf_linear, x_train, y_train)
    print "Accuracy of Rbf model ", best_model_rbf.score(x_test, y_test)
    y_pred = best_model_rbf.predict(x_test)
    print metrics.classification_report(y_test, y_pred)
    print "Accuracy of Linear model", best_model_linear.score(x_test, y_test)
    y_pred = best_model_linear.predict(x_test)
    print metrics.classification_report(y_test, y_pred)
    return best_model_rbf, best_model_linear


def read_text_data(filename):
    # function that takes text file as input and returns the
    #vector with just one flattened array
    data = []
    for line in open(filename):
        line = line.rstrip('\r\n')
        data.extend((line[i]) for i in xrange(0, len(line)))
    data = np.array(data, dtype=float)
    return data


def prediction(filename, model):
    # function that takes input as filename and model and
    #returns the classification result
    arr = read_text_data(filename)
    print "The classification for given file is", model.predict(arr)


def  reducing_features(features):
    #input parameters are the total features in the model
    #and returns the reduced features by removing the features
    #which are constant over time
    cols_to_include = []
    cols = len(features[0])
    rows = len(features)
    for col in range(0, cols):
        for row in range(0, rows):
            if int(features[row][col]):
                cols_to_include.append(col)
            break
    reduced_features = []

    for row in range(rows):
        one_row = []
        for col in cols_to_include:
            one_row.append(features[row][col])
        reduced_features.append(one_row)
    return reduced_features


def main():
    graphs_parameters_svm('2d_example.csv')
    features, target = read_data('letters_training.csv')
    best_model_rbf, best_model_linear = svm_model(features, target)
    prediction('9_54.txt', best_model_rbf)
    reduced_features = question2c(features)
    print "Results after reducing features"
    reducing_features(reduced_features, target)
main()