__author__ = 'deeksha'

# Logistic regression model applied to hand written digits dataset.
from sklearn import metrics
import csv
from sklearn import linear_model
from sklearn.cross_validation import ShuffleSplit
import pylab as plt
import numpy as np
from scipy.stats import sem
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import matplotlib.gridspec as gridspec

# Load the data from the csv file
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
            X.append(row[0:len(row)-1])
            Y.append(row[len(row)-1])
    col = len(X)
    X1 = []
    for i in range(col-1):
        X1.append([float(x) for x in X[i]])
    Y = [float(y) for y in Y if y]
    X1 = np.asarray(X1)
    Y1 = np.asarray(Y)
    Y1 = np.array(Y1).astype(np.float)
    return X1, Y1


def grid_search(model, x_train, y_train):
    #input parameters are the model and data and
    #it returns the best model by searching over common values of C and gamma
    #set the C and gamma range over which grid search will be applied
    C_range = 10.0 ** np.arange(-3, 4)
    #gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(C=C_range)
    #stratified k folds of data will be created
    cv = StratifiedKFold(y=y_train, n_folds=3)
    #below function searches over the entire space of param_grid
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(x_train, y_train)
    return grid.best_estimator_


def model_create(X, Y, C):
    #input parameters are features and target variables.
    #returns the logistic model
    ## divide the dataset into training and testing set
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.15, random_state = 0)
    #creating logistic classifiers
    logreg = linear_model.LogisticRegression(C= C)
    # train the classifier
    logreg.fit(x_train, y_train)
    return logreg, x_train, x_test, y_train, y_test


def best_model(model, x_train, x_test, y_train, y_test):
    y_pred = model.predict(x_test)
    print metrics.classification_report(y_test, y_pred)
    print "The accuracy is of the model for C=1: ", model.score(x_test, y_test)
    best_model = grid_search(model, x_train, y_train)
    y_pred = best_model.predict(x_test)
    print metrics.classification_report(y_test, y_pred)
    print "Accuracy of Logistic model after GridSearch", best_model.score(x_test, y_test)
    return best_model


def read_text_data(filename):
    #function that takes text file as input and returns the
    #vector with just one flattened array
    data = []
    for line in open(filename):
        line = line.rstrip('\r\n')
        data.extend((line[i]) for i in xrange(0, len(line)))
    data = np.array(data, dtype=float)
    return data


def prediction_class(filename, model):
    # function that takes input as filename and model and
    #returns the classification result
    arr = read_text_data(filename)
    print "The classification for given file is", model.predict(arr)


def plot_learning_curve(X, Y):
    C = [10**x for x in range(-4,4)]
    k = 0
    for c in C:
        k += 1
        model, x_train, x_test, y_train, y_test = model_create(X, Y, c)
        plt.subplot(len(C),2, k, aspect="equal")
        x = np.array(X, dtype=float)
        y = np.array(Y, dtype=int)
        train_sizes = np.logspace(6,7.4,9, base = 2).astype(np.int)
        n_iter = 20
        n_samples = x.shape[0]
        train_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)
        test_scores = np.zeros((train_sizes.shape[0], n_iter), dtype=np.float)
        for i, train_size in enumerate(train_sizes):
            cv = ShuffleSplit(n_samples, n_iter=n_iter, train_size=np.max(train_sizes), random_state = None)
            for j, (train, test) in enumerate(cv):
                lr = model.fit(x[train], y[train])
                train_scores[i, j] = lr.score(x[train], y[train])
                test_scores[i, j] = lr.score(x[test], y[test])
        # We can now plot the mean scores with error bars that reflect the standard errors of the means:
        mean_train = np.mean(train_scores, axis=1)
        confidence = sem(train_scores, axis=1) * 2
        plt.fill_between(train_sizes, mean_train - confidence, mean_train + confidence, color='b', alpha=.2)
        plt.plot(train_sizes, mean_train, 'o-k', c='b', label='Train score')
        mean_test = np.mean(test_scores, axis=1)
        confidence = sem(test_scores, axis=1) * 2
        plt.fill_between(train_sizes, mean_test - confidence, mean_test + confidence, color='g', alpha=.2)
        plt.plot(train_sizes, mean_test, 'o-k', c='g', label='Test score')
        plt.axis('tight')
        plt.xticks(())
        plt.yticks(())
        plt.title(" C 10^%d" % np.log10(c),size='medium')
    plt.tight_layout()
    plt.suptitle("Effect of C with L1 regularization")
    plt.show()


def model_create_regularization_l1(X, Y, C):
    #input parameters are features and target variables.
    #returns the training and test scores of logistic model with L1 regularization
    # divide the data set into training and testing set
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.7, random_state = 0)
    #creating two SVM classifiers
    logreg = linear_model.LogisticRegression(C=C, penalty='l1')
    # train the classifier
    logreg.fit(x_train, y_train)
    training_score = logreg.score(x_train, y_train)
    test_score = logreg.score(x_test, y_test)
    return training_score, test_score


def plot_graph_l1(training_scores,test_scores):
    #plots the graph showing the effect of regularization on score
    C = [ pow(10,i) for i in range(-2,5)]
    plt.plot(C, training_scores, label="Training Scores")
    plt.plot(C, test_scores, label="Test Scores")
    plt.xlabel('C')
    plt.ylabel('Score')
    plt.xscale('log')
    plt.ylim((None, 1.5))  # The best possible score is 1.0
    plt.legend(loc='best')
    plt.title("Effect of L1 regularization on Score")
    plt.show()


def main():
    features, target = read_data('letters_training.csv')
    model, x_train, x_test, y_train, y_test = model_create(features, target, C=1)
    model_best = best_model(model, x_train, x_test, y_train, y_test)
    prediction_class('9_54.txt', model_best)
    plot_learning_curve(features, target)
    training_scores = []
    test_scores = []
    C = [pow(10,i) for i in range(-2,5)]
    for c in C:
        training_score, test_score = model_create_regularization_l1(features, target, C=c)
        training_scores.append(training_score)
        test_scores.append(test_score)
    plot_graph_l1(training_scores, test_scores)

main()