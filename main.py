import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit

from metrics import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from AdalineAlgo import AdalineAlgo
import warnings
warnings.filterwarnings("ignore")


def model(x, y):
    # split to 66% - 33%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    # start the time
    start = timeit.default_timer()

    # R - recurre =1, N- nonrecurre= -1 
    y_train = np.where(y_train == 'N', -1, 1)
    y_test = np.where(y_test == 'N', -1, 1)

    # train Adaline
    model = AdalineAlgo()
    new, costs = model.train(x_train, y_train)
    y_pred = model.predict(x_train)

    # Plot the training error
    plt.plot(range(1, len(costs) + 1), costs, color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.show()

    #  time takes to train in seconds
    stop = timeit.default_timer()
    finish_time = stop - start

    print('Train')
    accuracy_train = accuracy_score(y_train, y_pred) * 100
    print('Accuracy:  %.2f' % accuracy_train, '%')
    print('Train finished in : %.2f' % finish_time, 'sec')

    # test Adaline
    print('Test')
    y_new_pred = model.predict(x_test)

    # time takes to test  in seconds
    newStop = timeit.default_timer()
    finish2_time = newStop - start

    accuracy_test = accuracy_score(y_test, y_new_pred) * 100
    print('Accuracy:  %.2f' % accuracy_test, '%')
    print('Train finished in : %.2f' % finish2_time, 'sec')

    # create the model
    knn = KNeighborsClassifier(n_neighbors=1)

    # Enter the data training to the mode
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    # print(metrics.accuracy_score(y_test, y_pred))

    # using cross validation to get the best percentage of success

    k_range = list(range(1, 41))
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', p=1)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print(k_scores)
    # graph of the value of k (x) and the cross validation accuracy
    plt.plot(k_range, k_scores)
    # names of x and y
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross validation Accuracy')
    plt.show()
    
    #before cross validation     
    CM = confusion_matrix(y_pred, y_test)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    print("before cross validation")
    print("TRUE NEGATIVE (TN):", TN)
    print("FALSE NEGATIVE (FN):", FN)
    print("TRUE POSITIVE (TP):", TP)
    print("FALSE POSITIVE (FP):", FP)
    nn = KNeighborsClassifier(n_neighbors=17, weights='distance', p=1)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    # accuracy = metrics.accuracy_score(y_test, y_pred)
    # print('% knn accuracy:', accuracy * 100)
    
    #after cross validation     
    CM = confusion_matrix(y_pred, y_test)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    print("after cross validation")
    print("TRUE NEGATIVE (TN):", TN)
    print("FALSE NEGATIVE (FN):", FN)
    print("TRUE POSITIVE (TP):", TP)
    print("FALSE POSITIVE (FP):", FP)
    return accuracy_test

# main
if __name__ == '__main__':
    # https://realpython.com/python-data-cleaning-numpy-pandas/
    # clean our data
    # set our x and y for the train and test models
    clean = pd.read_csv('wpbc.data', na_values="?", sep=",")
    data = clean.copy()
    data = data.dropna()
    y = data['N']
    X = data.drop(data[['N']], 1)

    finish = model(X, y)



