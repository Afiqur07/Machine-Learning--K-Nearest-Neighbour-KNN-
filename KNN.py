#
#  Assignment 1
#
#  Group 33:
#  <SM Afiqur Rahman> <smarahman@mun.ca>
#  <Jubaer Ahmed Bhuiyan> <Group Member 1 email>
#  <Group Member 3 name> <Group Member 1 email>

####################################################################################
# Imports
####################################################################################
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################


def classify():
    print('Performing classification...')


def Q1_results():
    print('Generating results for Q1...')

    # Load the required datasets
    train_NC = pd.read_csv("train.sNC.csv", header=None)
    train_DAT = pd.read_csv("train.sDAT.csv", header=None)
    gridPoints = pd.read_csv("2D_grid_points.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load the testing datasets
    test_NC = pd.read_csv("test.sNC.csv", header=None)
    test_DAT = pd.read_csv("test.sDAT.csv", header=None)

    # Combine the two testing datasets and add labels
    test_NC["label"] = 0
    test_DAT["label"] = 1
    test_data = pd.concat([test_NC, test_DAT])

    # Split the testing data into features (X) and labels (y)
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Train the KNN classifier using k=30 and Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=30, metric="euclidean")
    knn.fit(X_train, y_train)

    # Predict on the training data
    y_train_pred = knn.predict(X_train)

    # Calculate the training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on the testing data
    y_test_pred = knn.predict(X_test)

    # Calculate the testing accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # predict on 2D Grid Points
    grid_predict = knn.predict(gridPoints)

    # generating classification boundry
    x_min, x_max = gridPoints.iloc[:, 0].min(
    ) - 1, gridPoints.iloc[:, 0].max() + 1
    y_min, y_max = gridPoints.iloc[:, 1].min(
    ) - 1, gridPoints.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, levels=5, cmap='viridis', extend='both')

    # Plot the classification boundary
    colors = ['green' if l == 0 else 'blue' for l in y_train]
    gridColor = ['black']
    plt.scatter(gridPoints.iloc[:, 0], gridPoints.iloc[:, 1],
                c=gridColor, marker='.', label="Grid Points")
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                c=colors, marker='o', label="Train")
    colors = ['green' if l == 0 else 'blue' for l in y_test]
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                c=colors, marker='+', label="Test")
    # plt.contourf(grid_predict, alpha=0.3, cmap='coolwarm')
    plt.legend()
    plt.title("Train Error Rate: {:.2f}% \nTest Error Rate: {:.2f}%".format(
        (1-train_accuracy)*100, (1-test_accuracy)*100))
    plt.show()


def Q2_results():
    print('Generating results for Q2...')
    # Load the required datasets
    train_NC = pd.read_csv("train.sNC.csv", header=None)
    train_DAT = pd.read_csv("train.sDAT.csv", header=None)
    gridPoints = pd.read_csv("2D_grid_points.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load the testing datasets
    test_NC = pd.read_csv("test.sNC.csv", header=None)
    test_DAT = pd.read_csv("test.sDAT.csv", header=None)

    # Combine the two testing datasets and add labels
    test_NC["label"] = 0
    test_DAT["label"] = 1
    test_data = pd.concat([test_NC, test_DAT])

    # Split the testing data into features (X) and labels (y)
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Train the KNN classifier using k=30 and Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=30, metric="manhattan")
    knn.fit(X_train, y_train)

    # Predict on the training data
    y_train_pred = knn.predict(X_train)

    # Calculate the training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on the testing data
    y_test_pred = knn.predict(X_test)

    # Calculate the testing accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # predict on 2D Grid Points
    grid_predict = knn.predict(gridPoints)

    # generating classification boundry
    x_min, x_max = gridPoints.iloc[:, 0].min(
    ) - 1, gridPoints.iloc[:, 0].max() + 1
    y_min, y_max = gridPoints.iloc[:, 1].min(
    ) - 1, gridPoints.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, levels=5, cmap='viridis', extend='both')

    # Plot the classification boundary
    colors = ['green' if l == 0 else 'blue' for l in y_train]
    gridColor = ['black']
    plt.scatter(gridPoints.iloc[:, 0], gridPoints.iloc[:, 1],
                c=gridColor, marker='.', label="Grid Points")
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                c=colors, marker='o', label="Train")
    colors = ['green' if l == 0 else 'blue' for l in y_test]
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                c=colors, marker='+', label="Test")
    # plt.contourf(grid_predict, alpha=0.3, cmap='coolwarm')
    plt.legend()
    plt.title("Train Error Rate: {:.2f}% \nTest Error Rate: {:.2f}%".format(
        (1-train_accuracy)*100, (1-test_accuracy)*100))
    plt.show()


def Q3_results():
    print('Generating results for Q3...')
    # Load the required datasets
    train_NC = pd.read_csv("train.sNC.csv", header=None)
    train_DAT = pd.read_csv("train.sDAT.csv", header=None)
    gridPoints = pd.read_csv("2D_grid_points.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Load the testing datasets
    test_NC = pd.read_csv("test.sNC.csv", header=None)
    test_DAT = pd.read_csv("test.sDAT.csv", header=None)

    # Combine the two testing datasets and add labels
    test_NC["label"] = 0
    test_DAT["label"] = 1
    test_data = pd.concat([test_NC, test_DAT])

    # Split the testing data into features (X) and labels (y)
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Parameter space for 1/k
    ks = np.linspace(0.01, 1.0, num=100)

    train_error_rates = []
    test_error_rates = []

    for k in ks:
        # Train the KNN classifier using k and Euclidean distance
        knn = KNeighborsClassifier(n_neighbors=int(1/k), metric="euclidean")
        knn.fit(X_train, y_train)

        # Predict on the training data
        y_train_pred = knn.predict(X_train)

        # Calculate the training accuracy
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Predict on the testing data
        y_test_pred = knn.predict(X_test)

        # Calculate the testing accuracy
        test_accuracy = accuracy_score(y_test, y_test_pred)

        # Append the error rates to the lists
        train_error_rates.append(1 - train_accuracy)
        test_error_rates.append(1 - test_accuracy)

    # Plot the error rates versus 1/k
    plt.plot(ks, train_error_rates, label="Train")
    plt.plot(ks, test_error_rates, label="Test")
    plt.xscale('log')
    plt.xlabel("1/k")
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()


def diagnoseDAT(Xtest, data_dir):
    # Load the required datasets
    train_NC = pd.read_csv(data_dir + "/train.sNC.csv", header=None)
    train_DAT = pd.read_csv(data_dir + "/train.sDAT.csv", header=None)
    gridPoints = pd.read_csv(data_dir + "/2D_grid_points.csv", header=None)

    # Combine the two training datasets and add labels
    train_NC["label"] = 0
    train_DAT["label"] = 1
    train_data = pd.concat([train_NC, train_DAT])

    # Split the training data into features (X) and labels (y)
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    # Train the KNN classifier using k=30 and Euclidean distance
    knn = KNeighborsClassifier(n_neighbors=30, metric="euclidean")
    knn.fit(X_train, y_train)

    return knn.predict(Xtest)


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
