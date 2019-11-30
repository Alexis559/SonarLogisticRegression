#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Naive Bayes Multinomial Method
from sklearn.naive_bayes import MultinomialNB

# Logistic Regression Method
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, roc_curve, auc

# Neural network
from keras import Sequential
from keras.layers import Dense

"""
    [PYTHON 3.6.X]

    [IG5] Alexis SANCHEZ
    Predict mines

"""


def neuralNetwork(features_train, features_test, mines_train):
    classifier = Sequential()

    classifier.add(Dense(300, activation='relu', kernel_initializer='random_normal', input_dim=60))
    classifier.add(Dense(300, activation='relu', kernel_initializer='random_normal')) # Hidden layer
    classifier.add(Dense(300, activation='relu', kernel_initializer='random_normal')) # Hidden layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) # Output Layer

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    classifier.fit(features_train, mines_train, batch_size=32, epochs=300)

    y_pred = classifier.predict(features_test)
    y_pred = (y_pred > 0.5)
    return y_pred


# We get the data from the csv file
def prepare_dataset(path_csv):
    messages = pd.read_csv(path_csv, sep=';', encoding='latin-1')
    return messages


# Split dataset
def split_dataset(dataset):
    dataset_features = dataset.drop(['Class'], axis=1)
    print(dataset_features)
    return train_test_split(dataset_features, dataset['Class'], test_size=0.3, random_state=20)


# Logistic Regression Method
def logistic_regression(features_train, features_test, mines_train):
    mines_model = LogisticRegression(solver='lbfgs', max_iter=200)
    mines_model.fit(features_train, mines_train)
    return mines_model.predict(features_test)


# Naive Bayes Multinomial Method
def naiveBayesMethod(features_train, features_test, mines_train):
    # Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(features_train, mines_train)
    return classifier.predict(features_test)


def printROC(fpr, tpr, roc_auc, name):
    plt.title('Receiver Operating Characteristic ' + name)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    print('[Sonar] Loading data...')
    data = prepare_dataset('../data/sonar_data.csv')
    print('[Sonar] Data loaded.')

    print('[Sonar] Data:')
    print(data.describe())

    # Print a plot to see if the dataset is balanced
    plt.title('Repartition of data')
    plt.bar('Rocks', data['Class'].value_counts()[0])
    plt.bar('Mines', data['Class'].value_counts()[1])
    plt.show()

    # Split the dataset
    features_train, features_test, mines_train, mines_test = split_dataset(data)

    print("[Sonar] Size of train set:", mines_train.shape[0])
    print("[Sonar] Size of test set:", mines_test.shape[0])

    print("\n")
    # Doing preditions Neural network
    predictions = neuralNetwork(features_train, features_test, mines_train)
    
    # Print metrics
    print("[Sonar] ====== NEURAL NETWORK - POSIVITE CLASS ======")
    print("[Sonar][NN] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][NN] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][NN] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][NN] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][NN] ROC:", roc_auc_score(mines_test.values, predictions))
    print("[Sonar][NN] F-measure:", f1_score(mines_test.values, predictions))
    
    fpr, tpr, threshold = roc_curve(mines_test.values, predictions)
    roc_auc = auc(fpr, tpr)

    printROC(fpr, tpr, roc_auc, 'Neural Network')

    print("\n")

    # Doing preditions LR
    predictions = logistic_regression(features_train, features_test, mines_train)

    # Print metrics
    print("[Sonar] ====== LOGISTIC REGRESSION - POSIVITE CLASS ======")
    print("[Sonar][LR] Predictions:", predictions)
    print("[Sonar][LR] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][LR] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][LR] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][LR] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][LR] ROC:", roc_auc_score(mines_test.values, predictions))
    print("[Sonar][LR] F-measure:", f1_score(mines_test.values, predictions))

    fpr, tpr, threshold = roc_curve(mines_test.values, predictions)
    roc_auc = auc(fpr, tpr)

    printROC(fpr, tpr, roc_auc, 'Logistic Regression')

    print("\n")
    # Doing preditions NB
    predictions = naiveBayesMethod(features_train, features_test, mines_train)

    # Print metrics
    print("[Sonar] ====== NAIVE BAYES - POSIVITE CLASS ======")
    print("[Sonar][NB] Predictions:", predictions)
    print("[Sonar][NB] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][NB] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][NB] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][NB] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][NB] ROC:", roc_auc_score(mines_test.values, predictions))
    print("[Sonar][NB] F-measure:", f1_score(mines_test.values, predictions))

    fpr, tpr, threshold = roc_curve(mines_test.values, predictions)
    roc_auc = auc(fpr, tpr)

    printROC(fpr, tpr, roc_auc, 'Naive Bayes')