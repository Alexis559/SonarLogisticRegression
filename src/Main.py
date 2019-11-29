#!/usr/bin/python3
import pandas as pd

from sklearn.model_selection import train_test_split

# Naive Bayes Multinomial Method
from sklearn.naive_bayes import MultinomialNB

# Logistic Regression Method
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

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
    mines_model = LogisticRegression(solver='liblinear')
    mines_model.fit(features_train, mines_train)
    return mines_model.predict(features_test)


# Naive Bayes Multinomial Method
def naiveBayesMethod(features_train, features_test, mines_train):
    # Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(features_train, mines_train)
    return classifier.predict(features_test)



if __name__ == "__main__":
    print('[Sonar] Loading data...')
    data = prepare_dataset('../data/sonar_data.csv')
    print('[Sonar] Data loaded.')

    # Split the dataset
    features_train, features_test, mines_train, mines_test = split_dataset(data)

    print("[Sonar] Size of train set:", mines_train.shape[0])
    print("[Sonar] Size of test set:", mines_test.shape[0])

    print("\n")
    # Doing preditions Neural network
    predictions = neuralNetwork(features_train, features_test, mines_train)

    # Print metrics
    print("[Sonar] ====== NEURAL NETWORK ======")
    print("[Sonar][NN] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][NN] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][NN] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][NN] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][NN] ROC:", roc_auc_score(mines_test.values, predictions))

    print("\n")

    # Doing preditions LR
    predictions = logistic_regression(features_train, features_test, mines_train)

    # Print metrics
    print("[Sonar] ====== LOGISTIC REGRESSION ======")
    print("[Sonar][LR] Predictions:", predictions)
    print("[Sonar][LR] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][LR] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][LR] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][LR] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][LR] ROC:", roc_auc_score(mines_test.values, predictions))

    print("\n")
    # Doing preditions NB
    predictions = naiveBayesMethod(features_train, features_test, mines_train)

    # Print metrics
    print("[Sonar] ====== NAIVE BAYES ======")
    print("[Sonar][NB] Predictions:", predictions)
    print("[Sonar][NB] Confusion matrix:")
    print(confusion_matrix(mines_test.values, predictions))
    print("[Sonar][NB] Precision:", precision_score(mines_test.values, predictions))
    print("[Sonar][NB] Recall:", recall_score(mines_test.values, predictions))
    print("[Sonar][NB] Accuracy:", accuracy_score(mines_test.values, predictions))
    print("[Sonar][NB] ROC:", roc_auc_score(mines_test.values, predictions))
