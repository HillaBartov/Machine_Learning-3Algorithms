import sys
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics


def normalization():
    # min-max normalization
    max_features_values = np.amax(train_x, axis=0)
    min_features_values = np.amin(train_x, axis=0)
    train_size = int(train_x.size / 12)
    test_size = int(test_x.size / 12)
    for i in range(train_size):
        for j in range(11):
            train_x[i][j] = (train_x[i][j] - min_features_values[j]) / (max_features_values[j] - min_features_values[j])
            if i < test_size:
                test_x[i][j] = (test_x[i][j] - min_features_values[j]) / (
                        max_features_values[j] - min_features_values[j])


def knn(testset, trainset, trainy):
    k = 7
    test_y = []
    # find the distance for each test x from every train x
    for x_test in testset:
        dist = []
        classes = {"0": 0, "1": 0, "2": 0}
        size = int(trainset.size / 12)
        # save distance and matching label in a pair
        for train in range(size):
            dist.append([np.linalg.norm(x_test - trainset[train]), trainy[train]])
        dist_np = np.array(dist)
        # find k nearest neighbors
        sorted_array = dist_np[np.argsort(dist_np[:, 0])]
        nearest_neighbors = sorted_array[:k]
        # find the number of occurrences for each class
        for line in range(k):
            classes[str(int(nearest_neighbors[line][1]))] += 1
        # predict the majority class and assign to current x example
        max_class = max(classes, key=classes.get)
        test_y.append(int(max_class))
    return test_y


def perceptron(X, Y):
    w = np.array([np.random.rand(13), np.random.rand(13), np.random.rand(13)])
    eta = 0.2
    epochs = 20
    test_y = []
    for e in range(epochs):
        # shuffle train data every epoch
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = X[s]
        train_y_shuffled = Y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != int(y_i):
                w[int(y_i), :] = w[int(y_i), :] + eta * x_i
                w[y_hat, :] = w[y_hat, :] - eta * x_i
    for y in range(int(X_test.size / 13)):
        test_y.append(np.argmax(np.dot(w, X_test[y])))
    return test_y


def pa(X, Y):
    w = np.array([np.random.rand(13), np.random.rand(13), np.random.rand(13)])
    epochs = 1
    test_y = []
    for e in range(epochs):
        # shuffle train data every epoch
        s = np.arange(X.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = X[s]
        train_y_shuffled = Y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            w_out = np.delete(w, int(y_i), 0)
            y_hat = np.argmax(np.dot(w_out, x_i))
            if y_i == 0 or (y_i == 1 and y_hat == 1):
                y_hat += 1
            loss = max(0, 1 - np.dot(w[int(y_i)], x_i) + np.dot(w[y_hat], x_i))
            if loss:
                tau = 1
                denominator = 2 * (np.linalg.norm(x_i) ** 2)
                # when denominator is not 0 change initialization by formula
                if denominator:
                    tau = loss / denominator
                    # update w
                w[int(y_i), :] = w[int(y_i), :] + tau * x_i
                w[y_hat, :] = w[y_hat, :] - tau * x_i
    # use the trained weight vectors to assign best label prediction to current x example
    for x in range(int(X_test.size / 13)):
        test_y.append(np.argmax(np.dot(w, X_test[x])))
    return test_y


training_examples, training_labels, testing_examples = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = np.loadtxt(training_examples, delimiter=',', converters={11: lambda label: 1 if label == b'R' else 0})
train_y = np.loadtxt(training_labels)
test_x = np.loadtxt(testing_examples, delimiter=',', converters={11: lambda label: 1 if label == b'R' else 0})
train_size = int(train_x.size / 12)
test_size = int(test_x.size / 12)
# create normalized data for the training(x) and testing(test) sets
normalization()
train_x = np.c_[train_x, np.ones(train_size)]
test_x = np.c_[test_x, np.ones(test_size)]
kf = KFold(n_splits=10)
kf.get_n_splits(train_x)
# knn algorithm
average = 0
print("PERCEPTRON ACCURACY:")
for train_index, test_index in kf.split(train_x):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y[train_index], train_y[test_index]
    perceptron_yhat = np.array(perceptron(X_train, y_train))
    average += metrics.accuracy_score(y_test, perceptron_yhat)
    print(metrics.accuracy_score(y_test, perceptron_yhat))

print("AVERAGE:", average / 10)
