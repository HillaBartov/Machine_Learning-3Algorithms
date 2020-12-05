import sys
import numpy as np


def normalization():
    # min-max normalization
    max_features_values = np.amax(train_x, axis=0)
    min_features_values = np.amin(train_x, axis=0)
    for i in range(train_size):
        for j in range(11):
            train_x[i][j] = (train_x[i][j] - min_features_values[j]) / (max_features_values[j] - min_features_values[j])
            if i < test_size:
                test_x[i][j] = (test_x[i][j] - min_features_values[j]) / (
                        max_features_values[j] - min_features_values[j])


def knn():
    k = 7
    test_y = []
    # find the distance for each test x from every train x
    for x_test in test_x:
        dist = []
        classes = {"0": 0, "1": 0, "2": 0}
        # save distance and matching label in a pair
        for train in range(train_size):
            dist.append([np.linalg.norm(x_test - train_x[train]), train_y[train]])
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


def perceptron():
    w = np.array([np.zeros(13), np.zeros(13), np.zeros(13)])
    learning_rate = 0.2
    epochs = 20
    test_y = []
    for e in range(epochs):
        # shuffle train data every epoch
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = train_x[s]
        train_y_shuffled = train_y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            y_hat = np.argmax(np.dot(w, x_i))
            if y_hat != int(y_i):
                w[int(y_i), :] = w[int(y_i), :] + learning_rate * x_i
                w[y_hat, :] = w[y_hat, :] - learning_rate * x_i
    # use the trained weight vectors to assign best label prediction to current x example
    for x in range(test_size):
        test_y.append(np.argmax(np.dot(w, test_x[x])))
    return test_y


def pa():
    w = np.array([np.random.rand(13), np.random.rand(13), np.random.rand(13)])
    epochs = 1
    test_y = []
    for e in range(epochs):
        # shuffle train data every epoch
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x_shuffled = train_x[s]
        train_y_shuffled = train_y[s]
        # predict and update
        for x_i, y_i in zip(train_x_shuffled, train_y_shuffled):
            w_out = np.delete(w, int(y_i), 0)
            y_hat = np.argmax(np.dot(w_out, x_i))
            # get the correct label by it's place in original w
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
    for x in range(test_size):
        test_y.append(np.argmax(np.dot(w, test_x[x])))
    return test_y


training_examples, training_labels, testing_examples = sys.argv[1], sys.argv[2], sys.argv[3]
train_x = np.loadtxt(training_examples, delimiter=',', converters={11: lambda label: 1 if label == b'R' else 0})
train_y = np.loadtxt(training_labels)
test_x = np.loadtxt(testing_examples, delimiter=',', converters={11: lambda label: 1 if label == b'R' else 0})
train_size = len(train_x)
test_size = len(test_x)
# create normalized data for the training(x) and testing(test) sets
normalization()
# knn algorithm
knn_yhat = np.array(knn())
# add bias for perceptron and pa algorithms
train_x = np.c_[train_x, np.ones(train_size)]
test_x = np.c_[test_x, np.ones(test_size)]
train_size = len(train_x)
test_size = len(test_x)
perceptron_yhat = np.array(perceptron())
pa_yhat = np.array(pa())
labels_num = pa_yhat.size
for l in range(labels_num):
    print(f"knn: {knn_yhat[l]}, perceptron: {perceptron_yhat[l]}, pa: {pa_yhat[l]}")
