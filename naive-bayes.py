'''
This is code for a basic Naive Bayes model.
Achieved 89% test accuracy on the MNIST data-set.

Assumes class labels are 1-hot encoded vectors.

HOW TO USE:
    - Fit class-conditional means using map_estimate(train_data, train_labels).
    - Use argmax of nb_predic_likelihood(test_point, class, params) to classify.

'''
import numpy as np

NUM_CLASSES = 10
NUM_FEATURES = 784

def map_estimate(X, y):
    '''Return a NUM_CLASSES X NUM_FEATURES numpy matrix,
    representing MAP estimate of class conditional means fit to X.
    '''
    #create a list of lists of training points per label
    points = [[] for i in range(10)]
    for i in range(len(X)):
        points[np.argmax(y[i])].append(X[i])
    result = []
    for i in range(10):
        row = []
        for d in range(NUM_FEATURES):
            ind = len(points[i])
            num = sum([point[d] for point in points[i]]) + 1
            denom = ind + 2
            row.append(num/denom)
        result.append(np.array(row))
    return np.array(result)

def nb_predic_likelihood(x, c, params):
    '''Return the predictive likelihood of class c given point x and params'''
    theta_c = params[c]
    like = np.dot(x, np.log(theta_c)) + np.dot(np.add(-1*x, 1), np.log(np.add(-1*theta_c, 1)))
    return like + np.log(1/10)

def nb_predic_likelihoods(points, labels, params):
    '''Return a vector of the predictive likelihoods per training data point.'''
    return np.array([nb_predic_likelihood(points[i], np.argmax(labels[i]), params) for i in range(len(points))])

def nb_average_predic_likelihood(points, labels, params):
    '''Return the average predictive likelihood from points and labels, given params.'''
    return np.sum(nb_predic_likelihoods(points,labels,params))/len(points)

def nb_predic_accuracy(points, labels, params):
    '''Return the predictive accuracy for points, given params and labels.'''
    likelihoods = np.matrix([[nb_predic_likelihood(point, c, params) for c in range(10)] for point in points])
    predics = np.argmax(likelihoods, axis=1)
    correct = 0
    for i, c in enumerate(np.argmax(labels, axis=1)):
        if predics[i] == c:
            correct +=1
    return correct/len(labels)