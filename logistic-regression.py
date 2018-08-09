'''
    This is code for a basic Logistic Regression model, implemented from scratch (ie. no SK-Learn).

    HOW TO USE:
        - Compute weights for a Logistic Regression model using lr_optimized_weights on your training data.
        - Use lr_prediction to classify.
        - Use lr_predic_accuracy for test accuracy.

'''
import numpy as np
from autograd.scipy.misc import logsumexp


def softmax(c, x, w):
    '''Return a 10 X 784 matrix where element ci represents softmax value p(c|x_i, w).
    '''
    wc_x = np.exp(np.dot(w[c], x))
    denom = np.matmul(w, x)
    denom = np.exp(logsumexp(denom))
    if denom == 0:
        return 0
    return wc_x/denom


def softmax_grad(c, x, w):
    '''Return a 10 element vector where element i represents the derivative of softmax(c, x, w)
     wrt w_i '''
    return np.matrix([_softmax_grad(c_prime, c, x, w) for c_prime in range(10)])

def _softmax_grad(c_prime, c, x, w):
    s = softmax(c_prime, x, w)
    if c_prime == c:
        res = x - np.multiply(x, s)
        return res
    else:
        return (-1)*np.multiply(x, s)


def lr_optimized_weights(points, l, num_epochs):
    '''Return a 10 X 784 matrix, representing weights optimized by num_epochs of vanilla gradient descent
    over points with softmax gradient and step size l.'''
    w = np.zeros((10, 784))
    for t in range(num_epochs):
        for point in points:
            w+= l*softmax_grad(np.argmax(point), point, w)
    return w

def lr_predic_likelihood(c, x, w):
    '''Return the predictive log likelihood for point x, class c and weights w. '''
    return np.log(softmax(c, x, w))

def lr_avg_predic_likelihood(points, labels, w):
    '''Return the logistic regression average predictive log-likelihood for labels given points and weights w'''
    like = 0
    for i, point in enumerate(points):
        like += lr_predic_likelihood(np.argmax(labels[i]), point, w)
    return like/len(points)


def lr_prediction(x, w):
    '''Return arg_max c for p(c|x, w)'''
    likelihoods = np.array([lr_predic_likelihood(c, x, w) for c in range(10)])
    return np.argmax(likelihoods)

def lr_predic_accuracy(points, labels, w):
    '''Return the predictive accuracy for points, using logistic regression with weights w. '''
    predics = []
    for point in points:
        predics.append(lr_prediction(point, w))
    correct = 0
    for i, predic in enumerate(predics):
        if predic == np.argmax(labels[i]):
            correct += 1
    return correct/len(points)