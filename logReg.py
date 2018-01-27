'''
Implement the incomplete functions in the file.

Arguments:
    None
Returns:
    None
'''
import numpy as np
from read_dataset import mnist
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def sigmoid(scores):
    '''
    calculates the sigmoid of scores
    Inputs: 
        scores array
    Returns:
        sigmoid of scores
    '''
    sig = 1/(1 + np.exp(-scores))
    return sig

def step(X, Y, w, b):
    '''
    Implements cost and gradients for the logistic regression with one batch of data
    Inputs:
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
    Returns:
        cost = cost of the batch
        gradients = dictionary of gradients dw and db
    '''
    #print("Shape of W = ",w.shape)
    #print("Shape of X = ",X.shape)
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
	
    #print("Shape of A = ",A.shape)
    # compute the gradients and cost 
    m = X.shape[1]  # number of samples in the batch
    cost = - 1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum(A-Y)
    
    #print("Shape of dw = ",dw.shape())
    gradients = {"dw": dw,
                 "db": db}
    return cost, gradients

def optimizer(X, Y, w, b, learning_rate, num_iterations):
    '''
    Implements gradient descent and updates w and b
    Inputs: 
        X = (n,m) matrix
        Y = (1,m) matrix of labels
        w = (n,1) matrix
        b = scalar
        learning_rate = rate at which the gradient is updated
        num_iterations = total number of batches for gradient descent
    Returns:
        parameters = dictionary containing w and b
        gradients = dictionary contains gradeints dw and db
        costs = array of costs over the training 
    '''
    costs = []
    # update weights by gradient descent
    for ii in range(num_iterations):
        cost, gradients = step(X, Y, w, b)
        dw = gradients["dw"]
        db = gradients["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if ii % 10 == 0:
            costs.append(cost)
        if ii % 100 == 0:
            print("Cost at iteration %i is: %f" %(ii, cost))
 
    parameters = {"w": w, "b": b}
    return parameters, gradients, costs

def classify(X, w, b):
    '''
    Outputs the predictions for X

    Inputs: 
        X = (n,m) matrix
        w = (n,1) matrix
        b = scalar

    Returns:
        YPred = (1,m) matrix of predictions
    '''
    scores = np.dot(w.T, X) + b
    A = sigmoid(scores)
    YPred = np.zeros((1,X.shape[1]))
    
    for i in range(A.shape[0]):
        if A[0,i] >= 0.5:
            YPred[0,i] = 1
        else:
            YPred[0,i] = 0 

    return YPred

    
def main():
    # getting the sbuset dataset from MNIST
    train_data, train_label, test_data, test_label = mnist()

    # initialize learning rate and num_iterations
    learning_rate = 0.1
    num_iterations = 2000

    # initialize w as array (d,1) and b as a scalar
    w = np.zeros((train_data.shape[0],1))
    b = 0

    # learning the weights by using optimize function
    parameters, gradients, costs = optimizer(train_data, \
                    train_label, w, b, learning_rate, num_iterations)
    w = parameters["w"]
    b = parameters["b"]
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data,w,b)
    test_Pred = classify(test_data,w,b)
    trAcc = 100 - np.mean(np.abs(train_Pred - train_label)*100)
    teAcc = 100 - np.mean(np.abs(test_Pred - test_label)*100)
    print("Accuracy for training set is {} %".format(trAcc))
    print("Accuracy for testing set is {} %".format(teAcc))
    plt.plot(costs)
    plt.xlabel('error')
    plt.ylabel('iterations')
    plt.savefig('error_plot.png')

if __name__ == "__main__":
    main()
