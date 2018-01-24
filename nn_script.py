import numpy as np
import pandas as pd
import sklearn,pickle
import sklearn.datasets

"""
This is the implementation of a 3 layers NN using numpy array
Number of input layers: 2
Number of classes 2
"""

learning_rate = 0.0001
reg_paramter = 0.0001
num_input = 2
num_classes = 2
epoch = 2000
num_hidden_layer_nodes = 6

def data_set():
    """
    This fun generates sample X,y dataset
    X: input array
    Y: output array [0 or 1]
    """
    X,y = sklearn.datasets.make_moons(200, noise=0.20)
    return X,y


def loss(model):
    X, y = data_set()
    m = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(m), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_paramter / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / m * data_loss


def train_nn(nn_hdim):
    # nn_hdim - number node in hidden layers 
    X,y = data_set()
    m = len(X)
    
    # Initialize the parameters to random values. We need to learn these.
    W1 = np.random.randn(num_input, nn_hdim) / np.sqrt(num_input)
    b1 = np.zeros((1, nn_hdim))

    W2 = np.random.randn(nn_hdim, num_classes) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, num_classes))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, epoch):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(m), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_paramter * W2
        dW1 += reg_paramter * W1

        # Gradient descent parameter update
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Print the loss.
        if i % 200 == 0:
            print("Loss after iteration %i: %f" % (i, loss(model)))
    return model

def save_model(model):
    pickle.dump(model, open("static/model.sav", 'wb'))
    return "Saved successfuly"

if __name__ == "__main__":
    #train_nn 
    model = train_nn(num_hidden_layer_nodes)

    #Save nn_model
    save_model(model)


