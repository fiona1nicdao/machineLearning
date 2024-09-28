import numpy as np  # linear algebra

class Perceptron(object):
    """
    Perceptron classifier algorithm 
    Parameters 
    ----------
    learning_rate : float
        the learning rate - value between 0.0 and 1.0 from our choosing 
    epoch : int
        number of passes over the training dataset
    random_starter_weight : int  ???? do I need or start at zero ????? or pick a number 
        random number generator seed for random weight initialization 
    
    Attributes 
    ----------
    weights : one dimentation array 
        weights after weight update
    errors_ : list 
        number of misclassificatoin (updates) in each epoch. 
    """
    def __init__ (self, learning_rate=0.01, epoch=40, random_starter_weight=1):
        self.learning_rate = learning_rate # default learning rate is 0.01
        self.epoch = epoch # default epoch is 40 passes 
        self.random_starter_weight = random_starter_weight#random starting weight the features is 1 before randomized 
    
    def fit(self, X, y) :
        """ weight update for the training data 
        Paramters 
        ---------
         X : {array-list} , shape = [n_examples, n_features]
            X is the training vector with
                n_example is the number of examples 
                n_features is the number of features 
            y : array-list, shape = [n_examples]
                target values (also known as the expected class labels )
        """
        # randomized the weights 
        random_generator = np.random.RandomState(self.random_starter_weight)
        #weights 
        self.weights = random_generator.normal(loc=0.0, scale=0.1, size=1 + X.shape[1])
        # errors after each epoch (pass through of the dataset)
        self.errors_ = []

        # updating all the weights after the set number of epoch choosen
        for i in range(self.epoch):
            error_count = 0
            # updating all the weights in one epoch
            for xi, expected_y in zip(X,y):
                # weight update rule 
                update = self.learning_rate * (expected_y - self.predict(xi))
                # update the value of the weights with the update value 
                self.weights[1:] += update * xi 
                self.weights[0] += update
                # increase the error count if the update value is not zero 
                # which means the model got the class label correct 
                error_count += int(update != 0.0)
            self.errors_.append(error_count)
        return self

    def net_input(self, X):
        # calculate net input by taking the dot product of the features and the weights
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    def predict(self, X):
        # return predicted class label using the unit step function 
        # if the net input in greater than or equal to zero then the predicted class label is one 
        # else the predicted class label is negative one 
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def accuracy(self,y):
        percent_misclassified = self.errors_[-1] / len(y)
        return (1 - percent_misclassified) * 100