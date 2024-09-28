import numpy as np  # type: ignore # linear algebra
import matplotlib.pyplot as plt # type: ignore # for scatter plot 
from matplotlib.colors import ListedColormap # type: ignore # for color map 

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
        self.random_starter_weight = random_starter_weight
        #random starting weight the features is 1 before randomized 
    
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
                # expected y minus the predicted y (y - hat)
                diff = expected_y - self.predict(xi)
                # weight update rule 
                update = self.learning_rate * (diff)
                # update the value of the weights with the update value * the feature
                self.weights[1:] += update * xi 
                # update initial weight
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
        # to get the total number of examples, we are using the length of y
        # because assuming every example as a class label

        # percent of misclassified class labels 
        percent_misclassified = self.errors_[-1] / len(y)
        # percept of accurately classified class labels 
        return (1 - percent_misclassified) * 100

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation((self.net_input(X))) >= 0.0, 1,-1)
    
class BaselineAdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        # iterating with random weights and not updating ! 
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            # no updating will occur here! 
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

class Plot(object):
    def plot_decision_regions(X, y, classifer, resolution=0.02) :
        #setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red','blue', 'lightgreen','gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
        x2_min, x2_max = X[:,1].min() - 1, X[:,1].max()+1
        xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                                np.arange(x2_min,x2_max,resolution))
        Z = classifer.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1,xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(),xx1.max())
        plt.ylim(xx2.min(),xx2.max())
        # plot class examples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolor='black')

class TitanicDataset(object):
    def process_data(df):
        # drop useless data : PassengerId , Ticket , Fare , Cabin , Embarked
        df = df.drop(["PassengerId", "Ticket", "Fare", "Cabin", "Embarked","Name","Age"], axis=1)

        df["Survived"] = df["Survived"].map({0 : -1, 1:1})
        # no processing done with Pclass
         
        # sex : male = 0 and female = 1
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

        # parch 
        def parch_separate(passenger):
            parch = passenger
            if( parch > 0) : # if passenger have a parent / children  = 1
                return 1
            else : # else passenger has no parent / children = 0 
                return 0
            
        df["Parch"] = df["Parch"].apply(parch_separate)

        # sibsp
        def sibsp_separate(passenger):
            sibsp = passenger
            if(sibsp > 0): # if passengers has a sibling / spouse = 1
                return 1
            else: # passenger has no parent / children = 0
                return 0

        df["SibSp"] = df["SibSp"].apply(sibsp_separate)

        return df
