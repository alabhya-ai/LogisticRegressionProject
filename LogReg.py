'''LogisticRegression by Alabs (https://www.instagram.com/aealabs/)'''

class LogisticRegression(object):
  '''Run logistic Regression.
  Initialize an object and call the 'apply' function in this class.
  Use the 'pred' method to predict with given X or 'pred_prob' to output the probabilities.
  Use 'regularize' method to regularized found theta.

  -Numpy should be imported as np.
  -Both dependent and independent variables need to be 'np.array' class objects.
  -No need to feature scale your data since this class does that for you.
  '''

  def __init__(self, independent_var, dependent_var):
    self.m =  dependent_var.shape[0] # no. of data points
    self.y = dependent_var # dependent variable
    self.X = (independent_var - independent_var.mean()) / independent_var.std() # standardization for independent variable
    self.X = np.hstack((np.ones((self.m, 1)), self.X)) # adding column of ones in independent variable
    self.theta = np.zeros((self.X.shape[1], 1)) # setting initial theta as zeros
    self.n = self.X.shape[1] # no. of features

  def gd(self, num_of_iterations, alpha, hist):
    '''runs gradient descent and sets self.theta to best theta'''

    if hist:
      l = np.ones((1, num_of_iterations)) # to store loss history
      for i in range(num_of_iterations):
        h = 1.0 / (1 + np.exp(-np.dot(self.X, self.theta))) # hypothesis (sigmoid)
        grad = np.dot(self.X.T, np.array([h[i] - self.y[i] for i in range(self.m)])) / self.m # gradient
        self.theta = self.theta - (alpha * grad) # theta update rule
        l[0, i] = np.dot(self.y.T, np.log(h)) + np.dot((1 - self.y).T, np.log(1 - h)) # loss

      return -l / self.m

    else:
      for i in range(num_of_iterations):
        h = 1.0 / (1 + np.exp(-np.dot(self.X, self.theta))) # hypothesis (sigmoid)
        self.theta = self.theta - ((alpha / self.m) * np.dot(self.X.T, np.array([h[i] - self.y[i] for i in range(self.m)]))) # theta update rule

  def sgd(self, num_of_iterations, alpha, hist):
    '''runs stochastic gradient descent and returns best theta'''

    if hist:
      l = np.ones((1, num_of_iterations)) # to store loss history

      for i in range(num_of_iterations):
        r = np.random.randint(0, self.m)
        Xr, Yr = self.X[r, :], self.y[r] # random data point
        Hr = 1.0 / (1 + np.exp(-np.dot(Xr, self.theta))) # hypothesis given by the random data point (sigmoid)
        self.theta = self.theta - ((alpha * (Xr.reshape(self.n, 1) * (Hr - Yr))) / self.m) # theta update rule
        l[0, i] = (Yr * np.log(Hr)) + ((1 - Yr) * np.log(1 - Hr)) # loss

      return -l

    else:
      for i in range(num_of_iterations):
        r = np.random.randint(0, self.m)
        Xr, Yr = self.X[r, :], self.y[r] # random data point
        Hr = 1.0 / (1 + np.exp(-np.dot(Xr, self.theta))) # hypothesis given by the random data point (sigmoid)
        self.theta = self.theta - ((alpha * (Xr.reshape(self.n, 1) * (Hr - Yr))) / self.m) # theta update rule

  def pred(self, X, threshold):
    '''returns predictions made by the model according to the given threshold on given X'''
    X = (X - X.mean()) / X.std()# standardization
    z = self.theta[0, 0] + np.dot(X, self.theta[1:]) # value to be passed to sigmoid
    h = 1.0 / (1 + np.exp(-z)) # hypothesis

    return np.array([1 if i >= 0.5 else 0 for i in h])

  def pred_prob(self, X):
    '''Returns the probabilities output by the model on given X.'''

    X = (X - X.mean()) / X.std()
    z = self.theta[0, 0] + np.dot(X, self.theta[1:]) # value to be passed to sigmoid

    return 1.0 / (1 + np.exp(-z))

  def regularize(self, lambda_, set_param=True):
    '''returns regularized version of theta found by optimization algorithm.
    parameters:
    lambda_: Set to required value to be used as the regularization coefficient known as lambda
    set_param: default True (sets the original theta values to regularized ones),
               set to False to return regularized theta values
    '''

    if set_param:
      self.theta[1:, 0] = self.theta[1:, 0] * lambda_ / self.m
    else:
      tmp = np.zeros(self.theta.shape)
      tmp[0, 0] = self.theta[0, 0]
      tmp[1:, 0] = self.theta[1:, 0] * lambda_ / self.m
      return tmp

  def loss(self):
    h = 1.0 / (1 + np.exp(-np.dot(self.X, self.theta))) # hypothesis (sigmoid)
    return (np.dot(self.y.T, np.log(h)) + np.dot((1 - self.y).T, np.log(1 - h))) / -self.m

  def score(self, threshold):
    '''Returns the F-1 score for predictions on given X
    parameters:
    threshold: threshold for predicting whether data point lies in class 0 or 1
    '''

    s = 1.0 / (1 + np.exp(-np.dot(self.X, self.theta))) # probabilities
    p = np.array([1 if i >= 0.5 else 0 for i in s]) # predictions
    tp, tn, fp, fn = 0, 0, 0, 0 # true positive, true negative, false positive, false negative

    for i in range(self.y.shape[0]):
      if self.y[i] == p[i]:
        if self.y[i] == 1:
          tp += 1
        else:
          tn += 1
      else:
        if self.y[i] == 1:
          fp += 1
        else:
          fn += 1

    if not (tp == 0 or fp == 0 or fn == 0):
      p = tp / (tp + fp) # precision score
      r = tp / (tp + fn) # recall score
      
      return 2 * p * r / (p + r)
    else:
      return 0

  def apply(self, ret=(1, 6000, 0.1), history=False, cost=False):
    '''
    Applies logistic regression on given data

    parameters:
    ret: specifies how the algorithm should be run, by default it is set to (1, 6000, 0.1)
         set ret to a tuple of the following:
         (0 for gradient descent or 1 for stochastic gradient descent), no. of iterations, alpha (learning rate)
    history: default False, if set to True, returns array of the convergence of theta in optimization algorithm
    cost: default False, if set to True, returns the loss at optimum theta

    returns:
    loss history (if history is True), loss at optimum theta (if cost is True)
    '''

    if ret[0]:
      if cost:
        return self.sgd(num_of_iterations=ret[1], alpha=ret[2], hist=history), self.loss()
      else:
        return self.sgd(num_of_iterations=ret[1], alpha=ret[2], hist=history)
    else:
      if cost:
        return self.gd(num_of_iterations=ret[1], alpha=ret[2], hist=history), self.loss()
      else:
        self.gd(num_of_iterations=ret[1], alpha=ret[2], hist=history)
