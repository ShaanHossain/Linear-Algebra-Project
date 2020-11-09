import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math

class Value:
    def __init__(self,value, x, y):
        self.value = value
        self.x = x
        self.y = y
        self.matches = 0;

    def getValue(self):
        return self.value

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

#Linear Regresssion Method 1
def linearRegression(vector):
    #example for linear regression
    np.random.seed(101)
    tf.set_random_seed(101)

    x = np.linspace(0, len(vector) - 1, len(vector))
    y = vector

    # x = np.linspace(0, 100, 100)
    # y = np.linspace(0, 100, 100)

    # Adding noise to the random linear data
    # x += np.random.uniform(-10, 10, len(x))
    # y += np.random.uniform(-10, 10, len(y))

    n = len(x) # Number of data points

    # Plot of Training Data
    # plt.scatter(x, y)
    # plt.xlabel('x')
    # plt.xlabel('y')
    # plt.title("Training Data")
    # plt.show()

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(np.random.randn(), name = "W")
    b = tf.Variable(np.random.randn(), name = "b")

    learning_rate = 0.01
    training_epochs = 1000

    # Hypothesis
    y_pred = tf.add(tf.multiply(X, W), b)

    # Mean Squared Error Cost Function
    cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

    # Gradient Descent Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Global Variables Initializer
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(device_count={'CPU': 6})

    # Starting the Tensorflow Session
    with tf.Session(config=config) as sess:

        # Initializing the Variables
        sess.run(init)

        # Iterating through all the epochs
        for epoch in range(training_epochs):

            # Feeding each data point into the optimizer using Feed Dictionary
            for (_x, _y) in zip(x, y):
                sess.run(optimizer, feed_dict = {X : _x, Y : _y})

            # Displaying the result after every 50 epochs
            if (epoch + 1) % 50 == 0:
                # Calculating the cost a every epoch
                c = sess.run(cost, feed_dict = {X : x, Y : y})
                print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b))

        # Storing necessary values to be used outside the Session
        training_cost = sess.run(cost, feed_dict ={X: x, Y: y})
        weight = sess.run(W)
        bias = sess.run(b)

    # Calculating the predictions
    predictions = weight * x + bias
    print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n')

    # Plotting the Results
    plt.plot(x, y, 'ro', label ='Original data')
    plt.plot(x, predictions, label ='Fitted line')
    plt.title('Linear Regression Result')
    plt.legend()
    plt.show()

#Linear Regression Method 2
#Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['Polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['R Squared'] = ssreg / sstot

    return results

# Python program to find
# maximal Bipartite matching.
class GFG:
    def __init__(self,graph):

        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])

    # A DFS based recursive function
    # that returns true if a matching
    # for vertex u is possible
    def bpm(self, u, matchR, seen):

        # Try every job one by one
        for v in range(self.jobs):

            # If applicant u is interested
            # in job v and v is not seen
            if self.graph[u][v] and seen[v] == False:

                # Mark v as visited
                seen[v] = True

                '''If job 'v' is not assigned to
                   an applicant OR previously assigned
                   applicant for job v (which is matchR[v])
                   has an alternate job available.
                   Since v is marked as visited in the
                   above line, matchR[v]  in the following
                   recursive call will not get job 'v' again'''
                if matchR[v] == -1 or self.bpm(matchR[v],
                                               matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        '''An array to keep track of the
           applicants assigned to jobs.
           The value of matchR[i] is the
           applicant number assigned to job i,
           the value -1 indicates nobody is assigned.'''
        matchR = [-1] * self.jobs

        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):

            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs

            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result


bpGraph =[[0, 1, 1, 0, 0, 0],
          [1, 0, 0, 1, 0, 0],
          [0, 0, 1, 0, 0, 0],
          [0, 0, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1]]

# g = GFG(bpGraph)

# print ("Maximum number of applicants that can get job is %d " % g.maxBPM())

# This code is contributed by Neelam Yadav

# I do not use the code directly above, I tried to use the above example
# for my weighted bipartite graph, but I could not modify it to take in weights
# correctly.

v1size = 5;
# -10 to 10 for all of these
# How active are you?
# How much do you enjoy consuming passive media, i.e. netflix, books, tv?
# How much do you enjoy physical affection, i.e. sex, cuddling, kissing?
# How much time do you want to spend with your SO?
# How much do you enjoy learning new things?
v2size = 1;
# true or false
# What is your sexual orientation?

def computeComp(v1, v2):
    dict = polyfit(v1, v2, 1);
    return dict['R Squared'];

def matchSingle():
    #
    #
    #
    return 0;

def createData(numVectors):

    dataMatrix = [];

    for i in range(0, numVectors):

        x = np.linspace(0, 0, 5)
        # y = np.linspace(0, 0, 100)

        # Adding noise to the random linear data
        x += np.random.uniform(-10, 10, len(x))
        # y += np.random.uniform(-10, 10, len(y))

        dataMatrix.append(x);

    return dataMatrix;

def createCompCoefficientMatrix(dataMatrix):

    dim = len(dataMatrix)
    coeffMatrix = []

    for i in range(0, dim):
        row = [];
        for j in range(0, dim):
            row.append(0.0);
        coeffMatrix.append(row);

    for i in range(0, dim):
        for j in range(0, dim):
            coeffMatrix[i][j] = Value(truncate(computeComp(dataMatrix[i], dataMatrix[j]), 4), j, i);

    return coeffMatrix;

def createAdjMatrix(coeffMatrix):
    for i in range(len(coeffMatrix)):
        for j in range(len(coeffMatrix)):
            if(coeffMatrix[i][j].value > .7):
                coeffMatrix[i][j] = 1
            else:
                coeffMatrix[i][j] = 0
    return coeffMatrix;


def printMatrix(matrix):
    for i in matrix:
        row = []
        for j in i:
            row.append(j.getValue());
        print(row);

def setDiagTo0(matrix):
    for i in range(0, len(matrix)):
        matrix[i][i] = Value(0.0, i, i);

def removeRow(matrix, i):
    matrix.pop(i);

def removeColumn(matrix, j):
    for i in range(0, len(matrix)):
        matrix[i].pop(j);

def maxVal(matrix):
    val = 0.0;
    for i in range(0, len(matrix)):
        for j in range (0, len(matrix[0])):
            if(matrix[i][j].value > val):
                val = matrix[i][j].value;
    return val;

def getMaxValIndex(matrix):
    val = 0.0;
    x = -1;
    y = -1;
    for i in range(0, len(matrix)):
        for j in range (0, len(matrix[0])):
            if(matrix[i][j].value > val):
                val = matrix[i][j].value;
                x = i;
                y = j;

    return x, y;

def getMatch(matrix):
    i, j = getMaxValIndex(matrix);
    x = matrix[i][j].x
    y = matrix[i][j].y
    if(i < j):
        removeColumn(matrix, i)
        removeColumn(matrix, j - 1)
        removeRow(matrix, i)
        removeRow(matrix, j - 1)
    else:
        removeColumn(matrix, j)
        removeColumn(matrix, i - 1)
        removeRow(matrix, j)
        removeRow(matrix, i - 1)

    return x, y;


def makeMatches(dataMatrix):

    coeffMatrix = createCompCoefficientMatrix(dataMatrix);
    matches = [];
    numMatches = [];

    setDiagTo0(coeffMatrix);

    print("coeffmatrix");
    printMatrix(coeffMatrix);

    # print(maxVal(coeffMatrix));
    # print(getMaxValIndex(coeffMatrix));

    # adj = createAdjMatrix(createCompCoefficientMatrix(dataMatrix))
    # for i in adj:
    #     print(i)

    adjMatrix = tf.constant(createAdjMatrix(createCompCoefficientMatrix(dataMatrix)));
    matchMatrix = tf.matmul(adjMatrix, adjMatrix);

    while(maxVal(coeffMatrix) > .7):
        matches.append(getMatch(coeffMatrix));

    print("coeffmatrix");
    printMatrix(coeffMatrix);

    for i in range(0, len(dataMatrix)):
        numMatches.append(matchMatrix[i][i]);

    return matches, numMatches;


x = np.linspace(0, 4, 5)
a = tf.constant([1.0, 2.0, 3.0]);
a1 = [1.0, 2.0, 3.0, 4.0, 5.0]
b = tf.constant([1.0, 2.0, 3.0]);
c = tf.constant([1.0, 2.0, 3.0]);

linearRegression(a1);
# print(computeComp(x, a1))

#Randomized data for algorithm:

# x = np.linspace(0, len(vector) - 1, len(vector))
# y = vector

# dataMatrix = createData(25)
#
# matches, numMatches = makeMatches(dataMatrix);
# print(matches);

# linearRegression(dataMatrix[0]);

# for i in numMatches:
#     tf.print("value", i);

# linearRegression(dataMatrix[0]);

# print("data matrix");
# printMatrix(dataMatrix);
# print("compatibility coefficient matrix");
# printMatrix(coeffMatrix);
# setDiagTo0(coeffMatrix);
# print("compatibility coefficient matrix");
# printMatrix(coeffMatrix);
# removeRow(coeffMatrix, 0);
# print("compatibility coefficient matrix");
# printMatrix(coeffMatrix);
# removeColumn(coeffMatrix, 0);
# print("compatibility coefficient matrix");
# printMatrix(coeffMatrix);
