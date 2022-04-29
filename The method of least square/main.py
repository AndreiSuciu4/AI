import csv
import math
import os
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits import mplot3d
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
from MyLinearUnivariateRegression import MyLinearUnivariateRegression


def loadData(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    firstVariable = dataNames.index(inputVariabName[0])
    secondVariable = dataNames.index(inputVariabName[1])



    inputs = []
    for i in range(len(data)):
        row = []
        row.append(data[i][firstVariable])
        row.append(data[i][secondVariable])
        inputs.append(row)

    selectedOutput = dataNames.index(outputVariabName)
    outputs = [data[i][selectedOutput] for i in range(len(data))]
    return inputs, outputs

def addDataInNullColumn(fileName, inputVariabName, outputVariabName, i):
    inputs, outputs = loadData(fileName, inputVariabName, outputVariabName)
    floatOutputs = []
    floatInputs = []
    for i in range(len(outputs)):
        if outputs[i] != '':
            floatOutputs.append(float(outputs[i]))
            row = []
            row.append(float(inputs[i][0]))
            row.append(float(inputs[i][1]))
            floatInputs.append(row)
    regressor = MyLinearUnivariateRegression()
    # training the model by using the training inputs and known training outputs
    trainOutputsMat = [[float(x)] for x in floatOutputs]
    regressor.fit(floatInputs, trainOutputsMat)
    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    value = w0 + w1 * float(inputs[i][0]) + w2 * float(inputs[i][1])
    return value

def floatData(inputs, ouputs):
    floatInputs = []
    floatOutputs = []
    indexs = []
    for i in range(len(ouputs)):
        if ouputs[i] == '':
            indexs.append(i)
        else:
            floatOutputs.append(float(ouputs[i]))
    for i in range (len(ouputs)):
        if i in indexs:
            continue
        row = []
        if(inputs[i][0] == ''):
            value = addDataInNullColumn("v3_world-happiness-report-2017.csv", ['Whisker.low', 'Health..Life.Expectancy.'],'Economy..GDP.per.Capita.', i)
            row.append(value)
        else:
            row.append(float(inputs[i][0]))
        if (inputs[i][1] == ''):
            value = addDataInNullColumn("v3_world-happiness-report-2017.csv", ['Whisker.low', 'Health..Life.Expectancy.'],'Economy..GDP.per.Capita.', i)
            row.append(value)
        else:
            row.append(float(inputs[i][1]))
        floatInputs.append(row)
    return floatInputs, floatOutputs

def verifyData(inputs, outputs):
    matrix = []
    output = []
    copy = [x for x in inputs]
    inputs.sort(key=lambda row: row[0:])
    ln = len(inputs)
    if inputs[0] != inputs[1]:
        matrix.append(inputs[0])
        output.append(outputs[copy.index(inputs[0])])
    for i in range(1, ln - 1):
        if inputs[i - 1] != inputs[i] and inputs[i] != inputs[i + 1]:
           matrix.append(inputs[i])
           output.append(outputs[copy.index(inputs[i])])
    if inputs[ln - 2] != inputs[ln - 1]:
        matrix.append(inputs[ln - 1])
        output.append(outputs[copy.index(inputs[ln - 1])])

    return matrix, output



# see how the data looks (plot the histograms associated to input data - GDP feature - and output data - happiness)

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

def plotDataHistogramWithThoArrays(x, y, variableName):
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x, 10)
    ax1 = ax.twinx()
    ax1.hist(y, 10, color="red")
    plt.title('Histogram of ' + variableName)
    plt.show()

# check the liniarity (to check that a linear relationship exists between the dependent variable (y = happiness) and the independent variable (x = capita).)
def checkLiniarity(x, y, outputs, xLabel, yLabel, zLabel):
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(x, y, outputs, color='r')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.title("GDP capita and freedom vs happiness")
    plt.show()

def splitData(gdp, freedom, outputs):
    # Split the Data Into Training and Test Subsets
    # In this step we will split our dataset into training and testing subsets (in proportion 80/20%).
    # Training data set is used for learning the linear model. Testing dataset is used for validating of the model. All data from testing dataset will be new to model and we may check how accurate are model predictions.

    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(outputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]

    gdpInputs = [gdp[i] for i in trainSample]
    freedomInputs = [freedom[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]

    validationGdp = [gdp[i] for i in validationSample]
    validationFreedom = [freedom[i] for i in validationSample]
    validationOutputs = [outputs[i] for i in validationSample]

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(gdpInputs, freedomInputs, trainOutputs, color='r', label = 'training data')
    ax.scatter3D(validationGdp, validationFreedom, validationOutputs, color='y', label = 'validation data')
    ax.set_xlabel("gdp")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.legend()
    plt.title("train and validation data")
    plt.show()
    return gdpInputs, freedomInputs, trainOutputs, validationGdp, validationFreedom, validationOutputs

# learning step: init and train a linear regression model y = f(x) = w0 + w1 * x1 + w2 * x2
# Prediction step: used the trained model to estimate the output for a new input

# using sklearn
def learningModel(gdpInputs, freedomInputs, trainOutputs):
    xx = []
    for i in range(len(gdpInputs)):
        x = []
        x.append(gdpInputs[i])
        x.append(freedomInputs[i])
        xx.append(x)

    # model initialisation
    regressor = linear_model.LinearRegression()
    # training the model by using the training inputs and known training outputs
    regressor.fit(xx, trainOutputs)
    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, "* x2" )
    return  w0, w1, w2, regressor

def mylearningModel(gdpInputs, freedomInputs, trainOutputs):
    xx = []
    for i in range(len(gdpInputs)):
        x = []
        x.append(1)
        x.append(gdpInputs[i])
        x.append(freedomInputs[i])
        xx.append(x)

    # model initialisation
    #regressor = linear_model.LinearRegression()
    regressor = MyLinearUnivariateRegression()
    # training the model by using the training inputs and known training outputs
    trainOutputsMat = [[x] for x in trainOutputs]
    regressor.fit(xx, trainOutputsMat)
    # save the model parameters
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, "* x2" )
    return  w0, w1, w2, regressor

def createLine(trainInputs1,trainInputs2,w0,w1,w2):
    noOfPoints = 1000
    x1ref = []
    x2ref = []
    val1 = min(trainInputs1)
    val2 = min(trainInputs2)
    step1 = (max(trainInputs1) - min(trainInputs1)) / noOfPoints
    step2 = (max(trainInputs2) - min(trainInputs2)) / noOfPoints
    for _ in range(1, noOfPoints):
        x1ref.append(val1)
        val1 += step1
        x2ref.append(val2)
        val2 += step2
    zref = [(w0 + w1 * x1 + w2 * x2) for x1,x2 in zip(x1ref,x2ref)]
    return x1ref,x2ref,zref

def plotResult(gdpInputs, freedomInputs, trainOutputs, w0, w1, w2):
    # plot the learnt model
    # prepare some synthetic data (inputs are random, while the outputs are computed by the learnt model)
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    xline, yline, zline = createLine(gdpInputs, freedomInputs, w0, w1, w2)
    ax.scatter(gdpInputs, freedomInputs, trainOutputs, color='red', marker="o")
    ax.plot3D(xs=xline, ys=yline, zs=zline, c='black', label='Model')
    ax.set_xlabel("gdp")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.title('train data and the learnt model')
    plt.show()

def plotResultNewInputs(regresor, validationGdp, validationFreedom, validationOutputs):
    xx = []
    for i in range(len(validationGdp)):
        x = []
        x.append(validationGdp[i])
        x.append(validationFreedom[i])
        xx.append(x)


    computedValidationOutputs = regresor.predict(xx)

    noOfPoints = 1000
    x1ref = []
    x2ref = []
    val1 = min(validationGdp)
    val2 = min(validationFreedom)
    step1 = (max(validationGdp) - min(validationGdp)) / noOfPoints
    step2 = (max(validationFreedom) - min(validationFreedom)) / noOfPoints
    for _ in range(1, noOfPoints):
        x1ref.append(val1)
        val1 += step1
        x2ref.append(val2)
        val2 += step2

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(validationGdp, validationFreedom, validationOutputs, color='red', marker="o")
    ax.scatter(validationGdp, validationFreedom, computedValidationOutputs, marker="^",c='black', label='Model')
    ax.set_xlabel("gdp")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.title('computed validation and real validation data')
    plt.show()
    return  computedValidationOutputs

def error(computedValidationOutputs, validationOutputs):
    # compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedValidationOutputs, validationOutputs):
        error += (t1 - t2) ** 2
    error = math.sqrt(error / len(validationOutputs))
    print("prediction error (manual): ", error)

if __name__ == '__main__':

    inputs, outputs = loadData("v3_world-happiness-report-2017.csv", ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

    inputs, output = floatData(inputs, outputs)

    input, output = verifyData(inputs, output)

    gdp = [row[0] for row in input]
    freedom = [row[1] for row in input]
    #plot the histogram

    plotDataHistogram([row[0] for row in input], 'capita GDP')
    plotDataHistogram([row[1] for row in input], 'Freedom')
    plotDataHistogram(outputs, 'Happiness score')

    #plotDataHistogramWithThoArrays([row[0] for row in input], [row[1] for row in input], "capitalGDP and Freedom")

    checkLiniarity(gdp, freedom, output, "gdp", "freedom", "happiness")

    gdpInputs, freedomInputs, trainOutputs, validationGdp, validationFreedom, validationOutputs = splitData(gdp, freedom, output)
    #w0, w1, w2, regresor = learningModel(gdpInputs, freedomInputs, trainOutputs)
    w0, w1, w2, regresor = mylearningModel(gdpInputs, freedomInputs, trainOutputs)
    plotResult(gdpInputs, freedomInputs, trainOutputs, w0, w1, w2)
    computedValidationOutputs = plotResultNewInputs(regresor, validationGdp, validationFreedom, validationOutputs)
    error(computedValidationOutputs, validationOutputs)


