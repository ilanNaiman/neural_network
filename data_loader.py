from scipy.io import loadmat


def loadGMMData():
    data = loadmat('NNdata/GMMData.mat')

    Xtrain = data['Yt']
    Xtest = data['Yv']
    Ytrain = data['Ct']
    Ytest = data['Cv']

    return Xtrain, Xtest, Ytrain, Ytest


def loadSwissRollData():

    data = loadmat('NNdata/SwissRollData.mat')

    Xtrain = data['Yt']
    Xtest = data['Yv']
    Ytrain = data['Ct']
    Ytest = data['Cv']

    return Xtrain, Xtest, Ytrain, Ytest


