import sklearn
import numpy
from sklearn import svm
import matplotlib.pyplot as plt
import pickle

def train(infile):
    print "reading data from file"

    #cols = numpy.r_[3:33]
    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',')#, usecols=cols)

    print "finished reading file"

    x = tData[:,3:11]
    y = tData[:,2]

    print "N: " + str(x.shape[0])

    mins = x.min(0)
    x = x - mins
    x = x - (x.max(0) - x.min(0)) / 2
    maxs = x.max(0)
    maxs[maxs==0] = 1
    x = 2 * x / maxs
    #maxs = x.max(0)
    #ranges = maxs - mins
    #x = 2 * (x - mins) / ranges - 1;

    print "creating model"

    model = svm.SVR(kernel='rbf', C=10, gamma=0.001, verbose=3)
    model.cache_size = 1024

    print "fitting data"

    model.fit(x[0:100000,:],y[0:100000])

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def test():
    model = pickle.load(open('model.pkl', 'rb'))

    tData = numpy.genfromtxt('fordTrain.csv', skip_header=1, delimiter=',')

    print "finished reading file"

    x = tData[:,3:11]
    y = tData[:,2]

    mins = x.min(0)
    x = x - mins
    x = x - (x.max(0) - x.min(0)) / 2
    maxs = x.max(0)
    maxs[maxs==0] = 1
    x = 2 * x / maxs

    py = numpy.array(y[100000:200000])

    preds = numpy.array(model.predict(x[100000:200000,:]))
    preds[preds>0.5] = 1
    preds[preds<=0.5] = 0

    truepos = sum((preds == 1) & (py == 1))
    trueneg = sum((preds == 0) & (py == 0))
    falsepos = sum((preds == 1) & (py == 0))
    falseneg = sum((preds == 0) & (py == 1))

    correct = truepos + trueneg

    precision = truepos / sum(preds == 1)
    recall = truepos / sum(py == 1)
    f1 = 2 * precision * recall / (precision + recall)

    print "num correct: " + str(correct)
    print "true positives: " + str(truepos)
    print "true negatives: " + str(trueneg)
    print "false positives: " + str(falsepos)
    print "false negatives: " + str(falseneg)

    print "overall accuracy: " + str(1.0 * correct / len(py))
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "f1: " + str(f1)

if __name__ == "__main__":
    #ford = train('fordTrain.csv')
    test();
