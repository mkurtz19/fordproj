import sklearn
import numpy
from sklearn import svm
import matplotlib.pyplot as plt
import pickle
from sklearn import neighbors
import copy
import ml_metrics as metrics

def trainSVR(infile):
    print "training"

    print "reading data from file"

    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',', max_rows=400000)

    print "finished reading file"

    #inc = 80
    inc = 20

    z = []

    for i in range(inc):
        z.append(tData[i::inc])
    z = numpy.array(z)
    z = z.mean(0)

    #x = tData[::50,numpy.r_[3:33]]
    #y = tData[::50,2]

    x = z[:,numpy.r_[3:33]]
    y = z[:,2]

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

    model = svm.SVR(kernel='rbf', C=10, gamma=0.01, verbose=3)
    model.cache_size = 1048576

    print "fitting data"

    model.fit(x,y)

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def trainSVC(infile):
    print "reading data from file"

    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',', max_rows=400000)

    print "finished reading file"

    inc = 80

    z = []

    for i in range(inc):
        z.append(tData[i::inc])
    z = numpy.array(z)
    z = z.mean(0)

    #x = tData[::50,numpy.r_[3:33]]
    #y = tData[::50,2]

    x = z[:,numpy.r_[3:33]]
    y = z[:,2]>0.5

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

    model = svm.SVC(kernel='rbf', C=10, gamma=0.01, verbose=3)
    model.cache_size = 1024

    print "fitting data"

    model.fit(x,y)

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def trainKNN(infile):
    print "reading data from file"

    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',', max_rows=100000)

    print "finished reading file"

    x = tData[:,3:33]
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

    print "creating KNN model"

    model = neighbors.KNeighborsClassifier(n_neighbors=6, weights='distance', algorithm='auto', n_jobs=-1)
    model.cache_size = 1024

    print "fitting data"

    model.fit(x,y)

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def trainLR(infile):
    print "reading data from file"

    #cols = numpy.r_[3:33]
    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',', max_rows=100000)

    print "finished reading file"

    x = tData[:,3:33]
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

    model = sklearn.linear_model.LogisticRegression()
    model.cache_size = 1024

    print "fitting data"

    model.fit(x,y)

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def trainLL(infile):
    print "reading data from file"

    #cols = numpy.r_[3:33]
    tData = numpy.genfromtxt(infile, skip_header=1, delimiter=',')#, max_rows=100000)

    print "finished reading file"

    x = tData[:,11:33]
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

    model = sklearn.linear_model.LassoLars()
    model.cache_size = 1024

    print "fitting data"

    model.fit(x,y)

    print "saving model to model.pkl"

    pickle.dump(model, open('model.pkl', 'wb'))

    return {'model':model, 'x':x, 'y':y}

def holdout_test(modelfile, testfile, solutionsfile):
    print "holdout set"

    model = pickle.load(open(modelfile, 'rb'))

    tData = numpy.genfromtxt(testfile, skip_header=1, delimiter=',')
    sData = numpy.genfromtxt(solutionsfile, skip_header=1, delimiter=',')

    print "finished reading file"

    x = tData[:,numpy.r_[3:33]]
    y = sData[:,2]

    mins = x.min(0)
    x = x - mins
    x = x - (x.max(0) - x.min(0)) / 2
    maxs = x.max(0)
    maxs[maxs==0] = 1
    x = 2 * x / maxs

    py = numpy.array(y)

    preds = numpy.array(model.predict(x))
    savepreds = copy.deepcopy(preds)
    preds[preds>0.5] = 1
    preds[preds<=0.5] = 0

    truepos = sum((preds == 1) * (py == 1))
    trueneg = sum((preds == 0) * (py == 0))
    falsepos = sum((preds == 1) * (py == 0))
    falseneg = sum((preds == 0) * (py == 1))

    correct = truepos + trueneg

    precision = 1.0 * truepos / sum(preds == 1)
    recall = 1.0 * truepos / sum(py == 1)
    f1 = 2.0 * precision * recall / (precision + recall)

    print "num correct: " + str(correct)
    print "true positives: " + str(truepos)
    print "true negatives: " + str(trueneg)
    print "false positives: " + str(falsepos)
    print "false negatives: " + str(falseneg)

    print "overall accuracy: " + str(1.0 * correct / len(py))
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "f1: " + str(f1)

    tpos = [0]
    fpos = [0]

    for i in numpy.r_[0:1.001:0.05]:
        tpos.append(1.0 * sum(((savepreds > i) == 1) * (py == 1)) / sum(((savepreds > i) == 1)))
        fpos.append(1.0 * sum(((savepreds > i) == 0) * (py == 0)) / sum(py == 1))

    tpos.append(1)
    fpos.append(1)

    auc = 0.0
    for i in range(len(tpos) - 1):
        auc = auc + (tpos[i + 1] + tpos[i]) * (fpos[i + 1] - fpos[i]) / 2.0

    print "auc: " + str(auc)

    plt.plot(fpos, tpos)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Holdout ROC')
    plt.show()
    plt.savefig("holdoutroc.png")

def test(modelfile, testfile):
    print "test set"

    model = pickle.load(open(modelfile, 'rb'))

    tData = numpy.genfromtxt(testfile, skip_header=400001, delimiter=',')

    print "finished reading file"

    x = tData[:,numpy.r_[3:33]]
    y = tData[:,2]

    mins = x.min(0)
    x = x - mins
    x = x - (x.max(0) - x.min(0)) / 2
    maxs = x.max(0)
    maxs[maxs==0] = 1
    x = 2 * x / maxs

    py = numpy.array(y)

    preds = numpy.array(model.predict(x))
    savepreds = copy.deepcopy(preds)
    preds[preds>0.5] = 1
    preds[preds<=0.5] = 0

    truepos = sum((preds == 1) * (py == 1))
    trueneg = sum((preds == 0) * (py == 0))
    falsepos = sum((preds == 1) * (py == 0))
    falseneg = sum((preds == 0) * (py == 1))

    correct = truepos + trueneg

    precision = 1.0 * truepos / sum(preds == 1)
    recall = 1.0 * truepos / sum(py == 1)
    f1 = 2.0 * precision * recall / (precision + recall)

    print "num correct: " + str(correct)
    print "true positives: " + str(truepos)
    print "true negatives: " + str(trueneg)
    print "false positives: " + str(falsepos)
    print "false negatives: " + str(falseneg)

    print "overall accuracy: " + str(1.0 * correct / len(py))
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "f1: " + str(f1)

    tpos = [0]
    fpos = [0]

    for i in numpy.r_[0:1.001:0.05]:
        tpos.append(1.0 * sum(((savepreds > i) == 1) * (py == 1)) / sum(((savepreds > i) == 1)))
        fpos.append(1.0 * sum(((savepreds > i) == 0) * (py == 0)) / sum(py == 1))

    tpos.append(1)
    fpos.append(1)

    auc = 0.0
    for i in range(len(tpos) - 1):
        auc = auc + (tpos[i + 1] + tpos[i]) * (fpos[i + 1] - fpos[i]) / 2.0

    print "auc: " + str(auc)

    plt.plot(fpos, tpos)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC')
    plt.show()
    plt.savefig("testroc.png")

if __name__ == "__main__":
    ford = trainSVR('fordTrain.csv')
    test('model.pkl', 'fordTrain.csv')
    holdout_test('model.pkl', 'fordTest.csv', 'Solution.csv')
