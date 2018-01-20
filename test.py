from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

def predict(Xtr, Ytr, Xts, metric=None):



    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros((Xts.shape[0]))
    #print(N)
    #print(D)

    #print(metric.shape[0])
    #print(metric.shape[1])

    for i in range(Xts.shape[0]):

        kval = 12
        ######### This is the tuned value of k

        diffs = Xtr-Xts[i]
        diffst = diffs.T

        tempdiffs = diffs.dot(metric)
        diffs = np.multiply(tempdiffs, diffs)
        #diffs = tempdiffs.dot(diffs)
        hunones = np.ones((Xtr.shape[1], 1))
        diffs =diffs.dot(hunones)
        #diffs = np.diagonal(diffs)
        #diffs= np.squeeze(np.asarray(diffs))
        #print(i)
        diffs = np.array(diffs.T)
        diffs = diffs[0]
        idx = diffs.T.argsort()[:kval]
        #print(idx)
        count1 = 0
        count2 = 0
        count3 = 0
        for k in idx:
            if(Ytr[k] == 1):
                count1=count1+1
            if(Ytr[k] == 2):
                count2=count2+1
            if(Ytr[k] == 3):
                count3=count3+1
        if(count1 >= count2 and count1 >= count3):
            Yts[i] = 1
        if(count2 >= count1 and count2 >= count3):
            Yts[i] = 2
        if(count3 >= count1 and count3 >= count2):
            Yts[i] = 3
        #print(count1, count2,count3)

    #print(diffs)
    #print(Yts)
    return Yts

def accuracy(Yts, Ytsg):
	
	#print (Yts)
	#Ytsg = (np.matrix(Ytsg)).T
	#print (Ytsg)
    #print(Yts.shape[0])
    count = (Yts==Ytsg)
    #print(count)
    count = np.sum(count)
    #print(count)
    return 100*count*1.0/Yts.shape[0]


def main(): 

    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]
    testdatafile = sys.argv[2]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];

    # The testing file is in libSVM format too
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray();
    Ytsg = ts_data[1]
    #print(Ytsg.shape[0])
    
    metric = np.load("model.npy")

    ### Do soemthing (if required) ###

    Yts = predict(Xtr, Ytr, Xts, metric)
    print(accuracy(Yts,Ytsg))

    # Save predictions to a file
	# Warning: do not change this file name
    
    np.savetxt("testY.dat", Yts)

if __name__ == '__main__':
    main()
