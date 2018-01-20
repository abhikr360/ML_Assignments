from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

def predict(Xtr, Ytr, Xts, values):



    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    Yts = np.zeros([5,(Xts.shape[0])])

    NSizeMatrixOfones = np.ones((Xtr.shape[0], 1))# N * 1 matrix of ones
    DSizeMatrixOfones = np.ones((Xtr.shape[1], 1))# D * 1 matrix of ones

    for i in range(Xts.shape[0]):

        ToMatrix = NSizeMatrixOfones.dot(np.matrix(Xts[i]))
        Difference = ToMatrix - Xtr

        DifferenceSquare = np.square(Difference)
        Difference = DifferenceSquare.dot(DSizeMatrixOfones)

        Difference = np.array(Difference.T)
        Difference = Difference[0]

        for j in range(len(values)):
            smallestk= Difference.argsort()[:values[j]]
            count1,count2,count3=0,0,0
            for k in smallestk:
                if(Ytr[k] == 1):
                    count1=count1+1
                elif(Ytr[k] == 2):
                    count2=count2+1
                else:
                    count3=count3+1
            if(count1 >= count2 and count1 >= count3):
                Yts[j][i] = 1
            elif(count2 >= count1 and count2 >= count3):
                Yts[j][i] = 2
            else:
                Yts[j][i] = 3

    return Yts

def accuracy(Yts, Ytsg):
    count = (Yts==Ytsg)
    count = np.sum(count)
    return (100.0*count)/Yts.shape[0]


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
    # The test labels are useless for prediction. They are only used for evaluation

    # Load the learned metric
    
    #metric = np.load("model.npy")

    values =[1,2,3,5,10]

    Yts = predict(Xtr, Ytr, Xts, values)
    for i in range(5):
        print(values[i]),
        print(accuracy(Yts[i],Ytsg))

    # Save predictions to a file
    # Warning: do not change this file name
    
    #np.savetxt("testY.dat", Yts)

if __name__ == '__main__':
    main()
