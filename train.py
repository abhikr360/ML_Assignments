from __future__ import print_function
import random

import numpy as np
import sys
from modshogun import LMNN, RealFeatures, MulticlassLabels
from sklearn.datasets import load_svmlight_file



def predict(Xtr, Ytr, values):



    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    #assert D == Xts.shape[1], "Train and test dimensions don't match"

    #Yts = np.zeros([len(values),(Xts.shape[0])])

    #Held-out Validation
    #print(N),
    #print(D)
    split=int(N*0.65)

    Xtrtr = Xtr[:split]
    Ytrtr = Ytr[:split]
    #print(Xtrtr)
    #print(Ytrtr)
    Xtrval = Xtr[split:]
    Ytrval = Ytr[split:]
    Yts = np.zeros([len(values),(Xtrval.shape[0])])

    NSizeMatrixOfones = np.ones((Xtrtr.shape[0], 1))
    DSizeMatrixOfones = np.ones((Xtrtr.shape[1], 1))

    for i in range(Xtrval.shape[0]):
        ToMatrix = NSizeMatrixOfones.dot(np.matrix(Xtrval[i]))
        Difference = ToMatrix - Xtrtr

        DifferenceSquare = np.square(Difference)
        Difference = DifferenceSquare.dot(DSizeMatrixOfones)

        Difference = np.array(Difference.T)
        Difference = Difference[0]
        #print(i)
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

            #quit()
        #print(i)
    return accuracy(Yts, Ytrval, values)

def accuracy(Yts, YtsGiven, values):
    maxaccuracy=0
    koptimum=1
    for i in range(len(values)):
        #print(values[i]),
        count = (Yts[i]==YtsGiven)
        count = np.sum(count)
        #print(count)
        if(((100.0*count)/Yts.shape[1]) > maxaccuracy):
            koptimum=values[i]
    return koptimum


def main(): 

    # Get training file name from the command line
    traindatafile = sys.argv[1]

	# The training file is in libSVM format

    with open(traindatafile, mode="r") as myFile:
        lines=myFile.readlines()

    random.shuffle(lines)
    open("tempdata.dat", 'w').writelines(lines)


    tr_data = load_svmlight_file("tempdata.dat");#To randomly select 5000 points

    Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
    Ytr = tr_data[1]; # The trainig labels
    
    Xtr = Xtr[:5000]
    Ytr = Ytr[:5000]
    # Cast data to Shogun format to work with LMNN
    features = RealFeatures(Xtr.T)
    labels = MulticlassLabels(Ytr.astype(np.float64))


    #print(Xtr.shape)
    ### Do magic stuff here to learn the best metric you can ###
    kmax=25#inductive bias
    values = list(range(1,kmax+1))
    k=predict(Xtr, Ytr, values)
    # Number of target neighbours per example - tune this using validation
    #print(k)
    # Initialize the LMNN package
    print("K : "),
    print(k)

    k=5
    lmnn = LMNN(features, labels, k)
    init_transform = np.eye(Xtr.shape[1])

    # Choose an appropriate timeout
    lmnn.set_maxiter(25000)
    lmnn.train(init_transform)

    # Let LMNN do its magic and return a linear transformation
	# corresponding to the Mahalanobis metric it has learnt
    L = lmnn.get_linear_transform()
    M = np.matrix(np.dot(L.T, L))

    print("LMNN done")
    #print(M)
    # Save the model for use in testing phase
	# Warning: do not change this file name
    np.save("model.npy", M) 
    #print("Bye  ")

if __name__ == '__main__':
    main()
