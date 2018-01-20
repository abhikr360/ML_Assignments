from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

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

    return accuracy(Yts, Ytrval, values)

def accuracy(Yts, YtsGiven, values):
    kopt=1
    maxi=0
    for i in range(len(values)):
        #print(values[i]),
        count = (Yts[i]==YtsGiven)
        count = np.sum(count)
        #print(Yts.shape[1])
        #print((100.0*count)/Yts.shape[1])
        if((100.0*count)/Yts.shape[1] > maxi):
            kopt=values[i]
    return kopt


def main(): 

    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];

    # The testing file is in libSVM format too

    # The test labels are useless for prediction. They are only used for evaluation

    # Load the learned metric
    
    #metric = np.load("model.npy")

    ### Do soemthing (if required) ###
    kmax=30
    values = list(range(1,kmax+1))
    #print(values)
    print("kopt : ")
    print(predict(Xtr, Ytr, values))
    # for i in range(len(values)):
    #     print(values[i]),
    #     print(accuracy(Yts[i],YtsGiven))

    # Save predictions to a file
    # Warning: do not change this file name
    
    #np.savetxt("testY.dat", Yts)

if __name__ == '__main__':
    main()
