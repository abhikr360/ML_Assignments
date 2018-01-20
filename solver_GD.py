import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math

# def h(n):
#     return 1.0/n;

# def accuracy(Xts,Yts,w_final,w_finalavg, ntest):
#     Yp1 = np.matmul(Xts,w_final);
#     Yp2 = np.matmul(Xts,w_finalavg);

#     Yp1 = np.multiply(Yp1,Yts);
#     Yp2 = np.multiply(Yp2,Yts);

#     Yp1 = np.where(Yp1>0,1,0);
#     Yp2 = np.where(Yp2>0,1,0);

#     acy_nor = np.sum(Yp1);
#     acy_avg = np.sum(Yp2);

#     print("accuracy nor :"),
#     print(100*(acy_nor/float(ntest)))
#     print("accuracy avg :"),
#     print(100*(acy_avg/float(ntest)))


def main():
    # Get training file name from the command line
    traindatafile = sys.argv[1];
    # For how many iterations do we wish to execute GD?
    n_iter = int(sys.argv[2]);
    # After how many iterations do we want to timestamp?
    spacing = int(sys.argv[3]);

    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);

    Xtr = tr_data[0]; # Training features in sparse format
    Ytr = tr_data[1]; # Training labels

    n, d = Xtr.get_shape();
    # We have n data points each in d-dimensions
    #Xts = Xtr[int(0.85*n):]
    #Xtr = Xtr[:int(0.85*n)]
    # The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
    Ytr = 2*(Ytr - 1.5);
    Ytr = Ytr.astype(int);
    #Yts = Ytr[int(0.85*n):]
    #Ytr = Ytr[:int(0.85*n)]
    # Optional: densify the features matrix.
    # Warning: will slow down computations

    #n, d = Xtr.get_shape();
    #ntest, dtest = Xts.get_shape();
    Xtr = Xtr.toarray();
    #Xts = Xts.toarray();
    # Initialize model
    # For primal GD, you only need to maintain w
    # Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
    # w = csr_matrix((1, d));
    w = np.zeros(d)
    #wbar = np.ones(d)
    # We will take a timestamp after every "spacing" iterations
    time_elapsed = np.zeros(int(math.ceil(n_iter/spacing)));
    tick_vals = np.zeros(int(math.ceil(n_iter/spacing)));
    obj_val = np.zeros(int(math.ceil(n_iter/spacing)));
    theory_time = np.zeros(int(math.ceil(n_iter/spacing)));

    tick = 0;
    ttot = 0.0;
    t_start = datetime.now();

    Xtr_y = (Xtr.T*Ytr).T;
    gsqr = np.zeros(d);
    gsqr = gsqr + 1e-18;
    
    for t in range(n_iter):
        ### Doing primal GD ###


        # -------------------Compute gradient---------------------------
        temp1 = np.matmul(Xtr_y,w);
        temp2 = np.where(temp1<1,Ytr,0);
        subg = np.matmul(temp2,Xtr);
        g = w - subg;
        g.reshape(1,d); # Reshaping since model is a row vector
        #---------------------------------------------------------------

        #-----------Adagrad step-----------------
        gsqr = gsqr + np.multiply(g,g)
        g1 = np.power(gsqr,-0.5)
        diagG = np.diag(g1);
        #-----------------------------------------


        #update model using adgagrad
        w = w -2.0*diagG.dot(g);

        # Calculate step length. Step length may depend on n and t
        #eta = h(n) * 1/math.sqrt(t+1) * (-4);
        # Update the model without adagrad
        #w = w + eta * g;
        # Use the averaged model if that works better (see [\textbf{SSBD}] section 14.3)
        #wbar = (wbar*t + w)/float(t+1);
        # Take a snapshot after every few iterations
        # Take snapshots after every spacing = 5 or 10 GD iterations since they are slow
        """if t%spacing == 0:
            #Stop the timer - we want to take a snapshot
            t_now = datetime.now();
            delta = t_now - t_start;
            time_elapsed[tick] = ttot + delta.total_seconds();
            ttot = time_elapsed[tick];
            tick_vals[tick] = tick;

            temp4 = np.matmul(Xtr_y,w);
            temp5 = np.where(temp4<1,1-temp4,0);
            obj_val[tick] = (w.dot(w))*0.5 + np.sum(temp5);
            theory_time[tick] = tick_vals[tick]*spacing*d*n;

            print(obj_val[tick]),
            #print(theory_time[tick])
            print(time_elapsed[tick])
            tick = tick+1;
            #Start the timer again - training time!
            t_start = datetime.now();"""
    #Choose one of the two based on whichever works better for you
    w_final = w;
    #w_finalavg = wbar;
    #np.savetxt("obj_val_GD.dat", obj_val);
    #np.savetxt("theory_time_GD.dat", theory_time);
    #np.savetxt("time_elapsed_GD.dat", time_elapsed);
    # np.savetxt("obj_val.dat", obj_val);
    np.save("model_GD.npy", w_final);
    #np.save("model_bar_GD.npy", w_finalavg);

    #accuracy(Xts,Yts,w_final,w_finalavg,ntest)


if __name__ == '__main__':
    main()
