import numpy as np
from scipy.sparse import csr_matrix
import sys
from sklearn.datasets import load_svmlight_file
import random
from datetime import datetime
import math


# def accuracy(Xts,Yts,w_final, ntest):
# 	Yp1 = np.matmul(Xts,w_final);

# 	Yp1 = np.multiply(Yp1,Yts);

# 	Yp1 = np.where(Yp1>0,1,0);

# 	acy_nor = np.sum(Yp1);

# 	print("accuracy nor :"),
# 	print(100*(acy_nor/float(ntest)))


def main():
	traindatafile = sys.argv[1];
	# For how many iterations do we wish to execute SCD?
	n_iter = int(sys.argv[2]);
	# After how many iterations do we want to timestamp?
	spacing = int(sys.argv[3]);

	# The training file is in libSVM format
	tr_data = load_svmlight_file(traindatafile);
	##

	Xtr = tr_data[0]; # Training features in sparse format
	Ytr = tr_data[1]; # Training labels
	# We have n data points each in d-dimensions
	n, d = Xtr.get_shape();
	# The labels are named 1 and 2 in the data set. Convert them to our standard -1 and 1 labels
	#Xts = Xtr[int(0.85*n):]
	#Xtr = Xtr[:int(0.85*n)]
	Ytr = 2*(Ytr - 1.5);
	Ytr = Ytr.astype(int);
	#Yts = Ytr[int(0.85*n):]
	#Ytr = Ytr[:int(0.85*n)]

	# Optional: densify the features matrix.
	# Warning: will slow down computation
	#n, d = Xtr.get_shape();
	#ntest, dtest = Xts.get_shape();
	Xtr = Xtr.toarray();
	#Xts = Xts.toarray();
	# Initialize model
	# For dual SCD, you will need to maintain d_alpha and w
	# Note: if you have densified the Xt matrix then you can initialize w as a NumPy array
	# w = csr_matrix((1, d));
	w = np.zeros(d)
	# wbar = np.ones(d)
	d_alpha = np.zeros((n,));
	# We will take timestamp after every "spacing" iterations

	time_elapsed = np.zeros(int(math.ceil(n_iter/spacing)));
	tick_vals = np.zeros(int(math.ceil(n_iter/spacing)));
	obj_val = np.zeros(int(math.ceil(n_iter/spacing)));
	f_alpha = np.zeros(int(math.ceil(n_iter/spacing)));
	theory_time = np.zeros(int(math.ceil(n_iter/spacing)));
	tick = 0;
	ttot = 0.0;
	t_start = datetime.now();
	# print type(Xtr)

	# print(Xtr[1].shape)
	# print(Ytr[1].shape)
	# exit()
	# temp = (Xtr.T*Ytr).T;
	# Q = temp.dot(temp.T)
	Xtr_y = (Xtr.T*Ytr).T;
	# print(Xtr.shape)
	# print(temp.shape)
	# exit()
	for t in range(n_iter):
		i_rand = random.randint(0,n-1);
		# Store the old and compute the new value of alpha along that coordinate
		d_alpha_old = d_alpha[i_rand];

		qii = Xtr_y[i_rand].dot(Xtr_y[i_rand]);

		ret = (w).dot(Xtr_y[i_rand]);
		#print(ret.shape)
		#print(qii.shape)
		#exit()

		# xixj = Xtr[i_rand].dot(Xtr.T);
		# axy = np.matrix(np.multiply(Ytr,xixj.T));
		# alphaaxy = np.multiply(d_alpha,axy.T);
		# summation = np.ones((1,n)).dot(alphaaxy);
		# ret = Ytr[i_rand]*summation;


		d_alpha[i_rand] = min(max(d_alpha_old - 0.1*(ret-1)/float(qii),0),1);

		#Projection step
		# if(d_alpha[i_rand]<0):
		# d_alpha[i_rand]=0
		# elif(d_alpha[i_rand]>1):
		# d_alpha[i_rand]=1


		# Update the model - takes only O(d) time!
		w = w + (d_alpha[i_rand] - d_alpha_old)*Ytr[i_rand]*Xtr[i_rand];

		"""if t%spacing == 0:
			# Stop the timer - we want to take a snapshot
			t_now = datetime.now();
			delta = t_now - t_start;
			time_elapsed[tick] = ttot + delta.total_seconds();
			ttot = time_elapsed[tick];
			tick_vals[tick] = tick;
			# print((d_alpha.T).shape)
			# print(Q.shape)
			# temp1 = np.matmul(d_alpha.T,Q);
			# print(temp1.shape)
			# temp2 = np.matmul(temp1,d_alpha);
			# temp2 = temp2/2.0;
			# print(temp2.shape)

			# print(temp2)
			# s= np.sum(d_alpha);
			# print(s)
			# exit()

			temp4 = np.matmul(Xtr_y,w);
			temp5 = np.where(temp4<1,1-temp4,0);

			temp6 = d_alpha.dot(Xtr_y);

			temp7 = temp6.dot(temp6)
			f_alpha[tick] = temp7 - np.sum(d_alpha);
			obj_val[tick] = (w.dot(w))*0.5 + np.sum(temp5);
			theory_time[tick] = tick_vals[tick]*spacing*d;


			# Calculate the objective value f(w) for the current model w^t
			#print(f_alpha[tick]),
			print(obj_val[tick]),
			print(time_elapsed[tick])
			#print(theory_time[tick]),
			tick = tick+1;
			# Start the timer again - training time!
			t_start = datetime.now();"""

	w_final = w;

	#np.savetxt("obj_val_SCD.dat", obj_val);
	#np.savetxt("f_alpha_SCD.dat", f_alpha);
	#np.savetxt("time_elapsed_SCD.dat", time_elapsed);
	#np.savetxt("theory_time_SCD.dat", theory_time);
	np.save("model_SCD.npy", w_final);

	#accuracy(Xts,Yts,w_final,ntest)



if __name__ == '__main__':
	main()

