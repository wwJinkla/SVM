import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, lambda):
	"""
	SVM hinge loss function for two class problem

	Inputs:
	- theta: A numpy vector of size d containing coefficients.
	- X: A numpy array of shape mxd 
	- y: A numpy array of shape (m,) containing training labels; +1, -1
	- lambda: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to theta; an array of same shape as theta
"""

	m, d = X.shape
	grad = np.zeros(theta.shape)
	J = 0

	h = np.sum(theta*X,axis=1)
	lin = np.maximum(0,1-y*h)
	J = 1/(2.*m)*np.dot(theta,theta) + float(lambda)/m*np.sum(lin) 
	one_rows = np.where(y*h<1.)[0]
	grad = 1./m*theta -float(lambda)/m*np.sum((y[one_rows,np.newaxis]*(X[one_rows,:])),axis=0)

	return J, grad