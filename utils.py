import numpy as np
import matplotlib.pyplot as plt
def linear_kernel(x, y, b=1):
	"""Returns the linear combination of arrays `x` and `y` with
	the optional bias term `b` (set to 1 by default)."""
	
	return x @ y.T + b # Note the @ operator for matrix multiplication

def gaussian_kernel(x, y, sigma=1):
	"""Returns the gaussian similarity of arrays `x` and `y` with
	kernel width parameter `sigma` (set to 1 by default)."""
	
	if np.ndim(x) == 1 and np.ndim(y) == 1:
		result = np.exp(- (np.linalg.norm(x - y, 2)) ** 2 / (2 * sigma ** 2))
	elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
		result = np.exp(- (np.linalg.norm(x - y, 2, axis=1) ** 2) / (2 * sigma ** 2))
	elif np.ndim(x) > 1 and np.ndim(y) > 1:
		result = np.exp(- (np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], 2, axis=2) ** 2) / (2 * sigma ** 2))
	return result

def decision_function(alphas, target, kernel, X_train, x_test, b):
	"""Applies the SVM decision function to the input feature vectors in `x_test`."""
	
	result = (alphas * target) @ kernel(X_train, x_test) - b
	return result

def plot_decision_boundary_0(model, ax, resolution=100, colors=('b', 'k', 'r'), levels=(-1, 0, 1)):
		"""Plots the model's decision boundary on the input axes object.
		Range of decision boundary grid is determined by the training data.
		Returns decision boundary grid and axes object (`grid`, `ax`)."""
		
		# Generate coordinate grid of shape [resolution x resolution]
		# and evaluate the model over the entire space
		xrange = np.linspace(model.X[:,0].min(), model.X[:,0].max(), resolution)
		yrange = np.linspace(model.X[:,1].min(), model.X[:,1].max(), resolution)
		grid = [[decision_function(model.alphas, model.y,
								   model.kernel, model.X,
								   np.array([xr, yr]), model.b) for xr in xrange] for yr in yrange]
		grid = np.array(grid).reshape(len(xrange), len(yrange))
		
		# Plot decision contours using grid and
		# make a scatter plot of training data
		ax.contour(xrange, yrange, grid, levels=levels, linewidths=(1, 1, 1),
				   linestyles=('--', '-', '--'), colors=colors)
		ax.scatter(model.X[:,0], model.X[:,1],
				   c=model.y, cmap=plt.cm.viridis, lw=0, alpha=0.25)
		
		# Plot support vectors (non-zero alphas)
		# as circled points (linewidth > 0)
		mask = np.round(model.alphas, decimals=2) != 0.0
		ax.scatter(model.X[mask,0], model.X[mask,1],
				   c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')
		
		return grid, ax



def plot_twoclass_data(X,y,xlabel,ylabel,legend):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_adjustable('box')
	X0 = X[np.where(y==-1)]
	X1 = X[np.where(y==1)]
	plt.scatter(X0[:,0],X0[:,1],c='red', s=30, label = legend[0], alpha=0.25)
	plt.scatter(X1[:,0],X1[:,1],c='green', s = 30, label=legend[1], alpha=0.25)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc="upper right")
	return ax

def plot_decision_boundary(X,y,clf,  xlabel, ylabel, legend):

	plot_twoclass_data(X,y,xlabel,ylabel,legend)
	
	# create a mesh to plot in

	h = 0.01
	x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
	x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
						 np.arange(x2_min, x2_max, h))

	# make predictions on this mesh (but add intercept term)
	Z = np.array(clf.predict(np.c_[np.ones((xx1.ravel().shape[0],)), xx1.ravel(), xx2.ravel()]))

	# Put the result into a color contour plot
	Z = Z.reshape(xx1.shape)
	plt.contour(xx1,xx2,Z,cmap=plt.cm.viridis,levels=[0])




def plot_decision_kernel_boundary(X,y,scaler, sigma, clf,  xlabel, ylabel, legend):

	ax = plot_twoclass_data(X,y,xlabel,ylabel,legend)
	ax.autoscale(False)

	# create a mesh to plot in

	h = 0.05
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
						 np.arange(x2_min, x2_max, h))

	ZZ = np.array(np.c_[xx1.ravel(), xx2.ravel()])
	K = np.array([gaussian_kernel(x1,x2,sigma) for x1 in ZZ for x2 in X]).reshape((ZZ.shape[0],X.shape[0]))

	# need to scale it
	scaleK = scaler.transform(K)

	KK = np.vstack([np.ones((scaleK.shape[0],)),scaleK.T]).T

	# make predictions on this mesh (but add intercept term)
	Z = clf.predict(KK)

	# Put the result into a color contour plot
	Z = Z.reshape(xx1.shape)
	plt.contour(xx1,xx2,Z,cmap=plt.cm.gray,levels=[0.5])
