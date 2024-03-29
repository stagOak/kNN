# kNN
Implement kNN algo

This assignment is about replicating an existing algorithm from scratch. You are 
also required to apply the algorithm on two known datasets: IRIS and Titanic.

1)	K-NN
	a.	Must be able to accept both numeric and categorical features. 
	b.	Must at least perform classification, regression is optional. 
	c.	Use Gower distance (Minkowski’s for continuous and Jaccard for 
		categorical) 
	d.	Use Titanic data to predict survival and IRIS to predict type

From Wikipedia:

In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a 
non-parametric method used for classification and regression. 

In both cases, the input consists of the k closest training examples in the 
feature space. The output depends on whether k-NN is used for classification or 
regression:

        In k-NN classification, the output is a class membership. An object is 
		classified by a majority vote of its neighbors, with the object being 
		assigned to the class most common among its k nearest neighbors (k is a 
		positive integer, typically small). If k = 1, then the object is simply 
		assigned to the class of that single nearest neighbor.

        In k-NN regression, the output is the property value for the object. 
		This value is the average of the values of its k nearest neighbors.

k-NN is a type of instance-based learning, or lazy learning, where the function
is only approximated locally and all computation is deferred until 
classification. The k-NN algorithm is among the simplest of all machine learning
algorithms.

Both for classification and regression, a useful technique can be to assign 
weight to the contributions of the neighbors, so that the nearer neighbors 
contribute more to the average than the more distant ones. For example, a common
weighting scheme consists in giving each neighbor a weight of 1/d, where d is 
the distance to the neighbor.

The neighbors are taken from a set of objects for which the class 
(for k-NN classification) or the object property value (for k-NN regression) is
known. This can be thought of as the training set for the algorithm, though no 
explicit training step is required.

A peculiarity of the k-NN algorithm is that it is sensitive to the local 
structure of the data. The algorithm is not to be confused with k-means, another
popular machine learning technique.

Algorithm

The training examples are vectors in a multidimensional feature space, each with
a class label. The training phase of the algorithm consists only of storing the 
feature vectors and class labels of the training samples.

In the classification phase, k is a user-defined constant, and an unlabeled 
vector (a query or test point) is classified by assigning the label which is 
most frequent among the k training samples nearest to that query point.

A commonly used distance metric for continuous variables is Euclidean distance. 
For discrete variables, such as for text classification, another metric can be 
used, such as the overlap metric (or Hamming distance). In the context of gene 
expression microarray data, for example, k-NN has also been employed with 
correlation coefficients such as Pearson and Spearman. Often, the classification
accuracy of k-NN can be improved significantly if the distance metric is learned
with specialized algorithms such as Large Margin Nearest Neighbor or 
Neighbourhood components analysis.

A drawback of the basic "majority voting" classification occurs when the class 
distribution is skewed. That is, examples of a more frequent class tend to 
dominate the prediction of the new example, because they tend to be common among
the k nearest neighbors due to their large number. One way to overcome this 
problem is to weight the classification, taking into account the distance from 
the test point to each of its k nearest neighbors. The class (or value, in 
regression problems) of each of the k nearest points is multiplied by a weight 
proportional to the inverse of the distance from that point to the test point. 
Another way to overcome skew is by abstraction in data representation. For 
example, in a self-organizing map (SOM), each node is a representative 
(a center) of a cluster of similar points, regardless of their density in the 
original training data. K-NN can then be applied to the SOM.