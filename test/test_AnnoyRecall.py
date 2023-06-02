import os
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from AnnoyRecall import VQRecall

# Load iris 
data = datasets.load_iris().data 
labels = datasets.load_iris().target

N = data.shape[0]
d = data.shape[1]

# Generate prototypes via kmeans 
M = 10 # number of desired prototypes 
km = KMeans(n_clusters=M)
km.fit(data)
W = km.cluster_centers_

# ** Perform Recall. 
# This uses OMP in parallel. Can change number of threads used for calculation. 
os.environ["OMP_NUM_THREADS"] = "5"
# Initialize the recall class with the data dimension, number of BMUs requested to compute, and number of trees used during Annoy index search
rec = VQRecall(d = d, nBMU = 2, nAnnoyTrees=50)
# Equivalent of sklearn fit method. 
# Reshaping to a column-vector in C-ordering is required
rec.Recall(X=data.reshape(-1,1), W=W.reshape(-1,1), XL=labels) 


## ** Results are stored in class, examine them

# BMUs of every data vector
rec.BMU
# Quantization error of every data vector 
rec.QE 

# The (i,j) indices of the CADJ matrix, plus corresponding values 
rec.CADJi
rec.CADJj
rec.CADJ 

# The receptive field of every prototype: list of lists of data indices 
len(rec.RF)
rec.RF
# The size of each receptive field 
rec.RF_Size

# The Distribution of labels in each receptive field (a dictionary)
rec.RFL_Dist 
# The winning label of each receptive field 
rec.RFL

# Purity score within each receptive field 
rec.RFL_Purity
# Unweighted average of RF purities 
rec.RFL_Purity_UOA
# Weighted avg of RF purities (weights = RF Size)
rec.RFL_Purity_WOA


