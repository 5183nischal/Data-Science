import csv
import sys
import numpy as np
from random import shuffle
from numpy import genfromtxt
from cvxopt import matrix, solvers
import cvxopt as cvx
import cvxopt.lapack
from cvxopt import matrix, spmatrix, sparse
from scipy.sparse import csgraph
import picos as pic
from scipy.sparse import *
from scipy import *
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power
import math
import itertools
import networkx as nx


#Reading the adjacency matrix
graph = genfromtxt('Graph_6.csv', delimiter=',')
N = graph.shape[1]
#print(graph.shape)

#Graph laplacian:
Ll = 1/4*csgraph.laplacian(graph, normed=False)

#to find diagonal
D = graph.dot(graph)
diag = diag(D)
diag = diag.tolist()
sort = sorted(diag)


#the nodes:
x = []
for i in range(int(N/5)):
	pos = diag.index(sort[i])
	diag[pos] = -1
	x.append(pos)

print(x)
y = np.zeros(N)
for i in range(len(y)):
	y[i] = -1
for i in range(len(x)):
	y[x[i]] = 1

obj = y.transpose().dot(Ll.dot(y))
opt = 1/obj
print("optimized val", opt)

for i in range(len(y)):
	if y[i] == -1:
		y[i] = 0
	y[i] = int(y[i])

final_x = list(map(int, y))
#print(final_x)


#Output
with open('6_cluster_sol', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)





