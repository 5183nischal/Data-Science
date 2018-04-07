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

flag = True

size = N
y = []
cnt = 0
flag = True
while flag:
	if sum(diag) >= (size**2 - size)/2:
		flag = False
	else:
		pos = diag.index(sort[cnt])
		cnt += 1
		y.append(pos)
		del diag[pos]
		size = size - 1

obj = N - len(y)
print("Optimized value:", obj)

ans = np.zeros(N)
x = ans.tolist()

for i in range(len(x)):
	x[i] = 1
#removing the deleted nodes
for j in y:
	x[j] = 0

final_x = list(map(int, x))
#print(final_x)


#Output
with open('6_dense_sol', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)












