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
import random
import itertools
import networkx as nx



#Reading the adjacency matrix
graph = genfromtxt('Graph_5.csv', delimiter=',')
N = graph.shape[1]
#to find diagonal
D = graph.dot(graph)
diag = diag(D)
diag = diag.tolist()
sort = sorted(diag)


#eigen value:
w, v = LA.eig(graph)

#changing to lost
w = w.tolist()
e = sorted(w)[-1]
ub = e +1
print("upper bound:", ub)
print("max diag +1:", max(diag)+1)


for i in range(100):
	graph = np.delete(graph, (0), axis=0)
	graph = np.delete(graph, (0), axis=1)



print(graph.shape)
G=nx.from_numpy_matrix(graph)
cl = nx.find_cliques(G)
ans = list(cl)

length = 0
pos = 0
for i in range(len(ans)):
	if len(ans[i])>length:
		length = len(ans[i])
		pos = i

clique_final = ans[pos]
print("brute force answer:", len(clique_final))

x = np.zeros(N)

for i in clique_final:
	x[i] = 1

final_x = list(map(int, x))
#print(final_x)


#Output
with open('6_clique_sol', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)

	





