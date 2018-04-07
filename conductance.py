import csv
import sys
import numpy as np
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


#Reading the adjacency matrix
graph = genfromtxt('Graph_2.csv', delimiter=',')
N = graph.shape[1]


#Graph Normalized laplacian:
L = csgraph.laplacian(graph, normed=False)
NL = csgraph.laplacian(graph, normed=True)
D = L + graph

w, v = LA.eig(NL)

#changing to lost
w = w.tolist()
e2 = sorted(w)[-2]
pos = w.index(e2)
#print(e2, pos)
pos_vec = w.index(sorted(w)[3])

ev = v[pos_vec]

#finding opt y
#print(ev)
D_half = fractional_matrix_power(D, 0.5)
y = D_half.dot(ev)
#print(y)

print("upper bound:", 2/e2)


#finding optimal cut-off
a = max(y)
b = min(y)
step = (a-b)/20
print("max,min:",a,b)
obj = 0
temp_obj = 0
val = 0
i = b + step

for k in range(19):
	vec = y.copy()
	for j in range(len(y)):
		if vec[j] < i:
			vec[j] = 0
		else:
			vec[j] = 1
	#print(y)
	vol = min(np.sum(vec), N - np.sum(vec))
	cut = (vec.transpose()).dot(L.dot(vec))
	#print(vol, cut)
	obj = vol/cut
	if obj > temp_obj:
		val = i
		temp_obj = obj
	i = i + step
	print(val, obj)

for j in range(len(y)):
	if y[j] < val:
		y[j] = 0
	else:
		y[j] = 1

#print(y)


final_x = list(map(int, y))
#print(final_x)


'''
#Output
with open('6_cond_sol', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)
'''






















