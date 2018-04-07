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



#Reading the adjacency matrix
graph = genfromtxt('Graph_6.csv', delimiter=',')
N = graph.shape[1]


#Graph laplacian:
L = 1/4*csgraph.laplacian(graph, normed=False)

#to find degree
D = graph.dot(graph)

dg = D.diagonal()
edg = np.sum(dg)/2

x = np.zeros(N)
for i in range(N):
	if i % 2 == 0:
		x[i] = -1
	else:
		x[i] = 1

flag = True
while flag:
	a = 0
	for i in range(N):
		deg = dg[i]
		row = graph[i,:]
		cnt = 0
		for j in range(len(row)):
			#print(row[j], x[j])
			if row[j] == 1 and x[j] == -1*x[i]:
				cnt += 1
		#print(deg, cnt)
		if 2*cnt < deg :
			x[i] = -1*x[i]
			a += 1
	print(a)
	if a == 0:
		flag = False

#print(x)
np.resize(x,(N,1))
obj = (x.transpose()).dot(L.dot(x))
print("objective;",obj)
print("upper bound;",obj*2)
print("no. of edges", edg)


#Preparing for output
for i in range(len(x)):
	if x[i] == -1:
		x[i] = 0
	x[i] = int(x[i])

final_x = list(map(int, x))
#print(final_x)


#Output
with open('6_sol_greedy', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)






