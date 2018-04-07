import csv
import numpy as np
from numpy import genfromtxt
from cvxopt import matrix, solvers
import cvxopt as cvx
import cvxopt.lapack
from scipy.sparse import csgraph
import picos as pic
import networkx as nx
from cvxpy import *




#Reading the adjacency matrix
graph = genfromtxt('Graph_1.csv', delimiter=',')
#print(graph.shape)

#Graph laplacian:
Ll = 1/4*csgraph.laplacian(graph, normed=False)
Ll = pic.tools._retrieve_matrix(Ll)
Ll = Ll[0]

#Problem setup
N = graph.shape[1]
X = Variable(N,N)
X = pic.tools._retrieve_matrix(X)
X = X[0]
objective = Maximize(trace(Ll*X))




#------Constraints----------
#X positive semidefinite
constr = [(X >> 0)]
#ones on the diagonal
diag_val = []
for i in range(N):
	diag_val.append(1)
constr1 = (diag(X) == diag_val)
constr.append(constr1)



#objective
prob = Problem(objective, constr)

print('bound from the SDP relaxation: {0}'.format(prob.solve()))


'''
#Cholesky decomposition
V=X.value

cvxopt.lapack.potrf(V)
for i in range(N):
        for j in range(i+1,N):
                V[i,j]=0


#random projection algorithm
#Repeat 100 times or until we are within a factor .878 of the SDP optimal value
count=0
obj_sdp=maxcut.obj_value()
obj=0
while (count <2 or obj<.878*obj_sdp):
        r=cvx.normal(N,1)
        #print(V,r)
        x=cvx.matrix(np.sign(V*r))
        o=(x.T*L*x).value[0]
        if o>obj:
                x_cut=x
                obj=o
        count+=1


#original Objectve value from randomized rounding
print("Objective value:", o)


#Preparing for output
for i in range(len(x)):
	if x[i] == -1:
		x[i] = 0
	x[i] = int(x[i])

final_x = list(map(int, x))
#print(final_x)


#Output
with open('6_sol', 'w') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(final_x)

'''




