# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:24:08 2016

@author: GABRIS46
  """

import optinpy

A = [[-3,2,-4,-5],[4,-2,5,3],[2,4,1,2],[3,2,-2,4]]
b = [-10,10,10,15]
c = [-2,-2,-3,-3]

S = optinpy.simplex(A,b,c,lb=[-10,-10,-10,-10],ub=[5,5,5,5])
f = lambda x : (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
f_aurea = lambda x : np.exp(-x[0])+x[0]**2
print 'backtracking\n', optinpy.linesearch.backtracking(f,[0,0],-optinpy.finitediff.jacobian(f,[0,0]),1,rho=0.6)
print 'backtracking\n', optinpy.linesearch.backtracking(f,[0,0],-optinpy.finitediff.jacobian(f,[0,0]),1,rho=0.5)
print 'interpolate\n', optinpy.linesearch.interp23(f,[0,0],-optinpy.finitediff.jacobian(f,[0,0]),1)
print 'unimodality\n', optinpy.linesearch.unimodality(f,[0,0],-optinpy.finitediff.jacobian(f,[0,0]),1)
print 'golden_ratio\n', optinpy.linesearch.golden_ratio(f,[0,0],-optinpy.finitediff.jacobian(f,[0,0]),1)
print 'golden_ratio\n', optinpy.linesearch.golden_ratio(f_aurea,[0],-optinpy.finitediff.jacobian(f,[0,0]),1)
#optinpy.nonlinear.unconstrained.params['linesearch']['method'] ='golden_ratio'
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'gradient'
print 'gradient_descent\n', optinpy.nonlinear.unconstrained.fminunc(f,[0,0],0.001)
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'newton'
print 'Newton\n', optinpy.nonlinear.unconstrained.fminunc(f,[0,0],0.001)
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'modified-newton'
print 'Modified-Newton\n', optinpy.nonlinear.unconstrained.fminunc(f,[0,0],0.001)
#S.dual()
"""
n = optinpy.graph()
bs = zip(range(1,6),[5,-8,0,10,-7])
for b in bs:
    n.add_node(*b)
connections = [[1,2,6],[1,3,5],[1,4,10],[2,5,1],[2,3,2],[3,5,7],[4,2,10],[4,5,8]]
for c in connections:
    n.add_connection(*c)
optinpy.mcfp.bigm(n,100)

 # Minimum-Spanning Trees
n = optinpy.graph()
connections = [['A','B',7],['A','D',5],['B','C',8],['B','D',9],['B','E',7],['C','E',5],\
                ['D','E',15],['D','F',6],['E','F',8],['E','G',9],['F','G',11]]
for c in connections:
    n.add_connection(*c)
n2 = optinpy.graph()
connections = [['A','B',13],['A','C',6],['B','C',7],['B','D',1],['C','E',8],['C','D',14],\
                ['C','H',20],['D','E',9],['D','F',3],['E','F',2],['E','J',18],['G','H',15],\
                ['G','I',5],['G','J',19],['G','K',10],['H','J',17],['I','K',11],['J','K',16],\
                ['J','L',4],['K','L',12]]
for c in connections:
    n2.add_connection(*c)
print("Prims's")
arcs1 = optinpy.mst.prim(n2,'A',verbose=True)
print("Kruskal's")
arcs2 = optinpy.mst.kruskal(n2,verbose=True)
print("Boruvka's")
arcs3 = optinpy.mst.boruvka(n2,verbose=True)
"""
""" #SHORTEST PATH
n = optinpy.graph()
connections = [[1,2,2],[1,3,4],[1,4,5],[2,4,2],[3,4,1]]
for c in connections:
    n.add_connection(*c)
rot,ds = optinpy.sp.fmb(n,1,verbose=True)    
rot,ds = optinpy.sp.dijkstra(n,1,verbose=True)

# https://www.youtube.com/watch?v=obWXjtg0L64
n = optinpy.graph()
connections = [[1,2,10],[1,6,8],[2,4,2],[3,2,1],[4,3,-2],[5,2,-4],\
                [5,4,-1],[6,5,1]]
for c in connections:
    n.add_connection(*c)
rot,ds = optinpy.sp.fmb(n,1,verbose=True)
# https://www.youtube.com/watch?v=gdmfOwyQlcI
n = optinpy.graph()
connections = [['A','B',4],['A','E',7],['A','C',3],['B','C',6],['B','D',5],['C','D',11],\
                ['C','E',8],['D','E',2],['D','F',2],['D','G',10],['E','G',5]]
for c in connections:
    n.add_connection(*c)
rot,ds  = optinpy.sp.dijkstra(n,'A',verbose=True)

# BIG-M
n3 = optinpy.graph()
bs = zip(range(1,6),[10,4,0,-6,-8])
for b in bs:
    n3.add_node(*b)
connections = [[1,2,1],[1,3,8],[1,4,1],[2,3,2],[3,4,1],[3,5,4],[4,5,12],[5,2,7]]
for c in connections:
    n3.add_connection(*c)

optinpy.bigm(n3,20)
"""