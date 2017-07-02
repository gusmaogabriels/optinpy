# -*- coding: utf-8 -*-
"""
Created on Thu May 26 19:24:08 2016

@author: GABRIS46
  """

import optinpy
import numpy as np
from matplotlib import pyplot as plt

font = 'Consolas'
plt.rc('font',family=font)
plt.rc('mathtext',fontset='custom')
plt.rc('mathtext',rm=font)    
plt.rc('mathtext',it='{}:italic'.format(font))
plt.rc('mathtext',bf='{}:bold'.format(font))
plt.rc('mathtext',default='regular')

#A = [[-3,2,-4,-5],[4,-2,5,3],[2,4,1,2],[3,2,-2,4]]
#b = [-10,10,10,15]
#c = [-2,-2,-3,-3]
#S = optinpy.simplex(A,b,c,lb=[-10,-10,-10,-10],ub=[5,5,5,5])

plt.close('all')
rosen = lambda x : (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
gen = lambda x : np.exp(x[0])*(4*x[0]**2+2*x[1]**2+4*x[0]*x[1]+2*x[0]+1)
fun_ = rosen
delta = 0.0001
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = fun_(np.vstack([X.ravel(), Y.ravel()])).reshape((100,100))
# Note: the global minimum is at (1,1) in a tiny contour island
p0 = [-1.9,2]

optinpy.nonlinear.params['linesearch']['method'] ='interp23'
optinpy.nonlinear.params['linesearch']['method'] ='interp23'

"""
                                 plt.contour(X, Y, Z, np.arange(10)**5,linewidths=0.5)
plt.text(1, 1, 'O', va='center', ha='center', color='darkred', fontsize=20);
plt.text(1, 1, 'X', va='center', ha='center', color='darkred', fontsize=10);

               
f_aurea = lambda x : np.exp(-x[0])+x[0]**2
print 'backtracking\n', optinpy.linesearch.backtracking(fun_,[0,0],-optinpy.finitediff.jacobian(fun_,[0,0]),1,rho=0.6)
print 'backtracking\n', optinpy.linesearch.backtracking(fun_,[0,0],-optinpy.finitediff.jacobian(fun_,[0,0]),1,rho=0.5)
print 'interpolate\n', optinpy.linesearch.interp23(fun_,[0,0],-optinpy.finitediff.jacobian(fun_,[0,0]),1)
print 'unimodality\n', optinpy.linesearch.unimodality(fun_,[0,0],-optinpy.finitediff.jacobian(fun_,[0,0]),1)
print 'golden-section\n', optinpy.linesearch.golden_section(fun_,[0,0],-optinpy.finitediff.jacobian(fun_,[0,0]),1)

optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'gradient'
gradientdesc = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
print 1
#print 'gradient_descent\n', gradientdesc
plt.plot([i[0] for i in gradientdesc['x']],[i[1] for i in gradientdesc['x']],'-o',lw=1,ms=2,c='darkblue',mec='black',mfc='blue',label='Gradient')
print 2
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'newton'
newton = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Newton\n', newton
plt.plot([i[0] for i in newton['x']],[i[1] for i in newton['x']],'-o',lw=1,ms=2,c='darkred',mec='black',mfc='red',label='Newton')
print 3

optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'modified-newton'
optinpy.nonlinear.unconstrained.params['fminunc']['params']['modified-newton']['sigma'] = 0.5 
modnewton = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Modified-Newton\n', modnewton
plt.plot([i[0] for i in modnewton['x']],[i[1] for i in modnewton['x']],'-o',lw=1,ms=2,c='darkgreen',mec='black',mfc='green',label='Modified-Newton')
print 4
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'conjugate-gradient'
conj_gradient = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Modified-Newton\n', modnewton
plt.plot([i[0] for i in conj_gradient['x']],[i[1] for i in conj_gradient['x']],'-o',lw=1,ms=2,c='purple',mec='black',mfc='purple',label='Conjugate-Gradient')
print 5
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'fletcher-reeves'
fletcher_reeves = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Modified-Newton\n', modnewton
plt.plot([i[0] for i in conj_gradient['x']],[i[1] for i in conj_gradient['x']],'-o',lw=1,ms=2,c='turquoise',mec='black',mfc='blue',label='Fletcher-Reeves')
print 6
optinpy.nonlinear.unconstrained.params['fminunc']['method'] = 'quasi-newton'
optinpy.nonlinear.unconstrained.params['fminunc']['params']['quasi-newton']['hessian_update'] = 'davidon-fletcher-powell' 
dfp = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Modified-Newton\n', modnewton
plt.plot([i[0] for i in conj_gradient['x']],[i[1] for i in conj_gradient['x']],'-o',lw=1,ms=2,c='grey',mec='black',mfc='black',label='Davidon-Fletcher-Powell')
print 7
optinpy.nonlinear.unconstrained.params['fminunc']['params']['quasi-newton']['hessian_update'] = 'BFGS' 
bfgs = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,threshold=0.0005,vectorized=True)
#print 'Modified-Newton\n', modnewton
plt.plot([i[0] for i in conj_gradient['x']],[i[1] for i in conj_gradient['x']],'-o',lw=1,ms=2,c='gold',mec='black',mfc='black',label='BFGS')



plt.gca().set_ylabel('$x_1$')
plt.gca().set_xlabel('$x_2$')
plt.gca().set_title('rosenbrock function contour map')
plt.legend()

plt.figure()
plt.plot([i for i in gradientdesc['f']],label='Gradient',color='darkblue')
plt.plot([i for i in newton['f']],label='Newton',color='darkred')
plt.plot([i for i in modnewton['f']],label='Modified-Newton',color='darkgreen')
plt.plot([i for i in conj_gradient['f']],label='Conjugate-Gradient',color='purple')
plt.plot([i for i in fletcher_reeves['f']],label='Fletcher-Reeves',color='turquoise')
plt.plot([i for i in dfp['f']],label='Davidon-Fletcher-Powell',color='black') 
plt.plot([i for i in bfgs['f']],label='BFGS',color='gold') 
plt.gca().set_ylabel('$error$')
plt.gca().set_xlabel('iteration')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.legend()
plt.gca().set_title('Error evolution')

plt.figure()
plt.contour(X, Y, Z, np.arange(10)**5,linewidths=0.5)
plt.text(1, 1, 'O', va='center', ha='center', color='darkred', fontsize=20);
plt.text(1, 1, 'X', va='center', ha='center', color='darkred', fontsize=10);

optinpy.nonlinear.params['fmincon']['method'] = 'projected-gradient'
optinpy.nonlinear.params['fmincon']['params']['projected-gradient']['max_iter'] = 10000
optinpy.nonlinear.params['fminunc']['method'] = 'modified-newton'
optinpy.nonlinear.params['linesearch']['method'] = 'backtracking'

A = [[-2.,-1.],[-5.,3.],[2.,1.],[1.,-2.]]; b = [-1.,0.,3.,2.]

proj_grad = optinpy.nonlinear.constrained.fmincon(fun_,p0,A=A,b=b,threshold=1e-8,vectorized=True)
plt.plot([i[0] for i in proj_grad['x']],[i[1] for i in proj_grad['x']],'-o',lw=1,ms=2,c='gold',mec='black',mfc='black',label='Projected Gradient')
ax_x = plt.gca().get_xbound()
ax_y = plt.gca().get_ybound()
plt.plot(b[0]/A[0][0]-(A[0][1]/A[0][0])*x,x,'--',lw=1.,color='black',alpha=0.5)
plt.plot(b[1]/A[1][0]-(A[1][1]/A[1][0])*x,x,'--',lw=1.,color='red',alpha=0.5)
plt.plot(b[2]/A[2][0]-(A[2][1]/A[2][0])*x,x,'--',lw=1.,color='blue',alpha=0.5)
plt.plot(b[3]/A[3][0]-(A[3][1]/A[3][0])*x,x,'--',lw=1.,color='green',alpha=0.5)
plt.gca().set_xbound(ax_x)
plt.gca().set_ybound(ax_y)
plt.gca().set_xlabel('$x_1$')
plt.gca().set_ylabel('$x_2$')
plt.gca().set_title('Rosebrock function contour map')
plt.legend(loc=1)

fun284 = lambda x : x[0]**2.+x[1]**2.+x[2]**2.+x[3]**2.-2.0*x[0]-3.0*x[3]
A = -np.identity(4,np.float64); b = np.zeros(4,np.float64)
Aeq = np.array([[2.,1.,1.,4.],[1.,1.,2.,1.]],np.float64); beq = np.array([7.,6.],np.float64)
p0 = np.zeros(4,np.float64)
proj_grad284 = optinpy.nonlinear.constrained.fmincon(fun284,p0,A=A,b=b,Aeq=Aeq,beq=beq,threshold=1e-6,vectorized=True)
"""


fun278 = lambda x : (x[0]-2.)**2 + 2.*(x[1] - 4.)**2 + 3.*(x[2]-4.)**2
g = [lambda x : np.linalg.norm(x)**2-1.]
c = 1e-3
beta = 1.075
optinpy.nonlinear.params['fminunc']['method'] = 'modified-newton'
optinpy.nonlinear.params['linesearch']['method'] = 'backtracking'
optinpy.nonlinear.constrained.params['fminnlcon']['method'] = 'log-barrier'
logbarrier = optinpy.nonlinear.constrained.fminnlcon(fun278,[0.1,0.1,0.1],g,c,beta,threshold=1e-04,vectorized=True)
optinpy.nonlinear.constrained.params['fminnlcon']['method'] = 'barrier'
barrier = optinpy.nonlinear.constrained.fminnlcon(fun278,[0.1,0.1,0.1],g,c,beta,threshold=1e-4,vectorized=True)
optinpy.nonlinear.constrained.params['fminnlcon']['method'] = 'penalty'
penalty = optinpy.nonlinear.constrained.fminnlcon(fun278,[1.4,1.4,1.4],g,c,beta,threshold=1e-04,vectorized=True)

plt.figure()
plt.plot(barrier['c'][1:], [abs(i-57.124) for i in barrier['f']][1:],label='Barrier',color='darkblue')
plt.plot(logbarrier['c'][1:], [abs(i-57.124) for i in logbarrier['f']][1:],label='Log-barrier',color='darkred')
plt.plot(penalty['c'][1:],[abs(i-57.124) for i in penalty['f']][1:],label='Penalty',color='darkgreen')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xlabel('c')
plt.gca().set_ylabel('L2-norm of residue')
plt.legend()

plt.figure()
ax = plt.gca()
plt.plot([abs(i-57.124) for i in barrier['f']][1:],label='Barrier',color='darkblue')
plt.plot([abs(i-57.124) for i in logbarrier['f']][1:],label='Log-barrier',color='darkred')
plt.plot([abs(i-57.124) for i in penalty['f']][1:],label='Penalty',color='darkgreen')
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.gca().set_xlabel('iteration')
plt.gca().set_ylabel('L2-norm of residue [solid-line]')
plt.legend()
ax2 = plt.twinx()
plt.plot(barrier['err'][1:],'--',label='Barrier',color='darkblue')
plt.plot(logbarrier['err'][1:],'--',label='Log-barrier',color='darkred')
plt.plot(penalty['err'][1:],'--',label='Penalty',color='darkgreen')
plt.gca().set_yscale('log')
plt.gca().set_ylabel('P(x) or B(x) [dashed-line]')

#S.dual()
"""
plt.figure()
plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
l3, = plt.plot([abs(i) for i in modnewton['f']],label = 'Modified-Newton',c='green')
l1, = plt.plot([abs(i) for i in gradientdesc['f']],label = 'Gradient')
l2, = plt.plot([abs(i) for i in newton['f']],label = 'Newton')
for i in np.logspace(0,1.5,200):
    print i
    optinpy.nonlinear.unconstrained.params['fminunc']['params']['modified-newton']['sigma'] = i
    modnewton = optinpy.nonlinear.unconstrained.fminunc(fun_,p0,0.0005,vectorized=True)
    plt.plot([abs(j) for j in modnewton['f']],alpha=0.2+(0.8/200)*i,c='green')
plt.legend()
plt.gca().set_ybound([0.001,1e3])
"""
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