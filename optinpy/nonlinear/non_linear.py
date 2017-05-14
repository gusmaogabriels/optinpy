# -*- coding: utf-8 -*-

from . import gradient, hessian, xstep, backtracking, interp23, unimodality, golden_ration

def gradient_descent(fun,x0,threshold,jacobian_kwargs={},linesearch_kwargs={}):
    '''
        gradient descent method
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
        ..threshold as a numeric value; threshold at which to stop the iterations
        ..jacobian_kwargs as a dict object with jacobian function parameters
        ..linesearch_kwargs as a dict object with linesearch method definition and its parameters
    '''
    if len(linesearch_kwargs) == 0:
        linesearch_kwargs = {'method':'backtracking','kwargs':{'alpha':1,'rho':0.5,'c':1e-4},'jacobian_kwargs':{'algorithm':'central','epsilon':1e-6}}
    else:
        pass
    if len(jacobian_kwargs) == 0:
        jacobian_kwargs = {'algorithm':'central','epsilon':1e-6}
    else: 
        pass    
    iters = 0
    d = np.array(jacobian(fun,x0,**jacobian_kwargs))
    x = x0
    while np.dot(d,d) > threshold:
        alpha = backtracking(fun,x,**linesearch_kwargs['jacobian_kwargs'])['alpha']
        x = xstep(x,-d,alpha)
        d = np.array(jacobian(fun,x,**jacobian_kwargs))
        iters += 1
    return {'x':x, 'f':fun(x), 'iterations':iters}

f = lambda x : (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
print 'gradient_descent\n', gradient_descent(f,[0,0],0.1)