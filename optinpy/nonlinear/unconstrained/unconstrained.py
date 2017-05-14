# -*- coding: utf-8 -*-

from ...finitediff import jacobian as __jacobian, hessian as __hessian
from ...linesearch import xstep as __xstep, backtracking as __backtracking, interp23 as __interp23, unimodality as __unimodality, golden_ratio as __golden_ratio
from ... import np as __np
from ... import scipy as __sp

params = {'fminunc':{'method':'newton','params':\
                {'gradient':{},\
                 'newton':{},\
                 'modified-newton':{'sigma':1}}},\
            'jacobian':{'algorithm':'central','epsilon':1e-6},\
            'hessian':{'algorithm':'central','epsilon':1e-6},\
            'linesearch':{'method':'backtracking','params':
                {'backtracking':{'alpha':1,'rho':0.5,'c':1e-4},\
                'interp23':{'alpha':1,'alpha_min':0.1,'rho':0.5,'c':1e-4},\
                'unimodality':{'b':1,'threshold':1e-4},
                'golden_ratio':{'b':1,'threshold':1e-4}}}}  

def gradient(fun,x0):
    '''
        gradient step (linear convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    return -g, g, []

def newton(fun,x0):
    '''
        newton method (quadratic convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    return -__np.linalg.inv(Q).dot(g), g, Q

def mod_newton(fun,x0):
    '''
        newton method
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    eigs, nu = __np.linalg.eig(Q)
    eigs = abs(eigs)
    eigs[eigs<params['fminunc']['params']['modified-newton']['sigma']] = params['fminunc']['params']['modified-newton']['sigma']
    d = __sp.linalg.cho_solve(__sp.linalg.cho_factor(nu.dot(__np.diag(eigs)).dot(nu.T)),-g)
    return d, g, Q

__ls_algorithms = {'backtracking':__backtracking,\
                'interp21':__interp23,\
                'unimodality':__unimodality,\
                'golden_ratio':__golden_ratio}

__unc_algorithms = {'gradient':gradient,\
                'newton':newton,
                'modified-newton':mod_newton}
                    

def fminunc(fun,x0,threshold):
    '''
        Minimum Unconstrained Optimization
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
        ..threshold as a numeric value; threshold at which to stop the iterations
        .. see unconstrained.params for further details on the methods that are being used
    '''
    alg = __ls_algorithms[params['linesearch']['method']]
    ls_kwargs = params['linesearch']['params'][params['linesearch']['method']]
    d, g, _ = __unc_algorithms[params['fminunc']['method']](fun,x0)
    x = x0
    iters = 0
    lsiters = 0
    while __np.dot(g,g) > threshold:
        ls = alg(fun,x,d,**ls_kwargs)
        alpha = ls['alpha']
        lsiters += ls['iterations']
        x = __xstep(x,d,alpha)
        d, g, _ = __unc_algorithms[params['fminunc']['method']](fun,x)
        iters += 1
    return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}




