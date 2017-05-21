# -*- coding: utf-8 -*-

from ...finitediff import jacobian as __jacobian, hessian as __hessian
from ...linesearch import xstep as __xstep, backtracking as __backtracking, interp23 as __interp23, unimodality as __unimodality, golden_section as __golden_section
from ... import np as __np
from ... import scipy as __sp

params = {'fminunc':{'method':'newton','params':\
                {'gradient':{'max_iter':1e3},\
                 'newton':{'max_iter':1e3},\
                 'modified-newton':{'sigma':1,'max_iter':1e3},\
                 'conjugate-gradient':{'max_iter':1e3}}},\
            'jacobian':{'algorithm':'central','epsilon':1e-6},\
            'hessian':{'algorithm':'central','epsilon':1e-6},\
            'linesearch':{'method':'backtracking','params':
                {'backtracking':{'alpha':1,'rho':0.5,'c':1e-3,'max_iter':1e3},\
                'interp23':{'alpha':1,'alpha_min':0.1,'rho':0.5,'c':1e-3,'max_iter':1e3},\
                'unimodality':{'b':1,'threshold':1e-4,'max_iter':1e3},
                'golden-section':{'b':1,'threshold':1e-4,'max_iter':1e3}}}}  

def gradient(fun,x0,*args,**kwargs):
    '''
        gradient step (linear convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    return -g, g, []

def newton(fun,x0,*args,**kwargs):
    '''
        newton method (quadratic convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    return -__np.linalg.inv(Q).dot(g), g, Q

def mod_newton(fun,x0,*args,**kwargs):
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

def conj_gradient(fun,x0,d,*args,**kwargs):
    '''
        conjugate-gradient method
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''        
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    if sum(abs(d)) == 0 or kwargs['iters'] + 1 % len(x0) == 0:
        return -g, g, Q
    else:
        beta = g.T.dot(Q).dot(d)/d.T.dot(Q).dot(d)
        d = -g + beta*d
        return d, g, Q

__ls_algorithms = {'backtracking':__backtracking,
                'interp23':__interp23,
                'unimodality':__unimodality,
                'golden-section':__golden_section}

__unc_algorithms = {'gradient':gradient,
                'newton':newton,
                'modified-newton':mod_newton,
                'conjugate-gradient':conj_gradient}
                    

def fminunc(fun,x0,threshold,vectorized=False,**kwargs):
    '''
        Minimum Unconstrained Optimization
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
        ..threshold as a numeric value; threshold at which to stop the iterations
        .. see unconstrained.params for further details on the methods that are being used
    '''
    alg = __ls_algorithms[params['linesearch']['method']]
    ls_kwargs = params['linesearch']['params'][params['linesearch']['method']]
    d, g, _ = __unc_algorithms[params['fminunc']['method']](fun,x0,__np.zeros(len(x0)),iters=0)
    if kwargs.has_key('max_iter'):
        max_iter= kwargs['max_iter']
    else:
        max_iter = params['fminunc']['params'][params['fminunc']['method']]['max_iter']
    if vectorized:
        x_vec = [x0]
    else:
        pass
    x = x0
    iters = 0
    lsiters = 0
    while __np.dot(g,g) > threshold and iters < max_iter:
        ls = alg(fun,x,d,**ls_kwargs)
        alpha = ls['alpha']
        lsiters += ls['iterations']
        #Q = __hessian(fun,x0,**params['hessian'])
        #alpha = g.T.dot(g)/(g.T.dot(Q).dot(g))
        x = __xstep(x,d,alpha)
        if vectorized:
            x_vec += [x]
        else:
            pass
        d, g, _ = __unc_algorithms[params['fminunc']['method']](fun,x,d,iters=iters)
        iters += 1
    if vectorized:
        return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
    else:
        return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}



    
