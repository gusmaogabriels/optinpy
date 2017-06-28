# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function
from ...finitediff import jacobian as __jacobian, hessian as __hessian
from ...linesearch import xstep as __xstep, backtracking as __backtracking, interp23 as __interp23, unimodality as __unimodality, golden_section as __golden_section
from ... import np as __np
from ... import scipy as __sp

eps = __np.finfo(__np.float64).eps
resolution = __np.finfo(__np.float64).resolution


params = {'fminunc':{'method':'newton','params':\
                {'gradient':{'max_iter':1e4},\
                 'newton':{'max_iter':1e3},\
                 'modified-newton':{'sigma':1,'max_iter':1e3},\
                 'conjugate-gradient':{'max_iter':1e3},\
                 'fletcher-reeves':{'max_iter':1e3},\
                 'quasi-newton':{'max_iter':1e3,'hessian_update':'davidon-fletcher-powell'}}},\
            'jacobian':{'algorithm':'central','epsilon':__np.sqrt(eps)},\
            'hessian':{'algorithm':'central','epsilon':__np.sqrt(eps),'initial':None},\
            'linesearch':{'method':'backtracking','params':
                {'backtracking':{'alpha':1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'interp23':{'alpha':1,'alpha_min':0.1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'unimodality':{'b':1,'threshold':1e-4,'max_iter':1e3},
                'golden-section':{'b':1,'threshold':1e-4,'max_iter':1e3}}}}  

def gradient(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        gradient step (linear convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    return -g, g, []

def newton(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        newton method (quadratic convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    return -__np.linalg.inv(Q).dot(g), g, Q

def mod_newton(fun,x0,d0,g0,Q0,*args,**kwargs):
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

def conj_gradient(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        conjugate-gradient method
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''     
    g = __jacobian(fun,x0,**params['jacobian'])
    Q = __hessian(fun,x0,**params['hessian'])
    if sum(abs(d0)) < eps or ((kwargs['iters'] + 1) % len(x0)) < eps:
        return -g, g, Q
    else:
        beta = g.T.dot(Q).dot(d0)/(d0.T.dot(Q).dot(d0))
        d = -g + beta*d0
        return d, g, Q
    
def fletcher_reeves(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        fletcher_reeves
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''     
    g = __jacobian(fun,x0,**params['jacobian'])
    if sum(abs(d0)) < eps or ((kwargs['iters'] + 1) % len(x0)) < eps:
        return -g, g, []
    else:
        beta = g.T.dot(g)/(g0.T.dot(g0))
        d = -g + beta*d0
        return d, g, []
    
def dfp(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        Davidon-Flether-Powell
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''     
    g = __jacobian(fun,x0,**params['jacobian'])
    if sum(abs(d0)) < eps or ((kwargs['iters'] + 1) % len(x0)) < eps:
        Q = [params['hessian']['initial'] if params['hessian']['initial'] else __np.identity(len(x0))][0]
    else:
        q = (g-g0)[__np.newaxis].T
        p = (kwargs['alpha']*d0)[__np.newaxis].T
        Q = Q0 + __np.dot(p,p.T)/__np.dot(p.T,q) - ( __np.dot(Q0,q).dot(__np.dot(q.T,Q0)))/(__np.dot(q.T,Q0).dot(q))
    d = -Q.dot(g)
    return d, g, Q

def bfgs(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        Broyden-Fletcher-Goldfarb-Shanno
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''     
    g = __jacobian(fun,x0,**params['jacobian'])
    if sum(abs(d0)) < eps or ((kwargs['iters'] + 1) % len(x0)) < eps:
        Q = [params['hessian']['initial'] if params['hessian']['initial'] else __np.identity(len(x0))][0]
    else:
        q = (g-g0)[__np.newaxis].T
        p = (kwargs['alpha']*d0)[__np.newaxis].T
        Q = Q0 + (1.0 + q.T.dot(Q0).dot(q)/(q.T.dot(p)))*(p.dot(p.T))/(p.T.dot(q)) - (p.dot(q.T).dot(Q0)+Q0.dot(q).dot(p.T))/(q.T.dot(p))
    d = -Q.dot(g)
    return d, g, Q

def qn(fun,x0,d0,g0,Q0,*args,**kwargs):
    '''
        Quasi-Newton caller
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''     
    if params['fminunc']['params']['quasi-newton']['hessian_update'] in ('davidon-fletcher-powell','dfp'):
        return dfp(fun,x0,d0,g0,Q0,*args,**kwargs)
    elif params['fminunc']['params']['quasi-newton']['hessian_update'] in ('broyden-fletcher-goldfarb-shanno','BFGS','bfgs'):
        return bfgs(fun,x0,d0,g0,Q0,*args,**kwargs)
    else:
        raise Exception('Hessian update method ({}) not implemented'.format(params['fminunc']['params']['quasi-newton']['hessian_update']))

__ls_algorithms = {'backtracking':__backtracking,
                'interp23':__interp23,
                'unimodality':__unimodality,
                'golden-section':__golden_section}

__unc_algorithms = {'gradient':gradient,
                'newton':newton,
                'modified-newton':mod_newton,
                'conjugate-gradient':conj_gradient,
                'fletcher-reeves':fletcher_reeves,
                'quasi-newton':qn}
                    

def fminunc(fun,x0,threshold=1e-6,vectorized=False,**kwargs):
    '''
        Minimum Unconstrained Optimization
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
        ..threshold as a numeric value; threshold at which to stop the iterations
        ..**kwargs = initial_hessian : as matrix (default = identity)
        .. see unconstrained.params for further details on the methods that are being used
    '''
    alg = __ls_algorithms[params['linesearch']['method']]
    ls_kwargs = params['linesearch']['params'][params['linesearch']['method']]
    d, g, Q = __unc_algorithms[params['fminunc']['method']](fun,x0,__np.zeros(len(x0)),[],[],iters=0)
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
        d, g, _ = __unc_algorithms[params['fminunc']['method']](fun,x,d,g,Q,iters=iters,alpha=alpha)
        iters += 1
    if vectorized:
        return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
    else:
        return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}



    
