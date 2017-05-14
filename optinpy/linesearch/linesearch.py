# -*- coding: utf-8 -*-

from .. import np
from ..finitediff import jacobian as __jacobian, hessian as __hessian

def xstep(x0,d,alpha):
    '''
        returns x_1 given x_0, d and alpha
    '''
    return map(sum,zip(x0,[i*alpha for i in d]))

def __armijo(fun,x0,d,dd,alpha,c):
    '''
        Check's whether Armijo's conditions upholds
        i.e. fun(x0+alpha*d) <= fun(x0) + c*alpha*dd
    '''
    return fun(xstep(x0,d,alpha)) <= fun(x0)+c*alpha*dd

def backtracking(fun,x0,d=None,alpha=1,rho=0.5,c=1e-4,**kwargs):
    '''
        backtracking algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the __jacobian should be estimated
        ..d as a numeric array: direction towards which to walk
        ..alpha as a numeric value; maximum step from x_i
        ..rho as a numeric value; rate of decrement in alpha
        ..c as a numeric values; constant of Wolfe's condition
        ..**kwargs as a dictionary with the jacobian function parameters
    '''
    if any(d):
        pass
        dd = np.dot(d,__jacobian(fun,x0,**kwargs))     
    else:
        d = -np.array(__jacobian(fun,x0,**kwargs)) # steepest descent step
        dd = -np.dot(d,d) 
    iters = 0
    while not __armijo(fun,x0,d,dd,alpha,c):# Armijo's Condition
        alpha = rho*alpha
        iters += 1
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}

def interp23(fun,x0,d=None,alpha=1,c=1e-4,alpha_min=0.1,rho=0.5,**kwargs):
    '''
        interpolating algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the __jacobian should be estimated
        ..d as a numeric array: direction towards which to walk
        ..alpha as a numeric value; maximum step from x_i
        ..c as a numeric values; constant of Armijo's condition
        ..alpha_min; lower limit for alpha when alpha is too small
        ..**kwargs as a dictionary with the jacobian function parameters
    '''
    alpha0 = alpha
    if any(d):
        pass
        dd = np.dot(d,__jacobian(fun,x0,**kwargs))     
    else:
        d = -np.array(__jacobian(fun,x0,**kwargs)) # steepest descent step
        dd = -np.dot(d,d)
    iters = {'first_order':0,'second_order':0,'third_order':0}
    iters['first_order'] += 1
    if __armijo(fun,x0,d,dd,alpha0,c):
        return {'x':xstep(x0,d,alpha0), 'f':fun(xstep(x0,d,alpha0)), 'alpha':alpha0, 'iterations':iters}
    else:
        # second order approximation
        iters['second_order'] += 1
        alpha1 = -(dd*alpha0**2)/(2*(fun(xstep(x0,d,alpha0))-fun(x0)-dd*alpha0))
        if __armijo(fun,x0,d,dd,alpha1,c) and alpha1 > alpha_min:# Armijo's Condition
            return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1, 'iterations':iters}
        else:
            alpha1 = alpha0*rho
            iters['second_order'] += 1
            if __armijo(fun,x0,d,dd,alpha1,c) and alpha1 > alpha_min: # check whether a single backtracking iteration gives rise to a reasonable stepsize
                return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1, 'iterations':iters}
            else:
                while not __armijo(fun,x0,d,dd,alpha1,c):
                    iters['third_order'] += 1
                    coeff = (1/(alpha0**2*alpha1**2*(alpha1-alpha0)))
                    m = [[alpha0**2,-alpha1**2],[-alpha0**3,alpha1**3]]
                    v = [fun(xstep(x0,d,alpha1))-fun(x0)-alpha1*dd,fun(xstep(x0,d,alpha0))-fun(x0)-alpha0*dd]
                    a, b = coeff*np.dot(m,v)
                    alpha0 = alpha1
                    alpha1 = (-b+(b**2-3*a*dd)**0.5)/(3*a)
            return {'x':xstep(x0,d,alpha1), 'f':fun(xstep(x0,d,alpha1)), 'alpha':alpha1, 'iterations':iters}
        
def unimodality(fun,x0,d=None,b=1,threshold=0.01):
    '''
        unimodality algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun, a callable object, be strictly quasiconvex between the interval [a,b], a0 = 0
        ..x0 as a numeric array; starting point
        ..d as a numeric array: direction towards which to walk
        ..b as the upper bound for the interval
        ..threshold min mean-relative difference between interval bounds as of which the procedeure ceases flowing
    '''
    interv = [0,b]
    if any(d):
        pass
    else:
        d = -np.array(__jacobian(fun,x0,**kwargs)) # steepest descent step
    iters = 0
    while 2*abs((interv[-1]-interv[0])/(abs(interv[-1])+abs(interv[0]))) > threshold:
        iters += 1
        x = np.sort(np.random.uniform(*interv+[2])) # evaluate alpha and beta uniformily distributed in the interv
        phi = [fun(xstep(x0,d,x_)) for x_ in x]
        if phi[0] > phi[1]:
            interv[0] = x[0]
        else:
            interv[1] = x[1]
    alpha = (interv[1]+interv[0])/2.0
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}
                    
def golden_ratio(fun,x0,d=None,b=1,threshold=0.01):
    '''
        golden-ration algorithm, arg_min(alpha) fun(x0+alpha*d)
        ..fun, a callable object (function)
        ..x0 as a numeric array; starting point
        ..d as a numeric array: direction towards which to walk
        ..b as the upper bound for the alpha domain in Re (b>0)
        ..threshold min mean-relative difference between interval bounds as of which the procedeure ceases flowing
    '''
    if any(d):
        pass
    else:
        d = -np.array(__jacobian(fun,x0,**kwargs)) # steepest descent step
    r = (5**.5-1)/2.0
    interv = [0,b]
    alpha = interv[0] + (1-r)*b
    beta = interv[0] + r*b
    phi = [fun(xstep(x0,d,x_)) for x_ in [alpha,beta]]     
    iters = 0
    while 2*abs((interv[-1]-interv[0])/(abs(interv[-1])+abs(interv[0]))) > threshold:
        iters += 1
        if phi[0] > phi[1]:
            interv[0] = alpha
            alpha = beta
            phi[0] = phi[1]
            beta = alpha + r*(interv[1]-interv[0])
            phi[1] = fun(xstep(x0,d,beta))
        else:
            interv[1] = beta
            beta = alpha
            phi[1] = phi[0]
            alpha = interv[0] + (1-r)*(interv[1]-interv[0])
            phi[0] = fun(xstep(x0,d,alpha))
    alpha = (interv[1]+interv[0])/2.0
    return {'x':xstep(x0,d,alpha), 'f':fun(xstep(x0,d,alpha)), 'alpha':alpha, 'iterations':iters}
            
    
    
    
    
        
