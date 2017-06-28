# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function
from .. import np as __np
eps = __np.finfo(__np.float64).eps
resolution = __np.finfo(__np.float64).resolution

def jacobian(fun,x0,epsilon=__np.sqrt(eps),algorithm='central'):
    '''
        Jacobian calculator
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the jacobian should be estimated
        ..espilon as a numeric value; perturbation from which the jacobian is estimated
        ..algorithm as a string among 'central', 'forward', 'backward'; algorithm to be used to estimate the jacobian
    '''
    grad = []
    if algorithm == 'central':
        evpoints = [-1,1]
    elif algorithm == 'forward':
        evpoints = [0,1]
    elif algorithm == 'backward':
        evpoints = [-1,0]
    else:
        raise Exception("Algorithm must be either 'central', 'forward' or 'backward'.")    
    for i in range(len(x0)):
        fvals = []
        for _x in evpoints:
            x0[i] += _x*epsilon
            fvals += [fun(x0)]
            x0[i] -= _x*epsilon
        grad += [(fvals[1]-fvals[0])/((evpoints[1]-evpoints[0])*epsilon)]
    return __np.array(grad,__np.float64).copy()

def hessian(fun,x0,epsilon=__np.sqrt(eps),algorithm='central',**kwargs):
    '''
        hessian calculator
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point at which the hessian should be estimated
        ..espilon as a numeric value; perturbation from which the hessian is estimated
        ..algorithm as a string among 'central', 'forward', 'backward'; algorithm to be used to estimate the hessian
    '''
    hessian = [[None for j in range(len(x0))] for i in range(len(x0))]
    if algorithm == 'central':
        evpoints_ij = [[1,1],[1,-1],[-1,1],[-1,-1]]
        evpoints_ii = [[2,0],[1,0],[0,0],[-1,0],[-2,0]]
        coeffs_ij = [1,-1,-1,1]
        coeffs_ii = [-1,16,-30,16,-1]
        denom_ij = 4*epsilon**2
        denom_ii = 12*epsilon**2
    elif algorithm == 'forward':
        evpoints_ij = [[1,1],[1,0],[0,1],[0,0]]
        evpoints_ii = evpoints_ij
        coeffs_ii = [1,-1,-1,1]
        coeffs_ij = coeffs_ii
        denom_ij = epsilon**2
        denom_ii = denom_ij
    elif algorithm == 'backward':
        evpoints_ij = [[-1,-1],[-1,0],[0,-1],[0,0]]
        evpoints_ii = evpoints_ij
        coeffs_ii = [1,-1,-1,1]
        coeffs_ij = coeffs_ii
        denom_ij = epsilon**2
        denom_ii = denom_ij
    else:
        raise Exception("Algorithm must be either 'central', 'forward' or 'backward'.")    
    for i in range(0,len(x0)):
        fvals = []
        for points in evpoints_ii:
            x0[i] += points[0]*epsilon
            x0[i] += points[1]*epsilon
            fvals += [fun(x0)]
            x0[i] -= points[0]*epsilon
            x0[i] -= points[1]*epsilon
        hessian[i][i] = sum([fvals[k]*coeffs_ii[k] for k in range(len(fvals))])/denom_ii
        for j in range(i+1,len(x0)):
            fvals = []
            for points in evpoints_ij:
                x0[i] += points[0]*epsilon
                x0[j] += points[1]*epsilon
                fvals += [fun(x0)]
                x0[i] -= points[0]*epsilon
                x0[j] -= points[1]*epsilon
            hessian[i][j] = sum([fvals[k]*coeffs_ij[k] for k in range(len(fvals))])/denom_ij
            hessian[j][i] = hessian[i][j]
    return __np.array(hessian)