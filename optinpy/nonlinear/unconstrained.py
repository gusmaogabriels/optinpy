# -*- coding: utf-8 -*-

#from _future_ import division, absolute_import, print_function
from ..finitediff import jacobian as _jacobian, hessian as _hessian
from ..linesearch import xstep as _xstep, backtracking as _backtracking, interp23 as _interp23, unimodality as _unimodality, golden_section as _golden_section
from .. import np as _np
from .. import scipy as _sp


class unconstrained(object):
    
    def __init__(self,parameters):
        self.eps = _np.finfo(_np.float64).eps
        self.resolution = _np.finfo(_np.float64).resolution
        self.params = parameters 
        self._ls_algorithms = {'backtracking':_backtracking,
                'interp23':_interp23,
                'unimodality':_unimodality,
                'golden-section':_golden_section}
        self._unc_algorithms = {'gradient':self._gradient,
                'newton':self._newton,
                'modified-newton':self._mod_newton,
                'conjugate-gradient':self._conj_gradient,
                'fletcher-reeves':self._fletcher_reeves,
                'quasi-newton':self._qn}


    def _gradient(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            gradient step (linear convergence)
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        g = _jacobian(fun,x0,**self.params['jacobian'])
        return -g, g, []
    
    def _newton(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            newton method (quadratic convergence)
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        return -_np.linalg.inv(Q).dot(g), g, Q
    
    def _mod_newton(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            newton method
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        eigs, nu = _np.linalg.eig(Q)
        eigs = abs(eigs)
        eigs[eigs<self.params['fminunc']['params']['modified-newton']['sigma']] = self.params['fminunc']['params']['modified-newton']['sigma']
        d = _sp.linalg.cho_solve(_sp.linalg.cho_factor(nu.dot(_np.diag(eigs)).dot(nu.T)),-g)
        return d, g, Q
    
    def _conj_gradient(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            conjugate-gradient method
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''     
        g = _jacobian(fun,x0,**self.params['jacobian'])
        Q = _hessian(fun,x0,**self.params['hessian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            return -g, g, Q
        else:
            beta = g.T.dot(Q).dot(d0)/(d0.T.dot(Q).dot(d0))
            d = -g + beta*d0
            return d, g, Q
        
    def _fletcher_reeves(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            fletcher_reeves
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''     
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            return -g, g, []
        else:
            beta = g.T.dot(g)/(g0.T.dot(g0))
            d = -g + beta*d0
            return d, g, []
        
    def _dfp(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Davidon-Flether-Powell
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''     
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            Q = [self.params['hessian']['initial'] if self.params['hessian']['initial'] else _np.identity(len(x0))][0]
        else:
            q = (g-g0)[_np.newaxis].T
            p = (kwargs['alpha']*d0)[_np.newaxis].T
            Q = Q0 + _np.dot(p,p.T)/_np.dot(p.T,q) - ( _np.dot(Q0,q).dot(_np.dot(q.T,Q0)))/(_np.dot(q.T,Q0).dot(q))
        d = -Q.dot(g)
        return d, g, Q
    
    def _bfgs(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Broyden-Fletcher-Goldfarb-Shanno
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''     
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if sum(abs(d0)) < self.eps or ((kwargs['iters'] + 1) % len(x0)) < self.eps:
            Q = [self.params['hessian']['initial'] if self.params['hessian']['initial'] else _np.identity(len(x0))][0]
        else:
            q = (g-g0)[_np.newaxis].T
            p = (kwargs['alpha']*d0)[_np.newaxis].T
            Q = Q0 + (1.0 + q.T.dot(Q0).dot(q)/(q.T.dot(p)))*(p.dot(p.T))/(p.T.dot(q)) - (p.dot(q.T).dot(Q0)+Q0.dot(q).dot(p.T))/(q.T.dot(p))
        d = -Q.dot(g)
        return d, g, Q
    
    def _qn(self,fun,x0,d0,g0,Q0,*args,**kwargs):
        '''
            Quasi-Newton caller
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''     
        if self.params['fminunc']['params']['quasi-newton']['hessian_update'] in ('davidon-fletcher-powell','dfp'):
            return self.dfp(fun,x0,d0,g0,Q0,*args,**kwargs)
        elif self.params['fminunc']['params']['quasi-newton']['hessian_update'] in ('broyden-fletcher-goldfarb-shanno','BFGS','bfgs'):
            return self.bfgs(fun,x0,d0,g0,Q0,*args,**kwargs)
        else:
            raise Exception('Hessian update method ({}) not implemented'.format(self.params['fminunc']['params']['quasi-newton']['hessian_update']))
    
    def fminunc(self,fun,x0,threshold=1e-6,vectorized=False,**kwargs):
        '''
            Minimum Unconstrained Optimization
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
            ..threshold as a numeric value; threshold at which to stop the iterations
            ..**kwargs = initial_hessian : as matrix (default = identity)
            .. see unconstrained.params for further details on the methods that are being used
        '''
        alg = self._ls_algorithms[self.params['linesearch']['method']]
        ls_kwargs = self.params['linesearch']['params'][self.params['linesearch']['method']]
        d, g, Q = self._unc_algorithms[self.params['fminunc']['method']](fun,x0,_np.zeros(len(x0)),[],[],iters=0)
        if kwargs.has_key('max_iter'):
            max_iter= kwargs['max_iter']
        else:
            max_iter = self.params['fminunc']['params'][self.params['fminunc']['method']]['max_iter']
        if vectorized:
            x_vec = [x0]
        else:
            pass
        x = x0
        iters = 0
        lsiters = 0
        while _np.dot(g,g) > threshold and iters < max_iter:
            ls = alg(fun,x,d,**ls_kwargs)
            alpha = ls['alpha']
            lsiters += ls['iterations']
            #Q = _hessian(fun,x0,**params['hessian'])
            #alpha = g.T.dot(g)/(g.T.dot(Q).dot(g))
            x = _xstep(x,d,alpha)
            if vectorized:
                x_vec += [x]
            else:
                pass
            d, g, _ = self._unc_algorithms[self.params['fminunc']['method']](fun,x,d,g,Q,iters=iters,alpha=alpha)
            iters += 1
        if vectorized:
            return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        else:
            return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}

    
        
