# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 11:20:34 2017

@author: GABRIS46
"""
from ..finitediff import jacobian as _jacobian, hessian as _hessian
from ..linesearch import xstep as _xstep, backtracking as _backtracking, interp23 as _interp23, unimodality as _unimodality, golden_section as _golden_section
from .. import np as _np
from .. import scipy as _sp
from ..nonlinear import unconstrained

class constrained(object):
    
    def __init__(self,parameters,unconstrained):
        self.eps = _np.finfo(_np.float64).eps
        self.resolution = _np.finfo(_np.float64).resolution
        self.params = parameters
        self.__unconstrained = unconstrained
        self._ls_algorithms = {'backtracking':_backtracking,
                    'interp23':_interp23,
                    'unimodality':_unimodality,
                    'golden-section':_golden_section}
    
        self._con_algorithms = {'projected-gradient':self._proj_gradient
                        }

    def _proj_gradient(self,fun,x0,d0,g0,Q0,A_,I,J,K,*args,**kwargs):
        '''
            projected gradient (linear convergence)
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
        '''
        #print('### proj grad')
        #print('x0', x0)
        g = _jacobian(fun,x0,**self.params['jacobian'])
        if len(I|K) == 0:
            return -g, g, []
        else:
            pass
        Ak = A_[[i for i in I|K]]
        #print('Ak') 
        #print(Ak)
        P = _np.identity(len(x0),_np.float64)-Ak.T.dot(_np.linalg.inv(Ak.dot(Ak.T))).dot(Ak)
        d = -P.dot(g)
        #print('d', d)
        #print(' g', g)
        if d.dot(d) < self.params['fmincon']['params']['projected-gradient']['threshold'] or kwargs['alpha'] < self.resolution:
            if len(I) >0:
                #AkI = Ak#Ak[range(len(I)),:]
                ##print AkI
                ##print g
                lambdas = -_np.linalg.inv((Ak.dot(Ak.T))).dot(Ak).dot(g)
                #print 'lambdas', lambdas
                lambdas = lambdas[range(len(I))]
                if _np.min(lambdas) < 0.0: # check whether lagrange multipliers are less than 0 for any inequality constraint. If so, remove the constraint from the active set.
                    #pos, = _np.where(lambdas<0.0)
                    pos, = _np.where(lambdas==_np.min(lambdas))
                    js = [list(I)[i] for i in pos]
                    #print 0, I, J, K, pos
                    J |= set(js)
                    I ^= set(js)
                    #print 1, I, J, K
                    #print '### proj grad end in'
                    return self._proj_gradient(fun,x0,d0,g0,Q0,A_,I,J,K,alpha=_np.inf)
                else:
                    pass
            else:
                pass
        #print '### proj grad end out'
        return d, g, []
    
    def fmincon(self,fun,x0,A=[],b=[],Aeq=[],beq=[],threshold=1e-6,vectorized=False,**kwargs):
        '''
            Minimum Constrained Optimization
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
            ..threshold as a numeric value; threshold at which to stop the iterations
            .. A as a matrix of size n×m holding the coefficient of inequality constraints A×x <= b
            .. b as an array of size n of the RHS of the inequality constraints
            .. Aeq as a matrix of size n×m holding the coefficient of equality constraints Aeq×x = beq
            .. beq as an array of size n of the RHS of the equality constraints
            ..**kwargs = initial_hessian : as matrix (default = identity)
            .. see unconstrained.params for further details on the methods that are being used
        '''
        A = _np.array(A,_np.float64)
        Aeq = _np.array(Aeq,_np.float64)
        b = _np.array(b,_np.float64)
        beq = _np.array(beq,_np.float64)
        if all([i==None for i in (A,Aeq,b,beq)]):
            raise Exception('For unconstrained problem, use optinpy.unconstrained.*')
        else:
            pass
        alg = self._ls_algorithms[self.params['linesearch']['method']]
        ls_kwargs = self.params['linesearch']['params'][self.params['linesearch']['method']]
        self.params['fmincon']['params'][self.params['fmincon']['method']]['threshold'] = threshold
        ### FIND FEASIBLE INITIAL POINT
        if len(Aeq) == 0:
            A_ = A
            A_lp = _np.concatenate((A,_np.identity(_np.shape(A)[0])),axis=1)
            b_lp = b
            I0 = set(range(len(A)))
            _ = _np.shape(A)[0]
            K = set([])
        elif len(A) == 0:
            A_ = Aeq
            A_lp = _np.concatenate((Aeq,_np.zeros(_np.shape(Aeq)[0])),axis=1)
            b_lp = beq
            I0 = set([])
            _ = _np.shape(Aeq)[0]
            K = set(range(_))
        else:
            A_ = _np.concatenate((A,Aeq),axis=0)
            A_lp = _np.concatenate((_np.concatenate((A,_np.identity(_np.shape(A)[0])),axis=1),\
                                     _np.concatenate((Aeq,_np.zeros([_np.shape(Aeq)[0],_np.shape(A)[0]])),axis=1)),axis=0)
            b_lp = _np.concatenate((b,beq))
            I0 = set(range(len(A)))
            _ = _np.shape(A)[0]
            K = set(range(_np.shape(A)[0],_np.shape(A)[0]+_np.shape(Aeq)[0]))
        g = _jacobian(fun,x0,**self.params['jacobian'])
        res = _sp.optimize.linprog(_np.concatenate((-g,_np.zeros(_))),A_ub=None,b_ub=None,A_eq=A_lp,b_eq=b_lp,options={'disp':True})
        if not res['success']:
            raise Exception('A feasible initial point could not be found.')
        else:
            pass
        I = set(_np.where([_np.abs(i)<self.eps for i in res['x'][len(x0):len(x0)+_np.shape(A)[0]]])[0].tolist())
        J = I0^I
        #print A_
        def amax(alpha,x0,d,A_,b,I,J,K):
            if len(J) == 0 and len(K) == 0 :
                return alpha
            else:
                alpha_ = (_np.array(b[list(J)])-_np.array(A_[list(J)]).dot(x0))*(_np.array(A_[list(J)]).dot(d))**-1.0
                #print '### amax'
                #print 'x :', x0
                #print 'alphas Max_in:', alpha_
                alpha_[_np.where(alpha_<= self.eps)[0]] = _np.inf
                if _np.min(alpha_)-alpha < self.eps:
                    pos, = _np.where(_np.abs(alpha_-_np.min(alpha_))<self.eps)
                    js = [list(J)[i] for i in pos]
                    #print 2, I, J, K
                    #print js
                    J ^= set(js)
                    I |= set(js)
                    #print 3, I, J, K
                    #print '### amax end clipped'
                    return _np.min(alpha_)            
                else:
                    #print '### amax end unclipped'
                    return alpha
        if res['success']:
            x0 = _np.array(res['x'][0:len(x0)],_np.float64)
            #print x0
            #print I, J
            d, g, Q = self._con_algorithms[self.params['fmincon']['method']](fun,x0,[],[],[],A_,I,J,K,iters=0,alpha=_np.inf)
            if kwargs.has_key('max_iter'):
                max_iter = kwargs['max_iter']
            else:
                max_iter = self.params['fmincon']['params'][self.params['fmincon']['method']]['max_iter']
            if vectorized:
                x_vec = [x0]
            else:
                pass
            x = x0
            #print I, J
            #print '\n'
            iters = 0
            lsiters = 0
            alpha = _np.inf
            while _np.dot(d,d) > threshold and iters < max_iter:
                ls = alg(fun,x,d,**ls_kwargs)
                #print '\n####ITERATION {}\n'.format(iters)
                #print 'd: ',d
                #print 'x0: ',x
                #print 'Ax0:',_np.dot(A_,x)
                #print 'Sets: ', I, J, K
                alpha = ls['alpha']
                #print 'alpha Inicial', alpha
                alpha = amax(alpha,x,d,A,b,I,J,K)
                #print 'alpha Final', alpha, I, J
                lsiters += ls['iterations']
                #Q = _hessian(fun,x0,**params['hessian'])
                #alpha = g.T.dot(g)/(g.T.dot(Q).dot(g))
                x = _xstep(x,d,alpha)            
                #print 'x1: ',x
                #print 'Ax1:',_np.dot(A_,x), 'IJK 1', I, J, K
                if vectorized:
                    x_vec += [x]
                else:
                    pass
                d, g, _ =self. _con_algorithms[self.params['fmincon']['method']](fun,x,d,g,Q,A_,I,J,K,iters=iters,alpha=alpha)
                if len(I|K) > 0:
                    _A = _np.array([A_[i] for i in I|K],_np.float64)
                    #print 'lambdas',-_np.linalg.inv((_A.dot(_A.T))).dot(_A).dot(g)
                iters += 1
                #print 'Sets finais: ', I, J, K
                #print '\n'
                if alpha == 0:
                    break
            if vectorized:
                return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
            else:
                return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        else:
            raise Exception('Could not determine an initial feasible point.')
            
    def fminnlcon(self,fun,x0,g,c,beta,threshold=1e-6,vectorized=False,**kwargs):
        '''
            Minimum (Non Linearly) Constrained Optimization
              The weighted function is minimized using the defined algorithm for unconstrained optimization in .unconstrained
            ..fun as callable object; must be a function of x0 and return a single number
            ..x0 as a numeric array; point from which to start
            ..g as an array of callable inequality constraints functions
            ..c as numeric, initial constraint weight
            ..beta as numeric (>1) as the factor by which c growths after each iteration
            ..threshold as a numeric value; threshold at which to stop the iterations
            ..**kwargs = initial_hessian : as matrix (default = identity)
            .. see unconstrained.params for further details on the methods that are being used
        '''
        if self.params['fminnlcon']['method'] == 'penalty':
            P = lambda x : _np.sum([_np.max([0.,_(x)]) for _  in g])
            f = lambda x : fun(x)+c*P(x)
            chck = lambda x : c*P(x)
        elif self.params['fminnlcon']['method'] == 'log-barrier':
            B = lambda x : -_np.sum([_np.log(-_(x)) for _ in g]) # g(x) <= 0
            f = lambda x : fun(x)+(1./c)*B(x)
            chck = lambda x : (1./c)*B(x)
        elif self.params['fminnlcon']['method'] == 'barrier':
            B = lambda x : -_np.sum([1./_(x) for _ in g]) # g(x) <= 0
            f = lambda x : fun(x)+(1./c)*B(x)
            chck = lambda x : (1./c)*B(x)   
        else:
            raise Exception('The fminnlcon method ({}) has not been identified.'.format(self.params['fminnlcon']['method']))
        x = x0
        if vectorized:
            x_vec = [x0]
            c_vec = [c]
            err_vec =[chck(x)]
        else:
            pass
        iters = 0
        inner_iters = 0
        lsiters = 0
        print chck(x)
        #print P(x)
        while chck(x) > threshold:
            print 'f(x) = ', f(x)#,'P(x) = ', P(x)
            sol = self.__unconstrained.fminunc(f,x,threshold)
            print sol
            x = sol['x']
            print 'x',x
            print 'f(x)',f(x)
            #print 'P(x)',P(x)
            print 'chck',chck(x)
            c *= beta
            if vectorized:
                x_vec += [x]
                c_vec += [c]
                err_vec += [chck(x)]
            else:
                pass
            iters += 1
            inner_iters += sol['iterations']
            lsiters += sol['ls_iterations']
        if vectorized:
            print x_vec
            return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'c': c_vec, 'err' : err_vec, 'iterations':iters,'inner_iterations':inner_iters,'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        else:
            return {'x':x, 'f':fun(x),'c':c,'iterations':iters,'inner_iterations':inner_iters,'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        
        
        
        
        
