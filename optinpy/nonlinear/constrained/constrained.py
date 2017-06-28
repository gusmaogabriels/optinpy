# -*- coding: utf-8 -*-

from ...finitediff import jacobian as __jacobian, hessian as __hessian
from ...linesearch import xstep as __xstep, backtracking as __backtracking, interp23 as __interp23, unimodality as __unimodality, golden_section as __golden_section
from ... import np as __np
from ... import scipy as __sp

eps = __np.finfo(__np.float64).eps
resolution = __np.finfo(__np.float64).resolution

params = {'fmincon':{'method':'newton','params':\
                {'projected-gradient':{'max_iter':1e3,'threshold':1e-6},\
                 }},\
            'jacobian':{'algorithm':'central','epsilon':__np.sqrt(resolution)},\
            'hessian':{'algorithm':'central','epsilon':__np.sqrt(resolution),'initial':None},\
            'linesearch':{'method':'backtracking','params':
                {'backtracking':{'alpha':1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'interp23':{'alpha':1,'alpha_min':0.1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'unimodality':{'b':1,'threshold':1e-4,'max_iter':1e3},
                'golden-section':{'b':1,'threshold':1e-4,'max_iter':1e3}}}}  

def proj_gradient(fun,x0,d0,g0,Q0,A_,I,J,K,*args,**kwargs):
    '''
        projected gradient (linear convergence)
        ..fun as callable object; must be a function of x0 and return a single number
        ..x0 as a numeric array; point from which to start
    '''
    print '### proj grad'
    print 'x0', x0
    g = __jacobian(fun,x0,**params['jacobian'])
    if len(I|K) == 0:
        return -g, g, []
    else:
        pass
    Ak = A_[[i for i in I|K]]
    print 'Ak' 
    print Ak
    P = __np.identity(len(x0),__np.float64)-Ak.T.dot(__np.linalg.inv(Ak.dot(Ak.T))).dot(Ak)
    d = -P.dot(g)
    print 'd', d
    print ' g', g
    if d.dot(d) < params['fmincon']['params']['projected-gradient']['threshold'] or kwargs['alpha'] < resolution:
        if len(I) >0:
            #AkI = Ak#Ak[range(len(I)),:]
            #print AkI
            print g
            lambdas = -__np.linalg.inv((Ak.dot(Ak.T))).dot(Ak).dot(g)
            print 'lambdas', lambdas
            lambdas = lambdas[range(len(I))]
            if __np.min(lambdas) < 0.0: # check whether lagrange multipliers are less than 0 for any inequality constraint. If so, remove the constraint from the active set.
                pos, = __np.where(lambdas<0.0)
                js = [list(I)[i] for i in pos]
                print 0, I, J, K, pos
                J |= set(js)
                I ^= set(js)
                print 1, I, J, K
                print '### proj grad end in'
                return proj_gradient(fun,x0,d0,g0,Q0,A_,I,J,K,alpha=__np.inf)
            else:
                pass
        else:
            pass
    print '### proj grad end out'
    return d, g, []


__ls_algorithms = {'backtracking':__backtracking,
                'interp23':__interp23,
                'unimodality':__unimodality,
                'golden-section':__golden_section}

__con_algorithms = {'projected-gradient':proj_gradient
                    }
                    

def fmincon(fun,x0,A=None,b=None,Aeq=None,beq=None,threshold=1e-6,vectorized=False,**kwargs):
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
    if all([i==None for i in (A,Aeq,b,beq)]):
        raise Exception('For unconstrained problem, use optinpy.unconstrained.*')
    else:
        pass
    alg = __ls_algorithms[params['linesearch']['method']]
    ls_kwargs = params['linesearch']['params'][params['linesearch']['method']]
    params['fmincon']['params'][params['fmincon']['method']]['threshold'] = threshold
    ### FIND FEASIBLE INITIAL POINT
    if Aeq == None:
        A_ = A
        A_lp = __np.concatenate((A,__np.identity(__np.shape(A)[0])),axis=1)
        b_lp = b
        I0 = set(range(len(A)))
        _ = __np.shape(A)[0]
        K = set([])
    elif A == None:
        A_ = Aeq
        A_lp = __np.concatenate((Aeq,__np.zeros(__np.shape(Aeq)[0])),axis=1)
        b_lp = beq
        I0 = set([])
        _ = __np.shape(Aeq)[0]
        K = set(range(_))
    else:
        A_ = __np.concatenate((A,Aeq),axis=0)
        A_lp = __np.concatenate((__np.concatenate((A,__np.identity(__np.shape(A)[0])),axis=1),\
                                 __np.concatenate((Aeq,__np.zeros([__np.shape(Aeq)[0],__np.shape(A)[0]])),axis=1)),axis=0)
        b_lp = __np.concatenate((b,beq))
        I0 = set(range(len(A)))
        _ = __np.shape(A)[0]
        K = set(range(__np.shape(A)[0],__np.shape(A)[0]+__np.shape(Aeq)[0]))
    g = __jacobian(fun,x0,**params['jacobian'])
    res = __sp.optimize.linprog(__np.concatenate((-g,__np.zeros(_))),A_ub=None,b_ub=None,A_eq=A_lp,b_eq=b_lp,options={'disp':True})
    if not res['success']:
        raise Exception('A feasible initial point could not be found.')
    else:
        pass
    I = set(__np.where([__np.abs(i)<eps for i in res['x'][len(x0):len(x0)+__np.shape(A)[0]]])[0].tolist())
    J = I0^I
    print A_
    def amax(alpha,x0,d,A_,b,I,J,K):
        if len(J) == 0 and len(K) == 0 :
            return alpha
        else:
            alpha_ = (__np.array(b[list(J)])-__np.array(A_[list(J)]).dot(x0))*(__np.array(A_[list(J)]).dot(d))**-1.0
            print '### amax'
            print 'x :', x0
            print 'alphas Max_in:', alpha_
            alpha_[__np.where(alpha_<=eps)[0]] = __np.inf
            if __np.min(alpha_)-alpha < eps:
                pos, = __np.where(__np.abs(alpha_-__np.min(alpha_))<eps)
                js = [list(J)[i] for i in pos]
                print 2, I, J, K
                #print js
                J ^= set(js)
                I |= set(js)
                print 3, I, J, K
                print '### amax end clipped'
                return __np.min(alpha_)            
            else:
                print '### amax end unclipped'
                return alpha
    if res['success']:
        x0 = __np.array(res['x'][0:len(x0)],__np.float64)
        print x0
        print I, J
        d, g, Q = __con_algorithms[params['fmincon']['method']](fun,x0,[],[],[],A_,I,J,K,iters=0,alpha=__np.inf)
        if kwargs.has_key('max_iter'):
            max_iter = kwargs['max_iter']
        else:
            max_iter = params['fmincon']['params'][params['fmincon']['method']]['max_iter']
        if vectorized:
            x_vec = [x0]
        else:
            pass
        x = x0
        print I, J
        print '\n'
        iters = 0
        lsiters = 0
        alpha = __np.inf
        while __np.dot(d,d) > threshold and iters < max_iter:
            ls = alg(fun,x,d,**ls_kwargs)
            print '\n####ITERATION {}\n'.format(iters)
            print 'd: ',d
            print 'x0: ',x
            print 'Ax0:',__np.dot(A_,x)
            print 'Sets: ', I, J, K
            alpha = ls['alpha']
            print 'alpha Inicial', alpha
            alpha = amax(alpha,x,d,A,b,I,J,K)
            print 'alpha Final', alpha, I, J
            lsiters += ls['iterations']
            #Q = __hessian(fun,x0,**params['hessian'])
            #alpha = g.T.dot(g)/(g.T.dot(Q).dot(g))
            x = __xstep(x,d,alpha)            
            print 'x1: ',x
            print 'Ax1:',__np.dot(A_,x), 'IJK 1', I, J, K
            if vectorized:
                x_vec += [x]
            else:
                pass
            d, g, _ = __con_algorithms[params['fmincon']['method']](fun,x,d,g,Q,A_,I,J,K,iters=iters,alpha=alpha)
            print 'lambdas',-__np.linalg.inv((A_.dot(A_.T))).dot(A_).dot(g)
            iters += 1
            print 'Sets finais: ', I, J, K
            print '\n'
            if alpha == 0:
                break
        if vectorized:
            return {'x':x_vec, 'f':[fun(x) for x in x_vec], 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
        else:
            return {'x':x, 'f':fun(x), 'iterations':iters, 'ls_iterations':lsiters}#, 'parameters' : params.copy()}
    else:
        raise Exception('Could not determine an initial feasible point.')