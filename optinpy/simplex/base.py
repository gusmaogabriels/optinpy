# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

from . import np

def argparser(x,vartype,**kwargs):
    if isinstance(x,vartype):
        if kwargs.has_key('varsize'):
            if len(x) == kwargs['varsize']:
                pass
            else:
                raise IndexError('Size mismatch.')
        else:
            pass
        if kwargs.has_key('subvartype') and not isinstance(x,kwargs['subvartype']):
            if all([isinstance(i,kwargs['subvartype']) for i in x]):
                pass
            else:
                raise TypeError('subvariables types mismatch.')
        else:
            pass
        return x
    else:
        raise TypeError('variable type mismatch.')
        
class simplex(object):
        
    def __init__(self,A,b,c,mode='min',**kwargs):
        """
            This is a standad class for tableau (minimization as standard)
            A is matrix of size n×m
            c is a vector of size m
            b is a vecotr of size n
            mode as string 'min' or 'max'
            
            simplex minimizes or maximizes c'x s.t. Ax<=b
            therefore, A and b must be set in such fashion.
        """
        self.A = argparser(A,(np.ndarray,list,tuple))
        self.n = len(A)
        ms = [len(A[i]) for i in range(0,len(A))]
        if max(sorted(ms)) == min(sorted(ms)):
            self.m = len(A[0])
        else:
            raise Exception('A must be a matrix. i.e. all rows must have the same length.')
        self.c = argparser(c,(np.ndarray,list,tuple),varsize=self.m)
        self.b = argparser(b,(np.ndarray,list,tuple),varsize=self.n)
        
        # MODE MIN MAX
        if mode == 'min':
            self.z = 1
        elif mode == 'max':
            self.z = -1
            self.c *= self.z
        else:
            raise Exception("mode ({}) must be either 'max' or 'min'".format(mode))
        
        # In case LB and/or UB are definded
        if kwargs.has_key('ub') and argparser(kwargs['ub'],(list,tuple),varsize=self.m,subvartype=(int,long,float)):
            self.ub = kwargs['ub']+[float('inf') for i in range(0,self.n)]
        else:
            self.ub = [float('inf') for i in range(0,self.m+self.n)]
        if kwargs.has_key('lb') and argparser(kwargs['lb'],(list,tuple),varsize=self.m,subvartype=(int,long,float)):
            self.lb = kwargs['lb']+[0.0 for i in range(0,self.n)]
        else:
            self.lb = [0.0 for i in range(0,self.m+self.n)]
        self.delta = [self.ub[l]-self.lb[l] for l in range(0,self.m+self.n)]
        self.b = [self.b[l] - sum([self.A[l][k]*self.lb[k] for k in range(0,self.m)]) for l in range(0,self.n)]
        
        # Def basic and non basic
        self.non_basic = range(0,self.m)
        self.basic = range(self.m,self.n+self.m)               
        
        # Standardize
        self.delta_x = [0 for i in range(0,self.m)]+self.b
        self.pivot_matrix = [[1.0 if i == j else 0.0 for i in range(0,self.n+1)] for j in range (0,self.n+1)]
        self.A = [self.A[i]+self.pivot_matrix[i][0:self.n] for i in range(0,self.n)]
        self.c += [0 for i in range(0,self.n)]

    def pivot(self,i,j):
        A = self.A
        pivot_value = float(self.A[i][j])
        pivot_column = [float(v) for v in [self.A[l][j] for l in range(0,self.n)]]
        pivot_cost = self.c[j]
        self.c = [self.c[k]-(pivot_cost/pivot_value)*self.A[i][k] for k in range(0,self.m+self.n)]
        for l in list(set(range(1,self.n+1))^set([i+1])):
            self.b[l-1] = self.b[l-1]-(pivot_column[l-1]/pivot_value)*self.b[i]
        self.b[i] /= pivot_value
        for k in range(0,self.m+self.n):
            self.A[i][k] /= pivot_value
        for k in range(0,self.m+self.n):
            for l in list(set(range(0,self.n))^set([i])):
                self.A[l][k] -= float(self.A[i][k])*pivot_column[l]
        self.pivot_matrix = [[1.0]+[self.c[l] for l in self.non_basic]]+[[0.0]+[a[l] for l in self.non_basic] for a in A]
        self.non_basic[self.non_basic.index(j)] = self.basic[i]
        self.basic[i] = j
        return self.pivot_matrix
        
    def primal(self):
        if any([self.c[k] > 0 and self.delta_x[k] == self.delta[k] or self.c[k] < 0 and self.delta_x[k] == 0 for k in self.non_basic]):
            c_s = filter(lambda x : x != None, [c_ if c_[0]/c_[1]<0 else None for c_ in \
            [[self.c[k],[1.0 if self.delta_x[k] == 0 else -1.0][0],k] for k in self.non_basic]])
            # steepest descent
            if min(c_s)[0] < 0:
                c = min(c_s)
                j = c[2] # argmin(c)               
            else:
                c = max(c_s)
                j = c[2]
            epsilon_max = self.delta[j]*c[1]
            epsilon = [epsilon_max,j]
            for l in range(0,self.n):
                if self.A[l][j]*c[1]<0:
                    if abs((1.0/self.A[l][j])*(self.delta_x[self.basic[l]]-self.delta[l])) < abs(epsilon[0]):
                        epsilon = [(1.0/self.A[l][j])*(self.delta_x[self.basic[l]]-self.delta[l]),l]
                    else:
                        pass
                else:
                    if abs((1.0/self.A[l][j])*(self.delta_x[self.basic[l]])) < abs(epsilon[0]):
                        epsilon = [(1.0/self.A[l][j])*(self.delta_x[self.basic[l]]),l]
                    else:
                        pass
            for l in range(0,self.n):
                self.delta_x[self.basic[l]] -= self.A[l][j]*epsilon[0]
            self.delta_x[j] += epsilon[0]
            if epsilon[0] != epsilon_max:
                i = epsilon[1]               
                self.pivot(i,j)
            else:
                pass
        else:
            pass
        
    def dual(self):
        if any([b_<0 for b_ in self.b]):            
            b_s = filter(lambda x : x != None, [b_ if b_[0]<0 else None for b_ in zip(self.b,range(0,self.n))])
            while len(b_s)>0:
                i = min(b_s)[1]
                ratio = [-self.c[k]/self.A[i][k] if self.c[k]>0 and self.A[i][k]<0 else float('inf') for k in range(0,self.n+self.m)]
                if min(ratio)<np.float('inf'):                
                    j = list.index(ratio,min(ratio)) # argmin(c)
                    self.delta_x[j] = self.b[i]/self.A[i][j]
                    self.delta_x[self.basic[i]] = 0.0
                    self.pivot(i,j)
                    break
                else:
                    b_s.pop(0)
        else:
            pass            
        
        
    
        