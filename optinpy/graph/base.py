# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

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
            if all([isinstance(i,kwargs['subvartype']) for i in kwargs['subvartype']]):
                pass
            else:
                raise TypeError('subvariables types mismatch.')
        else:
            pass
        return x
    else:
        raise TypeError('variable type mismatch.')

class graph(object):
    
    def __init__(self):
        self.nodes = {}
        self.arcs = []
        self.costs = []
            
    def add_connection(self, nfrom, nto, cost):
        if len(self.arcs)==0 or [nfrom, nto] not in self.arcs:
            if nfrom not in self.nodes.keys():
                self.nodes.__setitem__(nfrom, node(nfrom,child=nto))
            elif nto not in nfrom.__child:
                nfrom.__child += [nto]
            else:
                pass
            if nto not in self.nodes.keys():
                self.nodes.__setitem__(nto,node(nto,parent=nfrom))
            elif nfrom not in nto.__parent:
                nto.__parent += [nfrom]
            else:
                pass
            self.arcs += [[nfrom,nto]]
            self.costs += [cost]
        else:
            self.costs[self.arcs.index([nfrom,nto])] = cost
        
class node(object):
    
    def __init__(self, key, **kwargs):
        """
        .. key as integer
        .. ** kwargs        .. parent as integer
                            .. child as integer
                            .. b as numeric
        """
        argparser(key,int)      
        self.__key = key
        self.__b = 0 # 
        if kwargs.has_key('parent'):
            argparser(kwargs['parent'],(int,tuple,list),subvartype=int)
            self.__parent = kwargs['parent']
        else:
            self.__parent = []
        if kwargs.has_key('child'):
            argparser(kwargs['child'],(int,tuple,list),subvartype=int)
            self.__child = kwargs['child']
        else:
            self.__child = []
        if kwargs.has_key('b'):
            argparser(kwargs['b'],(int,long,float))
            self.__b = kwargs['b']
        else:
            self.__b = []
            
    def set_b(self,value):
        """
        .. value as numeric
        """
        argparser(value,(int,long,float))
        self.__b = value
        