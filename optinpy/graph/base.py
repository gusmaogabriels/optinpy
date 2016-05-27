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
                self.nodes.__setitem__(nfrom, node(nfrom,children=nto))
            elif nto not in self.nodes[nfrom].children:
                self.nodes[nfrom].children += [nto]
            else:
                pass
            if nto not in self.nodes.keys():
                self.nodes.__setitem__(nto,node(nto,parents=nfrom))
            elif nfrom not in self.nodes[nto].parents:
                self.nodes[nto].parents += [nfrom]
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
        .. ** kwargs        .. parents as integer
                            .. children as integer
                            .. b as numeric
        """
        argparser(key,int)      
        self.key = key
        self.b = 0 #
        self.children = []
        self.parents = []
        if kwargs.has_key('parents'):
            argparser(kwargs['parents'],int)
            self.parents = [kwargs['parents']]
        else:
            self.parents = []
        if kwargs.has_key('children'):
            argparser(kwargs['children'],int)
            self.children = [kwargs['children']]
        else:
            self.children = []
        if kwargs.has_key('b'):
            argparser(kwargs['b'],(int,long,float))
            self.b = kwargs['b']
        else:
            self.b = []
            
    def set_b(self,value):
        """
        .. value as numeric
        """
        argparser(value,(int,long,float))
        self.b = value
        