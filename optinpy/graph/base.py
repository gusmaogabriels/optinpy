# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function
from . import _argparser
from .bfm import bfm
from .dijkstra import dijkstra

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
        .. parent as integer
        .. child as integer
        """
        _argparser(key,int)      
        self.__key = key
        if kwargs.has_key('parent'):
            _argparser(kwargs['parent'],(int,tuple,list),subvartype=int)
            self.__parent = kwargs['parent']
        else:
            self.__parent = []
        if kwargs.has_key('child'):
            _argparser(kwargs['child'],(int,tuple,list),subvartype=int)
            self.__child = kwargs['child']
        else:
            self.__child = []
