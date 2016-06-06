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
            if all([isinstance(i,kwargs['subvartype']) for i in x]):
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
        """
            ..graph object instantiatior
        """
        self.nodes_map = {'root':0}
        self.arcs_map = {}
        self.nodes = {0:node('root',0)}   
        self.arcs = {}
        
    def add_node(self,key,b):
        """
            ..graph object instantiatior
            ..key as a list of numeric or string for node label
            ..b as a list/tuple of numeric b values for the nodes
        """
        key, b = [x if isinstance(x,(list,tuple)) else [x] for x in [key,b]]
        argparser(b,(list,tuple),varsize=len(key),subvartype=(int,long,float))
        for i in range(0,len(key)):
            self.nodes_map.__setitem__(key[i],len(self.nodes_map))
            self.nodes.__setitem__(key[i],node(key[i],b[i]))
            self.nodes[0].b -= b[i]
            if b[i]>=0 and key[i] not in self.nodes_map.keys():
                self.add_connection(key[i],'root',float('inf'))
            elif b[i]<0 and key[i] not in self.nodes_map.keys():
                self.add_connection('root',key[i],float('inf'))
            else:
                pass
                               
    def add_connection(self, source, destination, cost, bidirectional = False):
        """ 
            ..add_connection
            ..source as list/tuple of label or int references to nodes
            ..destination as list/tuple of label or int references to nodes
            ..cost as list/tuple of numeric values of the size of source and destination
            ..bidirection as list/tuple of boolean values of the size of source and destination
        """
        source, destination, cost, bidirectional = [x if isinstance(x,(list,tuple)) else [x] for x in [source, destination, cost, bidirectional]]
        argparser(source,(list,tuple),subvartype=(int,long,float,basestring))
        argparser(destination,(list,tuple),varsize=len(source),subvartype=(int,long,float,basestring))
        argparser(cost,(list,tuple),varsize=len(source),subvartype=(int,long,float,basestring))
        argparser(bidirectional,(list,tuple),varsize=len(source),subvartype=(bool))
        nodes = list(set(source)^set(destination))+list(set(source)&set(destination))
        for n in filter(lambda x : x != False, [x if x not in self.nodes.keys() else False for x in nodes]):
            self.add_node(n,0)
        for i in range(0,len(source)):
            if self.arcs.has_key(source[i]):
                self.arcs[source[i]].__setitem__(destination[i],arc(self.nodes[source[i]],self.nodes[destination[i]],cost[i]))
            else: 
                self.arcs.__setitem__(source[i],{destination[i]:arc(self.nodes[source[i]],self.nodes[destination[i]],cost[i])})
            self.arcs[source[i]][destination[i]].activate()
            self.arcs_map.__setitem__(str(source[i])+'_'+str(destination[i]),self.arcs[source[i]][destination[i]])
            
class arc(object):
    
    def __init__(self,source,destination,cost=0.0):
        """
        .. this is the arc class and ties two node objects
        .. source and destination as node objects        
        .. cost as numeric
        """
        argparser(cost,(int,long,float))
        self.source = source
        self.destination = destination
        self.cost = cost
    
    def activate(self):
        self.source.set_to(self.destination)
        self.destination.set_from(self.source)
        
    def deactivate(self):
        self.source.remove_to(self.destination)            
        self.destination.remove_from(self.source)                  
        
class node(object):
    
    def __init__(self, key, b):
        """
        .. key as any label (numeric or string)
        .. b as numeric
        """     
        self.key = key
        self.b = b
        self.children = []
        self.parents = []
            
    def set_b(self,value):
        """
        .. value as numeric
        """
        argparser(value,(int,long,float))
        self.b = value
        
    def set_to(self,destination):
        self.children += [destination]
    
    def set_from(self,source):
        self.parents += [source]
    
    def remove_to(self,destination):
        self.children.remove(destination)
   
    def remove_from(self,source):
        self.parents.remove(source)
        
    def get_connections(self):
        return [[self.key,i.key] for i in self.children]+[[i.key,self.key] for i in self.parents]
        