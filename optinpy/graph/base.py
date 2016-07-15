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
        self.nodes_map = {0:0}
        self.nodes = {0:node(0,0)}   
        self.arcs = {}
        self.root_weight = float('inf')
        self.basic = []
        self.non_basic = []
        self.__root_arcs = []
        
    def set_root_weight(self,weight):
        self.root_weight = weight        
        for a in self.__root_arcs:
            a.cost = self.root_weight
        self.nodes[0].update()
        self.__update_crawler(self.nodes[0])
    
    def __update_crawler(self,node):
        for n in node.children:
            if n == n.__node_to_root__ or n == self.nodes[0]:
                pass
            else:
                n.update()
                self.__update_crawler(n)
        for n in node.parents:
            if n == n.__node_to_root__ or n == self.nodes[0]:
                pass
            else:
                n.update()
                self.__update_crawler(n)
            
    def add_node(self,key,b):
        """
            ..graph object instantiatior
            ..key as a list of numeric or string for node label
            ..b as a list/tuple of numeric b values for the nodes
        """
        key, b = [x if isinstance(x,(list,tuple)) else [x] for x in [key,b]]
        argparser(b,(list,tuple),varsize=len(key),subvartype=(int,long,float))
        for i in range(0,len(key)):
            self.nodes.__setitem__(key[i],node(key[i],b[i]))
            self.nodes[0].b -= b[i]
            if b[i]>=0 and key[i] not in self.nodes_map.keys():
                self.add_connection(key[i],0,self.root_weight)
                self.arcs[key[i]][0].activate()
                self.nodes[key[i]].__arc_to_root__ = [self.arcs[key[i]][0],1.0]
                self.nodes[key[i]].__node_to_root__ = self.nodes[0]
                self.arcs[key[i]][0].flow = b[i]
                self.__root_arcs += [self.arcs[key[i]][0]]
            elif b[i]<0 and key[i] not in self.nodes_map.keys():
                self.add_connection(0,key[i],self.root_weight)
                self.arcs[0][key[i]].activate()
                self.nodes[key[i]].__arc_to_root__ = [self.arcs[0][key[i]],-1.0]
                self.nodes[key[i]].__node_to_root__ = self.nodes[0]
                self.arcs[0][key[i]].flow = -b[i]
                self.__root_arcs += [self.arcs[0][key[i]]]
            else:
                pass
            self.nodes_map.__setitem__(key[i],len(self.nodes_map))
                               
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
                self.arcs[source[i]].__setitem__(destination[i],arc(self.nodes[source[i]],self.nodes[destination[i]],cost[i],self))
            else: 
                self.arcs.__setitem__(source[i],{destination[i]:arc(self.nodes[source[i]],self.nodes[destination[i]],cost[i],self)})
            
class arc(object):
    
    def __init__(self,source,destination,cost=0.0,graph=None):
        """
        .. this is the arc class and ties two node objects
        .. source and destination as node objects        
        .. cost as numeric
        """
        argparser(cost,(int,long,float))
        self.source = source
        self.destination = destination
        self.source.__children__ += [destination]
        self.destination.__parents__ += [source]
        self.cost = cost
        self.c_hat = lambda : (self.cost-(self.source.__pi__[0]-self.destination.__pi__[0]))
        self.flow = 0
        self.graph = graph
        if self.graph != None:
            self.graph.non_basic += [self]
        else:
            pass
        self.__state__ = [False]
        
    def __call__(self):
        return('ARC f:{} t:{}'.format(self.source.key,self.destination.key))
               
    def pivot(self):
        """
            activate this arc and then deactivate the barrier arc whithin the created loop.
        """
        source = self.source
        destination = self.destination
        nsource = [self.source.key]
        ndestination = [self.destination.key]
        asource = []
        adestination = []
        out_arc = []
        while not set(nsource) & set(ndestination):
            if source.__node_to_root__ != None:             
                nsource += [source.__node_to_root__.key]
                asource += [[source.__arc_to_root__[0],[1 if source==source.__arc_to_root__[0].destination else -1][0]]]
                if source.__arc_to_root__[1]>0:
                    out_arc += [[source.__arc_to_root__[0].flow,source.__arc_to_root__[0],-1]]                  
                else:
                    pass # we've reached the root from the source side
                source = source.__node_to_root__
            else:
                pass
            if destination.__node_to_root__ != None:           
                ndestination += [destination.__node_to_root__.key]
                adestination += [[destination.__arc_to_root__[0],[1 if destination==destination.__arc_to_root__[0].source else -1][0]]]
                if destination.__arc_to_root__[1]<0:
                    out_arc += [[destination.__arc_to_root__[0].flow,destination.__arc_to_root__[0],1]]
                else:
                    pass # we've reached the root from the destination side
                destination = destination.__node_to_root__
            else:
                pass
        lcn = list(set(nsource) & set(ndestination))[0]
        asource = [asource[0:nsource.index(lcn)] if len(asource)>0 else []][0]
        adestination = [adestination[0:ndestination.index(lcn)] if len(adestination)>0 else[]][0]
        nsource = nsource[0:nsource.index(lcn)]
        ndestination = ndestination[0:ndestination.index(lcn)]
        arcs = [i[0] for i in asource]+[i[0] for i in adestination]
        out_arc.sort()
        while out_arc[0][1] not in arcs:
            out_arc.pop(0)
        out_arc = out_arc[0]
        print('\n')
        print('-----')
        print('IN',self())
        print('OUT',out_arc[1]())
        print(out_arc[1]())
        out_arc[1].deactivate()
        self.activate()
        for arc in adestination+asource:
            arc[0].flow += out_arc[0]*arc[1]
        if out_arc[2] > 0:
            arcs = [[self,1]]+adestination
            nodes = ndestination
        else:
            arcs = [[self,1]]+asource
            nodes = nsource
        self.flow = out_arc[0]
        d = 0
        i = 0
        while i<len(nodes):
            if arcs[i][0] == out_arc[1]:
                d = 1
            else:
                pass
            print('Update: node {}'.format(nodes[i]))
            print('Update: arc {}'.format(arcs[i][0]()))
            self.graph.nodes[nodes[i]].update(arcs[i+d][0])
            i+= 1
            
    def activate(self):
        self.source.set_to(self.destination,self)
        self.destination.set_from(self.source,self)
        if self.graph != None and self.__state__[0] == False:
            self.graph.non_basic.remove(self)
            self.graph.basic += [self]
        else:
            pass 
        self.__state__[0] = True
        
    def deactivate(self):
        self.source.remove_to(self.destination,self)            
        self.destination.remove_from(self.source,self)                  
        if self.graph != None and self.__state__[0] == True:
            self.graph.basic.remove(self)
            self.graph.non_basic += [self]
        else:
            pass
        self.__state__[0] = False
        
class node(object):
    
    def __init__(self, key, b):
        """
        .. key as any label (numeric or string)
        .. b as numeric
        """     
        self.key = key
        self.b = b
        self.__children__ = []
        self.__parents__ = []
        self.children = []
        self.parents = []
        self.mapper = []
        self.arcpos = {}
        self.__dist_to_root__ = [0 if self.key==0 else float('inf')]
        self.__arc_to_root__ = [None]
        self.__node_to_root__ = None
        self.__pi__ = [0]

    def update(self,arc=None):
        if len(self.mapper) > 0 and self.key != 0:
            #print('node key:',self.key)
            self.__arc_to_root__ = [min(self.mapper)[1:3] if arc==None else [[arc,-1.0] if self==arc.destination else [arc,1.0]][0]][0]          
            #print('arc_to_root and multiplier',self.__arc_to_root__[0](),self.__arc_to_root__[1])
            self.__dist_to_root__ = [float('inf') if self.__arc_to_root__ == None else[min(self.mapper)[0][0] + 1]][0]      
            self.__node_to_root__ = [None if self.key==0 or self.__arc_to_root__ == None else [self.__arc_to_root__[0].source \
            if self.__arc_to_root__[1]<0 else self.__arc_to_root__[0].destination][0]][0]
            #print('node_to_root',self.__node_to_root__.key)
            self.__pi__ = [0 if self.key ==0 else self.__node_to_root__.__pi__[0]+self.__arc_to_root__[1]*self.__arc_to_root__[0].cost]
        else:
            self.__dist_to_root__ = [0 if self.key==0 else float('inf')]
            self.__arc_to_root__ = [None]
            self.__node_to_root__ = None
            self.__pi__ = [0]
        
    def set_b(self,value):
        """
        .. value as numeric
        """
        argparser(value,(int,long,float))
        self.b = value

    def set_to(self,destination,arc):
        self.children += [destination]
        self.mapper += [[[[0] if arc.destination.key == 0 or self.key ==0 else destination.__dist_to_root__][0],arc,1.0]]
        self.arcpos.__setitem__(id(arc),id(self.mapper[-1]))
    
    def set_from(self,source,arc):
        self.parents += [source]
        self.mapper += [[[[0] if arc.source.key == 0 or self.key ==0 else source.__dist_to_root__][0],arc,-1.0]]
        self.arcpos.__setitem__(id(arc),id(self.mapper[-1]))
    
    def remove_to(self,destination,arc):
        self.children.remove(destination)
        self.mapper.pop([id(i) for i in self.mapper].index(self.arcpos[id(arc)]))
        self.arcpos.__delitem__(id(arc))
        if id(self.__arc_to_root__) == id(arc):
            self.__arc_to_root__ = [None]
        else:
            pass
   
    def remove_from(self,source,arc):
        self.parents.remove(source)
        self.mapper.pop([id(i) for i in self.mapper].index(self.arcpos[id(arc)]))
        self.arcpos.__delitem__(id(arc))
        if id(self.__arc_to_root__) == id(arc):
            self.__arc_to_root__ = [None]
        else:
            pass
        
    def get_connections(self):
        return [[self.key,i.key] for i in self.children]+[[i.key,self.key] for i in self.parents]
        