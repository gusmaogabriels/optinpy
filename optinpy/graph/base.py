# -*- coding: utf-8 -*-

class graph(object):
    
    def __init__(self):
        self.network = {}
        
    def create(self,nodedic):
        return []
    
    def add_node(self,node):
        self.network.__setitem__(node.name,node.connection)
        
class node(object):
    
    def __init__(self, name, parent, child):
        
        self.name = name
        self.parent = parent
        self.child = child