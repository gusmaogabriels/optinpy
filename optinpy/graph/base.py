# -*- coding: utf-8 -*-

class graph(object):
    
    def __init__(self):
        self.nodes = {}
        self.arcs = []
            
    def add_connection(self, nfrom, nto):
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
        
class node(object):
    
    def __init__(self, key, **kwargs):
        """
        .. key as integer
        .. parent as integer
        .. child as integer
        """
        _argparser(key,int)      
        self.key = key
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
    

def _argparser(x,vartype,**kwargs):
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