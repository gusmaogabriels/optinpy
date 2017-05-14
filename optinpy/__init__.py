# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

__author__ = {'Gabriel S. Gusmao' : 'gusmaogabriels@gmail.com'}
__version__ = '1.0a'

"""
By Gabriel S. Gusmão (Gabriel Sabença Gusmão)
Oct, 2015
    optinpy version 1.0a
    ~~~~
    "General linear and nonlinear optimization methods"
    :copyright: (c) 2015 Gabriel S. Gusmão
    :license: GPU, see LICENSE for more details.
"""

import numpy as np
from .graph import *
from .simplex import *
from .mcfp import *
from .sp import *
from .mst import *
from .finite_diff import *
from .linesearch import *
from .nonlinear import *

__all__ = ['np','graph','__author__','__version__']
