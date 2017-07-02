# -*- coding: utf-8 -*-
from __future__ import division, absolute_import, print_function

from .unconstrained import unconstrained as __unconstrained
from .constrained import constrained as __constrained
from .. import np as __np

__eps = __np.finfo(__np.float64).eps
__resolution = __np.finfo(__np.float64).resolution

params = {'fminunc':{'method':'newton','params':\
                {'gradient':{'max_iter':1e3},\
                 'newton':{'max_iter':1e3},\
                 'modified-newton':{'sigma':1,'max_iter':1e3},\
                 'conjugate-gradient':{'max_iter':1e3},\
                 'fletcher-reeves':{'max_iter':1e3},\
                 'quasi-newton':{'max_iter':1e3,'hessian_update':'davidon-fletcher-powell'}
                 }},\
          'fmincon':{'method':'newton','params':\
                {'projected-gradient':{'max_iter':1e3,'threshold':1e-6},\
                 }},\
          'fminnlcon':{'method':'penalty','params':\
                {'log-barrier':{'max_iter':1e3,'threshold':1e-6},\
                 'barrier':{'max_iter':1e3,'threshold':1e-6},\
                 'penalty':{'max_iter':1e3,'threshold':1e-6},\
                 }},  
          'jacobian':{'algorithm':'central','epsilon':__np.sqrt(__eps)},\
          'hessian':{'algorithm':'central','epsilon':__np.sqrt(__eps),'initial':None},\
          'linesearch':{'method':'backtracking','params':
                {'backtracking':{'alpha':1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'interp23':{'alpha':1,'alpha_min':0.1,'rho':0.6,'c':1e-3,'max_iter':1e3},\
                'unimodality':{'b':1,'threshold':1e-4,'max_iter':1e3},
                'golden-section':{'b':1,'threshold':1e-4,'max_iter':1e3}}}} 


unconstrained = __unconstrained(params)
constrained = __constrained(params,unconstrained)
