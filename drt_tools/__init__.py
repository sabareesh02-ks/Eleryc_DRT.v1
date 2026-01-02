# -*- coding: utf-8 -*-
"""
DRT Tools Package for Web Integration
pyDRTtools integrated for web applications
"""

# Import main functions for easy access
from .runs import EIS_object, simple_run
from .basics import x_to_gamma, assemble_A_re, assemble_A_im, assemble_M_1, assemble_M_2
from .parameter_selection import GCV, mGCV, rGCV, LC, kf_CV, re_im_CV

__version__ = "1.0.0"
__authors__ = "Francesco Ciucci, Adeleke Maradesa, Baptiste Py, Ting Hei Wan"

print(f"DRT Tools package initialized for web integration (v{__version__})")

