# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:35:42 2016

@author: shiwu_001
"""

from config import DTYPE
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

"""
    Elementwise Kernel
"""
# r = a
seta = ElementwiseKernel("float a, float *r",
                         "r[i] = a",
                         "kernel_seta")

# r = ax + b
axpb = ElementwiseKernel("float a, float *x, float b, float *r",
                         "r[i] = a*x[i] + b",
                         "kernel_axpb")

# r = ax + by
axpby = ElementwiseKernel("float a, float *x, float b, float *y, float *r",
                          "r[i] = a*x[i] + b*y[i]",
                          "kernel_axpby")

# r = ax + by + cz 
axpbypcz = ElementwiseKernel(
    "float a, float *x, float b, float *y, float c, float *z, float *r",
    "r[i] = a*x[i] + b*y[i] + c*z[i]",
    "kernel_axpbypcz")

"""
    Reduction Kernel
"""
# (r) = sum(abs(x))
sumabs = ReductionKernel(DTYPE, neutral="0.",
                         reduce_expr="a+b",
                         map_expr="abs(x[i])",
                         arguments="float *x",
                         name="kernel_sumabs")

# (r) = sum(sqr(x))
sumsqr = ReductionKernel(DTYPE, neutral="0.",
                         reduce_expr="a+b",
                         map_expr="x[i]*x[i]",
                         arguments="float *x",
                         name="kernel_sumsqr")

#if __name__ == "__main__":
#    import pycuda.autoinit
#    import pycuda.gpuarray as garr
#    import numpy as np
#    a = np.arange(5, dtype=DTYPE)
#    print a
#    a_gpu = garr.to_gpu(a)
#    print "sumsqr", sumsqr(a_gpu)
#    a_gpu -= 4
#    a_gpu.get(a)
#    print a
#    print "sumabs", sumabs(a_gpu)