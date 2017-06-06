# distutils: language=c++
# cython: embedsignature=True

from cpython cimport Py_INCREF, PyTuple_New, PyTuple_SET_ITEM
from cython.parallel import prange
cimport cython

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float32_t, int64_t

cdef extern from "csrc/array.cc":
    void ftake(float32_t* a, int64_t* indices, float32_t* out, int64_t minibatchsize, int64_t M) nogil
               
    void itake(int64_t* a, int64_t* indices, int64_t* out, int64_t minibatchsize, int64_t M) nogil
    
    void fcpy(float32_t* a, float32_t* out, int64_t M) nogil
    
    void icpy(int64_t* a, int64_t* out, int64_t M) nogil
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ndarray take2d(ndarray a, int64_t[:,::1] minibatch_indices, ndarray out=None):
    """
    Does the same as np.concatenate([a[minibatch_indices[i]][None] for i in range(len(minibatch_indices))], axis=0) 
    (where a must be of dtype np.float32 and multidimensional) but multithreaded (if environment variable OMP_NUM_THREADS > 1) 
    and with no boundchecking.
    """
    # type-checking
    if not a.flags.c_contiguous or (a.dtype != np.float32 and a.dtype != np.int64) or a.ndim <= 1:
        msg = "a must be in c-order, of dtype float32 and multidimensional"
        raise TypeError(msg)
        
    cdef:
        tuple outshape = PyTuple_New( a.ndim + 1 )
        int64_t M = 1
        int64_t K = minibatch_indices.shape[0]
        int64_t minibatchsize = minibatch_indices.shape[1]
        int64_t k, i
    # buid outputshape tuple
    PyTuple_SET_ITEM(outshape, 0, K)
    Py_INCREF(K)
    PyTuple_SET_ITEM(outshape, 1, minibatchsize )
    Py_INCREF(minibatchsize)
    for i in range(1,a.ndim):
        PyTuple_SET_ITEM(outshape, i+1, a.shape[i] )
        Py_INCREF(a.shape[i])
        M *= a.shape[i]
    # initialize out if not provided
    if out is None:
        out = np.empty(outshape, dtype=np.float32)
    else:
        # type-checking
        if not out.flags['C'] or (a.dtype != np.float32 and a.dtype != np.int64) or out.ndim != len(outshape):
            msg = "output buffer must be in c-order, of dtype float32 and with {} dimension".format(len(outshape))
            raise TypeError(msg)
        for i in range(len(outshape)):
            if out.shape[i] != outshape[i]:
                msg = "output buffer shape is not correct. should be been {0}".format(outshape)
                raise ValueError(msg)
        
    cdef:
        float32_t* fa_
        float32_t* fout_
        int64_t* ia_
        int64_t* iout_
        int64_t* indices_ = <int64_t*>&minibatch_indices[0,0]
    
    if out.dtype == np.float32:
        fa_ = <float32_t*>a.data
        fout_ = <float32_t*>out.data
        for k in prange(K, nogil=True, schedule='static'):
            ftake(fa_, &indices_[k*minibatchsize], &fout_[k*minibatchsize*M], minibatchsize, M)
    else:
        ia_ = <int64_t*>a.data
        iout_ = <int64_t*>out.data
        for k in prange(K, nogil=True, schedule='static'):
            itake(ia_, &indices_[k*minibatchsize], &iout_[k*minibatchsize*M], minibatchsize, M)
            
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ndarray take1d(ndarray a, int64_t[:] minibatch_indices, ndarray out=None):
    """
    Does the same as a[minibatch_indices] (where a must be of dtype np.float32 or int64 and multidimensional) 
    but multithreaded (if environment variable OMP_NUM_THREADS > 1) and with no boundchecking.
    """
    # type-checking
    if not a.flags.c_contiguous or (a.dtype != np.float32 and a.dtype != np.int64) or a.ndim <= 1:
        msg = "a must be in c-order, of dtype float32 and multidimensional"
        raise TypeError(msg)
    cdef:
        tuple outshape = PyTuple_New(a.ndim)
        int64_t M = 1
        int64_t minibatchsize = minibatch_indices.shape[0]
        int64_t i, n
    # buid outputshape tuple
    PyTuple_SET_ITEM(outshape, 0, minibatchsize )
    Py_INCREF(minibatchsize)
    for i in range(1,a.ndim):
        PyTuple_SET_ITEM(outshape, i, a.shape[i] )
        Py_INCREF(a.shape[i])
        M *= a.shape[i]
    # initialize out if not provided
    if out is None:
        out = np.empty(outshape, dtype=np.float32)
    else:
        # type-checking
        if not out.flags.c_contiguous or (a.dtype != np.float32 and a.dtype != np.int64) or out.ndim != len(outshape):
            msg = "output buffer must be in c-order, of dtype float32 or int64 and with {} dimension".format(len(outshape))
            raise ValueError(msg)
        for i in range(len(outshape)):
            if out.shape[i] != outshape[i]:
                msg = "output buffer shape is not correct. should be been {0}".format(outshape)
                raise ValueError(msg)
    cdef:
        float32_t* fa_
        float32_t* fout_
        int64_t* ia_
        int64_t* iout_
        int64_t* indices_ = <int64_t*>&minibatch_indices[0]
    
    if out.dtype == np.float32:
        fa_ = <float32_t*>a.data
        fout_ = <float32_t*>out.data
        for n in prange(minibatchsize, nogil=True, schedule='static'):
            fcpy(&fa_[indices_[n]*M], &fout_[n*M], M)
    else:
        ia_ = <int64_t*>a.data
        iout_ = <int64_t*>out.data
        for n in prange(minibatchsize, nogil=True, schedule='static'):
            icpy(&ia_[indices_[n]*M], &iout_[n*M], M)
            
    return out