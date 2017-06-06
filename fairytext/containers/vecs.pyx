# distutils: language=c++
# cython: embedsignature=True

import torch
from torch import FloatTensor, LongTensor, IntTensor, ShortTensor

import numpy as np
cimport numpy as np
from numpy cimport ndarray, float32_t, int64_t, int32_t, int16_t

from cython.parallel import prange

cdef extern from "csrc/vecs.cc":
    void fill_float_vec(float32_t* in_vec, int64_t in_vec_len, 
                        float32_t* out_vec, int64_t out_vec_len, float32_t fill_value) nogil
    void fill_int_vec(int32_t* in_vec, int64_t in_vec_len, 
                        int32_t* out_vec, int64_t out_vec_len, int32_t fill_value) nogil
    void fill_short_vec(int16_t* in_vec, int64_t in_vec_len, 
                        int16_t* out_vec, int64_t out_vec_len, int16_t fill_value) nogil


cdef int64_t _get_max_len(int64_t* lengths_ptr, int64_t* indices_ptr, int64_t num_vecs, int64_t num_indices) nogil:
    cdef int64_t i, idx, length
    cdef int64_t max_len = 0
    for i in range(num_indices):
        idx = indices_ptr[i]
        if i < 0 or i >= num_vecs:
            with gil:
                raise IndexError('index {0} is out of bounds for Vecs with {1} vectors'.format(i, num_vecs))
        length = lengths_ptr[idx]
        if length > max_len:
            max_len = length
    return max_len

cdef class FloatVecs(object):
    
    """
    This extention type implements a container for a list of variable length 1d FloatTensors.
    """
    
    cdef readonly:
        int64_t num_vecs
        list _list_of_vecs
        object _vec_locs
        int64_t vec_locs_loc
        object _lengths
        int64_t lengths_loc
        
        
    def __init__(self, list list_of_vecs):
        """ 
        Args:
            list of 1d FloatTensors or 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vecs = len(list_of_vecs)
        if self.num_vecs < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self._list_of_vecs = []
        self._lengths = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        self._vec_locs = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        for i, vec in enumerate(list_of_vecs):
            if isinstance(vec, ndarray):
                vec = torch.from_numpy(vec)
            if isinstance(vec, FloatTensor ):
                if vec.ndimension() != 1:
                    raise ValueError('Vectors must be one-dimensional')
                length = len(vec)
                if length < 1:
                    raise ValueError('The length of vectors must be at least 1')
            else:
                raise TypeError('list elements must be 1d FloatTensors or 1d ndarrays')
            vec = vec.clone()
            self._list_of_vecs.append( vec )
            self._lengths[i] = length
            self._vec_locs[i] = vec.data_ptr()
            
        self.vec_locs_loc = self._vec_locs.data_ptr()
        self.lengths_loc = self._lengths.data_ptr()
        
    def __getitem__(self, int64_t i ):
        if i < 0 or i >= self.num_vecs:
            raise IndexError('index {0} is out of bounds for FloatVecs with {1} vectors'.format(i, self.num_vecs))
        cdef int64_t* lengths_ptr = <int64_t*>self.lengths_loc
        cdef int64_t length = lengths_ptr[i]
        cdef ndarray out_vec = np.empty((length,), dtype=np.float32)
        cdef float32_t* out_ptr = <float32_t*>out_vec.data
        cdef int64_t* vec_locs_ptr = <int64_t*>self.vec_locs_loc
        cdef float32_t* vec_ptr = <float32_t*>vec_locs_ptr[i]
        cdef int64_t n
        with nogil:
            for n in range(length):
                out_ptr[n] = vec_ptr[n]
        return torch.from_numpy(out_vec)
    
    def make_padded_minibatch(self, object indices, float32_t fill_value):
        return make_padded_minibatch_from_float_vecs(self, indices, fill_value)
    
    def make_minibatch_with_random_lengths(self, object indices, float32_t fill_value, object out=None, int64_t max_len=10):
        return make_minibatch_with_random_lengths_from_float_vecs(self, indices, fill_value, out, max_len)
        
cdef tuple _get_indices_loc_and_num_indices(object indices):
    cdef int64_t indices_loc
    if isinstance(indices, LongTensor ):
        if indices.ndimension() != 1:
            raise ValueError('Indices must be one-dimensional')
        indices_loc = indices.data_ptr()
    elif isinstance(indices, ndarray):
        if indices.ndim != 1:
            raise ValueError('Indices must be one-dimensional')
        if indices.dtype != np.int64:
            raise TypeError('Indices must be of type int64')
        indices_loc = indices.ctypes.data
    else:
        raise TypeError('Indices must be a one-dimensional LongTensor of ndarray')
    
    cdef int64_t num_indices = len(indices)
    if num_indices < 1:
        raise ValueError('Indices must contain atleast 1 element')
        
    return indices_loc, num_indices

    
cpdef make_minibatch_with_random_lengths_from_float_vecs(FloatVecs Vecs, object indices, float32_t fill_value, 
                                                         object out=None, int64_t max_len=10):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    if out is None:
        out = torch.from_numpy( np.empty((num_indices,max_len), dtype=np.float32) )
    elif isinstance(out, FloatTensor):
        if out.ndimension() != 2:
            raise ValueError('out must be 2-dimensional')
        if out.size(0) != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.size(1)
    elif isinstance(out, ndarray):
        if out.dtype != np.float32:
            raise ValueError('out must be of dtype float32')
        if out.ndim != 2:
            raise ValueError('out must be 2-dimensional')
        if out.shape[0] != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.shape[1]
        out = torch.from_numpy(out)
    else:
        msg = "out must be a 2dimensional float32 ndarray or FloatTensor"
        raise TypeError(msg)
        
    urand_nums = torch.rand(num_indices)
    cdef int64_t urand_nums_loc = urand_nums.data_ptr()
    cdef float32_t* urand_nums_ptr = <float32_t*>urand_nums_loc
    
    cdef int64_t out_loc = out.data_ptr()
    cdef float32_t* out_ptr = <float32_t*>out_loc
    cdef int64_t i, idx, in_length, L, start_idx, j, l
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    cdef float32_t* out_vec
    cdef float32_t* in_vec
    
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        out_vec = &out_ptr[i*max_len]
        in_vec = <float32_t*>vec_locs_ptr[idx]
        in_length = lengths_ptr[idx]
        start_idx = (<int64_t>(urand_nums_ptr[i]*<float32_t>(in_length+max_len+1)))-max_len
        j = 0
        for l in range(start_idx,0):
            out_vec[j] = fill_value
            j += 1
        L = min(in_length,max_len+start_idx)
        for l in range(max(0,start_idx),L):
            out_vec[j] = in_vec[l]
            j += 1
        if (j+1) < max_len:
            for j in range(j,max_len):
                out_vec[j] = fill_value
    
    return out
    
cpdef object make_padded_minibatch_from_float_vecs(FloatVecs Vecs, object indices, float32_t fill_value):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    cdef int64_t max_len = _get_max_len( lengths_ptr, indices_ptr, Vecs.num_vecs, num_indices)
    
    
    cdef ndarray out = np.empty((num_indices, max_len), dtype=np.float32)
    cdef float32_t* out_ptr = <float32_t*>out.data
    cdef int64_t i, idx, in_length
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        fill_float_vec(<float32_t*>vec_locs_ptr[idx], lengths_ptr[idx], &out_ptr[i*max_len], max_len, fill_value)
        
    return torch.from_numpy(out)
    
cdef class IntVecs(object):
    
    """
    This extention type implements a container for a list of variable length 1d IntTensors.
    """
    
    cdef readonly:
        int64_t num_vecs
        list _list_of_vecs
        object _vec_locs
        int64_t vec_locs_loc
        object _lengths
        int64_t lengths_loc
        
        
    def __init__(self, list list_of_vecs):
        """ 
        Args:
            list of 1d IntTensors or 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vecs = len(list_of_vecs)
        if self.num_vecs < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self._list_of_vecs = []
        self._lengths = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        self._vec_locs = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        for i, vec in enumerate(list_of_vecs):
            if isinstance(vec, ndarray):
                vec = torch.from_numpy(vec)
            if isinstance(vec, IntTensor ):
                if vec.ndimension() != 1:
                    raise ValueError('Vectors must be one-dimensional')
                length = len(vec)
                if length < 1:
                    raise ValueError('The length of vectors must be at least 1')
            else:
                raise TypeError('list elements must be 1d IntTensors or 1d ndarrays')
            vec = vec.clone()
            self._list_of_vecs.append( vec )
            self._lengths[i] = length
            self._vec_locs[i] = vec.data_ptr()
            
        self.vec_locs_loc = self._vec_locs.data_ptr()
        self.lengths_loc = self._lengths.data_ptr()
        
    def __getitem__(self, int64_t i ):
        if i < 0 or i >= self.num_vecs:
            raise IndexError('index {0} is out of bounds for IntVecs with {1} vectors'.format(i, self.num_vecs))
        cdef int64_t* lengths_ptr = <int64_t*>self.lengths_loc
        cdef int64_t length = lengths_ptr[i]
        cdef ndarray out_vec = np.empty((length,), dtype=np.int32)
        cdef int32_t* out_ptr = <int32_t*>out_vec.data
        cdef int64_t* vec_locs_ptr = <int64_t*>self.vec_locs_loc
        cdef int32_t* vec_ptr = <int32_t*>vec_locs_ptr[i]
        cdef int64_t n
        with nogil:
            for n in range(length):
                out_ptr[n] = vec_ptr[n]
        return torch.from_numpy(out_vec)
    
    def make_padded_minibatch(self, object indices, int32_t fill_value):
        return make_padded_minibatch_from_int_vecs(self, indices, fill_value)
    
    def make_minibatch_with_random_lengths(self, object indices, int32_t fill_value, object out=None, int64_t max_len=10):
        return make_minibatch_with_random_lengths_from_int_vecs(self, indices, fill_value, out, max_len)
    
cpdef make_minibatch_with_random_lengths_from_int_vecs(IntVecs Vecs, object indices, int32_t fill_value, 
                                                         object out=None, int64_t max_len=10):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    if out is None:
        out = torch.from_numpy( np.empty((num_indices,max_len), dtype=np.int32) )
    elif isinstance(out, IntTensor):
        if out.ndimension() != 2:
            raise ValueError('out must be 2-dimensional')
        if out.size(0) != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.size(1)
    elif isinstance(out, ndarray):
        if out.dtype != np.int32:
            raise ValueError('out must be of dtype int32')
        if out.ndim != 2:
            raise ValueError('out must be 2-dimensional')
        if out.shape[0] != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.shape[1]
        out = torch.from_numpy(out)
    else:
        msg = "out must be a 2dimensional int32 ndarray or IntTensor"
        raise TypeError(msg)
        
    urand_nums = torch.rand(num_indices)
    cdef int64_t urand_nums_loc = urand_nums.data_ptr()
    cdef float32_t* urand_nums_ptr = <float32_t*>urand_nums_loc
    
    cdef int64_t out_loc = out.data_ptr()
    cdef int32_t* out_ptr = <int32_t*>out_loc
    cdef int64_t i, idx, in_length, L, start_idx, j, l
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    cdef int32_t* out_vec
    cdef int32_t* in_vec
    
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        out_vec = &out_ptr[i*max_len]
        in_vec = <int32_t*>vec_locs_ptr[idx]
        in_length = lengths_ptr[idx]
        start_idx = (<int64_t>(urand_nums_ptr[i]*<float32_t>(in_length+max_len+1)))-max_len
        j = 0
        for l in range(start_idx,0):
            out_vec[j] = fill_value
            j += 1
        L = min(in_length,max_len+start_idx)
        for l in range(max(0,start_idx),L):
            out_vec[j] = in_vec[l]
            j += 1
        if (j+1) < max_len:
            for j in range(j,max_len):
                out_vec[j] = fill_value
    
    return out
    
cpdef object make_padded_minibatch_from_int_vecs(IntVecs Vecs, object indices, int32_t fill_value):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    cdef int64_t max_len = _get_max_len( lengths_ptr, indices_ptr, Vecs.num_vecs, num_indices)
    
    
    cdef ndarray out = np.empty((num_indices, max_len), dtype=np.int32)
    cdef int32_t* out_ptr = <int32_t*>out.data
    cdef int64_t i, idx, in_length
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        fill_int_vec(<int32_t*>vec_locs_ptr[idx], lengths_ptr[idx], &out_ptr[i*max_len], max_len, fill_value)
        
    return torch.from_numpy(out)

cdef class ShortVecs(object):
    
    """
    This extention type implements a container for a list of variable length 1d ShortTensors.
    """
    
    cdef readonly:
        int64_t num_vecs
        list _list_of_vecs
        object _vec_locs
        int64_t vec_locs_loc
        object _lengths
        int64_t lengths_loc
        
        
    def __init__(self, list list_of_vecs):
        """ 
        Args:
            list of 1d ShortTensors or 1d ndarrays of variable length
        """
        # get number of vectors
        self.num_vecs = len(list_of_vecs)
        if self.num_vecs < 1:
            raise ValueError('list must contain atleast 1 vector')
        # check vectors, copy them and get their lengths and locations in memory
        self._list_of_vecs = []
        self._lengths = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        self._vec_locs = torch.from_numpy(np.empty((self.num_vecs,), dtype=np.int64))
        for i, vec in enumerate(list_of_vecs):
            if isinstance(vec, ndarray):
                vec = torch.from_numpy(vec)
            if isinstance(vec, ShortTensor ):
                if vec.ndimension() != 1:
                    raise ValueError('Vectors must be one-dimensional')
                length = len(vec)
                if length < 1:
                    raise ValueError('The length of vectors must be at least 1')
            else:
                raise TypeError('list elements must be 1d ShortTensors or 1d ndarrays')
            vec = vec.clone()
            self._list_of_vecs.append( vec )
            self._lengths[i] = length
            self._vec_locs[i] = vec.data_ptr()
            
        self.vec_locs_loc = self._vec_locs.data_ptr()
        self.lengths_loc = self._lengths.data_ptr()
        
    def __getitem__(self, int64_t i ):
        if i < 0 or i >= self.num_vecs:
            raise IndexError('index {0} is out of bounds for ShortVecs with {1} vectors'.format(i, self.num_vecs))
        cdef int64_t* lengths_ptr = <int64_t*>self.lengths_loc
        cdef int64_t length = lengths_ptr[i]
        cdef ndarray out_vec = np.empty((length,), dtype=np.int16)
        cdef int16_t* out_ptr = <int16_t*>out_vec.data
        cdef int64_t* vec_locs_ptr = <int64_t*>self.vec_locs_loc
        cdef int16_t* vec_ptr = <int16_t*>vec_locs_ptr[i]
        cdef int64_t n
        with nogil:
            for n in range(length):
                out_ptr[n] = vec_ptr[n]
        return torch.from_numpy(out_vec)
    
    def make_padded_minibatch(self, object indices, int16_t fill_value):
        return make_padded_minibatch_from_short_vecs(self, indices, fill_value)
    
    def make_minibatch_with_random_lengths(self, object indices, int16_t fill_value, object out=None, int64_t max_len=10):
        return make_minibatch_with_random_lengths_from_short_vecs(self, indices, fill_value, out, max_len)
    
cpdef make_minibatch_with_random_lengths_from_short_vecs(ShortVecs Vecs, object indices, int16_t fill_value, 
                                                         object out=None, int64_t max_len=10):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    if out is None:
        out = torch.from_numpy( np.empty((num_indices,max_len), dtype=np.int16) )
    elif isinstance(out, ShortTensor):
        if out.ndimension() != 2:
            raise ValueError('out must be 2-dimensional')
        if out.size(0) != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.size(1)
    elif isinstance(out, ndarray):
        if out.dtype != np.int16:
            raise ValueError('out must be of dtype int16')
        if out.ndim != 2:
            raise ValueError('out must be 2-dimensional')
        if out.shape[0] != num_indices:
            raise ValueError('the number of rows in out must be the same as the length of indices')
        max_len = out.shape[1]
        out = torch.from_numpy(out)
    else:
        msg = "out must be a 2dimensional int16 ndarray or ShortTensor"
        raise TypeError(msg)
        
    urand_nums = torch.rand(num_indices)
    cdef int64_t urand_nums_loc = urand_nums.data_ptr()
    cdef float32_t* urand_nums_ptr = <float32_t*>urand_nums_loc
    
    cdef int64_t out_loc = out.data_ptr()
    cdef int16_t* out_ptr = <int16_t*>out_loc
    cdef int64_t i, idx, in_length, L, start_idx, j, l, T
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    cdef int16_t* out_vec
    cdef int16_t* in_vec
    
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        out_vec = &out_ptr[i*max_len]
        in_vec = <int16_t*>vec_locs_ptr[idx]
        in_length = lengths_ptr[idx]
        start_idx = (<int64_t>(urand_nums_ptr[i]*<float32_t>(in_length+max_len+1)))-max_len
        j = 0
        for l in range(start_idx,0):
            out_vec[j] = fill_value
            j += 1
        L = min(in_length,max_len+start_idx)
        for l in range(max(0,start_idx),L):
            out_vec[j] = in_vec[l]
            j += 1
        if (j+1) < max_len:
            for j in range(j,max_len):
                out_vec[j] = fill_value
    
    return out
    
cpdef object make_padded_minibatch_from_short_vecs(ShortVecs Vecs, object indices, int16_t fill_value):
    
    cdef int64_t indices_loc, num_indices
    indices_loc, num_indices = _get_indices_loc_and_num_indices(indices)
    cdef int64_t* indices_ptr = <int64_t*>indices_loc
    cdef int64_t* lengths_ptr = <int64_t*>Vecs.lengths_loc
    
    cdef int64_t max_len = _get_max_len( lengths_ptr, indices_ptr, Vecs.num_vecs, num_indices)
    
    
    cdef ndarray out = np.empty((num_indices, max_len), dtype=np.int16)
    cdef int16_t* out_ptr = <int16_t*>out.data
    cdef int64_t i, idx, in_length
    cdef int64_t* vec_locs_ptr = <int64_t*>Vecs.vec_locs_loc
    for i in prange(num_indices, nogil=True, schedule='static'):
        idx = indices_ptr[i]
        fill_short_vec(<int16_t*>vec_locs_ptr[idx], lengths_ptr[idx], &out_ptr[i*max_len], max_len, fill_value)
        
    return torch.from_numpy(out)
    

def make_padded_minibatch(Vecs, indices, fill_value):
    if isinstance(Vecs, FloatVecs):
        return make_padded_minibatch_from_float_vecs(Vecs, indices, fill_value)
    else:
        raise NotImplementedError
    
def make_padded_minibatch_with_random_lengths(Vecs, indices, fill_value, out=None, max_len=10):
    if isinstance(Vecs, FloatVecs):
        return make_minibatch_with_random_lengths_from_float_vecs(Vecs, indices, fill_value, out, max_len)
    elif isinstance(Vecs, IntVecs):
        return make_minibatch_with_random_lengths_from_int_vecs(Vecs, indices, fill_value, out, max_len)
    elif isinstance(Vecs, ShortVecs):
        return make_minibatch_with_random_lengths_from_short_vecs(Vecs, indices, fill_value, out, max_len)
    else:
        raise NotImplementedError
        
    
