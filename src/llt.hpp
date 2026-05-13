#pragma once
// LLt  complex Cholesky on GPU.
// Called from scalar_array.cpp in lltDecomposition() for C_t and Z_t (complex & double complex)

#ifdef HAVE_CUDA
void llt_complex_cuda_C(void* mat, int n, int lda); 
void llt_complex_cuda_Z(void* mat, int n, int lda); 
#endif