// llt_complex_cuda.cu
/// Block Cholesky decomposition (A = L * L^T).

#include <cuda_runtime.h> 
#include <cublas_v2.h>    
#include <cuComplex.h>    
#include <stdio.h>        
#include <math.h>         

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d \n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ─────────────────────────────────────────
//  Special functions for complex numbers executed on the GPU
// ─────────────────────────────────────────


__device__ inline cuFloatComplex cmul_T(cuFloatComplex a, cuFloatComplex b) { return cuCmulf(a, b); }
__device__ inline cuDoubleComplex cmul_T(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }


__device__ inline cuFloatComplex csub(cuFloatComplex a, cuFloatComplex b) { return cuCsubf(a, b); }
__device__ inline cuDoubleComplex csub(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a, b); }

__device__ inline cuFloatComplex cdiv(cuFloatComplex a, cuFloatComplex b) { return cuCdivf(a, b); }
__device__ inline cuDoubleComplex cdiv(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a, b); }


// Complex square  evaluated using Cartesian coordinates.

__device__ inline cuFloatComplex csqrt_llt(cuFloatComplex v) {
    if (v.x == 0.0f && v.y == 0.0f) return make_cuFloatComplex(0.0f, 0.0f); 
    float x = v.x; float y = v.y; 
    float r = cuCabsf(v);         
    
    float u = sqrtf((r + fabsf(x)) * 0.5f); 
    if (x >= 0.0f) {
        return make_cuFloatComplex(u, y / (2.0f * u));
    } else {
        float real_part = fabsf(y) / (2.0f * u);
        float imag_part = (y >= 0.0f) ? u : -u;
        return make_cuFloatComplex(real_part, imag_part);
    }
}


__device__ inline cuDoubleComplex csqrt_llt(cuDoubleComplex v) {
    if (v.x == 0.0 && v.y == 0.0) return make_cuDoubleComplex(0.0, 0.0);
    double x = v.x; double y = v.y;
    double r = cuCabs(v);
    double u = sqrt((r + fabs(x)) * 0.5);
    if (x >= 0.0) {
        return make_cuDoubleComplex(u, y / (2.0 * u));
    } else {
        double real_part = fabs(y) / (2.0 * u);
        double imag_part = (y >= 0.0) ? u : -u;
        return make_cuDoubleComplex(real_part, imag_part);
    }
}

// ─────────────────────────────────────────
// CUDA Kernels  
// ─────────────────────────────────────────



// STEP 1: Apply the square to the diagonal element of the current column.
template <typename CuT>
__global__ void llt_diag(CuT* __restrict__ A, int lda, int j) {
    
    A[j + j * lda] = csqrt_llt(A[j + j * lda]);
}

//STEP 2: Divide all elements below the diagonal by the diagonal value.
template <typename CuT>
__global__ void llt_col_scale(CuT* __restrict__ A, int lda, int j, int n) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (i < n) { 
        CuT ljj = A[j + j * lda]; 
        A[i + j * lda] = cdiv(A[i + j * lda], ljj); 
    }
}

// STEP 3: Update the submatrix.
template <typename CuT>
__global__ void llt_update(CuT* __restrict__ A, int lda, int j, int n, int panel_end_col) { 
    int row = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    int col = blockIdx.y * blockDim.y + threadIdx.y + j + 1;

    
    if (col < panel_end_col && row < n && row >= col) {
        CuT lik = A[row + j * lda]; 
        CuT ljk = A[col + j * lda]; 
         
        A[row + col * lda] = csub(A[row + col * lda], cmul_T(lik, ljk));
    }
}

// ─────────────────────────────────────────
// Main host functions
// ─────────────────────────────────────────




// Simple Précision version
void llt_complex_cuda_C(void* mat, int n, int lda) {
    cuFloatComplex* d_A = nullptr; 
    size_t sizeA = sizeof(cuFloatComplex) * (size_t)lda * n; 

    CUDA_CHECK(cudaMalloc(&d_A, sizeA)); 
    CUDA_CHECK(cudaMemcpy(d_A, mat, sizeA, cudaMemcpyHostToDevice)); 

    cublasHandle_t handle; 
    CUBLAS_CHECK(cublasCreate(&handle)); 
    // Enables the use of Tensor Cores for matrix multiplication.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    const int B = 256;  
    const int BLOCK_1D = 256; 
    dim3 BLOCK_2D(16, 16); 
    
    
    cuFloatComplex alpha = make_cuFloatComplex(-1.0f, 0.0f);
    cuFloatComplex beta  = make_cuFloatComplex(1.0f, 0.0f);

    
    for (int j = 0; j < n; j += B) {
        int b_size = (j + B <= n) ? B : (n - j); 
        int panel_end = j + b_size; 

      
        for (int k = 0; k < b_size; k++) {
            int current_col = j + k; // 
            
            
            llt_diag<cuFloatComplex><<<1, 1>>>(d_A, lda, current_col);
            
            int rows_below = n - current_col - 1;  
            if (rows_below > 0) {
                int grid1D = (rows_below + BLOCK_1D - 1) / BLOCK_1D; 
                
                llt_col_scale<cuFloatComplex><<<grid1D, BLOCK_1D>>>(d_A, lda, current_col, n);
                
                int panel_cols_remaining = panel_end - current_col - 1;
                if (panel_cols_remaining > 0) {
                    dim3 grid2D((rows_below + BLOCK_2D.x - 1) / BLOCK_2D.x, 
                                (panel_cols_remaining + BLOCK_2D.y - 1) / BLOCK_2D.y);
                     
                    llt_update<cuFloatComplex><<<grid2D, BLOCK_2D>>>(d_A, lda, current_col, n, panel_end);
                }
            }
        } 

       
        int trailing_size = n - panel_end; 
        if (trailing_size > 0) {
            cuFloatComplex* L_panel = d_A + panel_end + j * lda;       
            cuFloatComplex* A_sub   = d_A + panel_end + panel_end * lda; 

            CUBLAS_CHECK(cublasCsyrk(
                handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                trailing_size, b_size,
                &alpha, L_panel, lda,
                &beta, A_sub, lda
            ));
        }
    }
    
    CUDA_CHECK(cudaMemcpy(mat, d_A, sizeA, cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(handle)); 
    CUDA_CHECK(cudaFree(d_A)); 
}

//double complex version
void llt_complex_cuda_Z(void* mat, int n, int lda) {
    cuDoubleComplex* d_A = nullptr;
    size_t sizeA = sizeof(cuDoubleComplex) * (size_t)lda * n;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMemcpy(d_A, mat, sizeA, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    

    const int B = 256; 
    const int BLOCK_1D = 256;
    dim3 BLOCK_2D(16, 16);

    cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0); 
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0);

    for (int j = 0; j < n; j += B) {
        int b_size = (j + B <= n) ? B : (n - j);
        int panel_end = j + b_size;

        for (int k = 0; k < b_size; k++) {
            int current_col = j + k;
            llt_diag<cuDoubleComplex><<<1, 1>>>(d_A, lda, current_col);
            int rows_below = n - current_col - 1;
            if (rows_below > 0) {
                int grid1D = (rows_below + BLOCK_1D - 1) / BLOCK_1D;
                llt_col_scale<cuDoubleComplex><<<grid1D, BLOCK_1D>>>(d_A, lda, current_col, n);
                int panel_cols_remaining = panel_end - current_col - 1;
                if (panel_cols_remaining > 0) {
                    dim3 grid2D((rows_below + BLOCK_2D.x - 1) / BLOCK_2D.x, 
                                (panel_cols_remaining + BLOCK_2D.y - 1) / BLOCK_2D.y);
                    llt_update<cuDoubleComplex><<<grid2D, BLOCK_2D>>>(d_A, lda, current_col, n, panel_end);
                }
            }
        }

        int trailing_size = n - panel_end;
        if (trailing_size > 0) {
            cuDoubleComplex* L_panel = d_A + panel_end + j * lda;       
            cuDoubleComplex* A_sub   = d_A + panel_end + panel_end * lda; 
            CUBLAS_CHECK(cublasZsyrk(
                handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                trailing_size, b_size,
                &alpha, L_panel, lda,
                &beta, A_sub, lda
            ));
        }
    }
    CUDA_CHECK(cudaMemcpy(mat, d_A, sizeA, cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
}

