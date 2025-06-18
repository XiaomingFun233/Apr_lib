#include <cooperative_groups.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include <math.h>
namespace apr{

template<const int num_threads_x,
        const int num_threads_y,
        const int N,
        const int d,
        const int b_r,
        const int b_c
>
__global__ void flash_kernel(float* Q,float* K,float* V,float* O,float* L,float* M){
    /*
    parameter: name base on paper

    m in N
    n in d_k
    k in d_v

    o_size = N x d

    q_size = N x d
    k_size = N x d
    v_size = N x d
    */

    //SRAM allocation step 3 in paper
    __shared__ float q[b_r][d];
    __shared__ float k[b_c][d];
    __shared__ float v[b_c][d];
    int T_r =  (int)(N / b_r); // divide operations are slow in GPU find a way to replace it
    int T_c =  (int)(N / b_c);
    //SRAM allocation step 4 in paper
    __shared__ float o[b_r][d];
    __shared__ float s[b_r][b_c];
    __shared__ float l[b_r];
    __shared__ float m[b_r];
    float scale = 1.0f / sqrtf((float)d);
    //calculate the thread num in block and the thread num in row and col
    int thread_num_inblock = num_threads_x*num_threads_y;
    unsigned int b_idx = threadIdx.x + threadIdx.y*blockDim.x;// 2D matrix but 1D idx
    int data_num_block_q = d * b_r;//for q and o
    int data_num_block_k = d * b_c;//for k v
    //assumption that is one data for one thread and b_r == b_c
   
    //load gmem to smem
    for(int j = 0;j < T_c;j++){//step 5
        for(int idx = b_idx;idx < data_num_block_k;idx += blockDim.x * blockDim.y){
            
            v[idx / d][idx % d] = V[idx  + j * data_num_block_k];//each block load its own gmem to smem
            k[idx / d][idx % d] = K[idx  + j * data_num_block_k];//in paper it is step 6
        }
        __syncthreads();
        for(int iter = 0;iter < T_r;iter++){//step 7
           for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
                int row = i / d;
                int col = i % d;
                q[row][col] = Q[iter * b_r * d + i];
                o[row][col] = O[iter * b_r * d + i];
            }

            for(int idx = b_idx;idx < b_r;idx += blockDim.x * blockDim.y){
                l[idx] = L[idx + iter*b_r];
                m[idx] = M[idx + iter*b_r];
            }
            __syncthreads();
            //mat multiply
            //semm step 9
            int tx = threadIdx.x;
            int ty = threadIdx.y;
           
            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < b_c; col += blockDim.x) {
                    float sum = 0.0f;
                    for (int k_dim = 0; k_dim < d; k_dim++) {
                        sum += q[row][k_dim] * k[col][k_dim];
                    }
                    // *** BUG FIX 1: 增加了缩放步骤 ***
                    s[row][col] = sum * scale;
                }
            }
            __syncthreads(); // Essential: All threads must finish computing their part of 's'
           
            //step 10
            //calculate the max
            __shared__ float m_up[b_r];
            __shared__ float l_up[b_r];
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    // 1. 找最大值 m_up
                    float row_max = -FLT_MAX;
                    for (int col = 0; col < b_c; col++) {
                        if (s[row][col] > row_max) {
                            row_max = s[row][col];
                        }
                    }
                    m_up[row] = row_max;

                    // 2. 计算 P_ij 并求和得到 l_up
                    float row_sum_exp = 0.0f;
                    for (int col = 0; col < b_c; col++) {
                        float p_val = __expf(s[row][col] - row_max);
                        s[row][col] = p_val; // 将 s 矩阵原地更新为 P 矩阵
                        row_sum_exp += p_val;
                    }
                    l_up[row] = row_sum_exp;
                }
            }
            __syncthreads();

            //step 11
            __shared__ float m_new[b_r];
            __shared__ float l_new[b_r];
            if (tx == 0) {
                for (int row = ty; row < b_r; row += blockDim.y) {
                    m_new[row] = fmaxf(m[row], m_up[row]);
                    l_new[row] = __expf(m[row] - m_new[row]) * l[row] + __expf(m_up[row] - m_new[row]) * l_up[row];
                }
            }
            __syncthreads();
            //calculate o_i
            __shared__ float pv[b_r][d];
            // Each thread (ty, tx) will be responsible for computing one element pv[ty][tx].
            // This requires iterating through the inner dimension 'b_c'.

            for (int row = ty; row < b_r; row += blockDim.y) {
                for (int col = tx; col < d; col += blockDim.x) {
                    float pv_sum = 0.0f;
                    for (int k_dim = 0; k_dim < b_c; k_dim++) {
                        pv_sum += s[row][k_dim] * v[k_dim][col];
                    }
                    pv[row][col] = pv_sum;
                }
            }
            __syncthreads();
            for (int row = ty; row < b_r; row += blockDim.y) {
                float m_old = m[row];
                float l_old = l[row];
                float m_new_val = m_new[row];
                float l_new_val = l_new[row];

                for (int col = tx; col < d; col += blockDim.x) {
                    float o_old = o[row][col];
                    float pv_val = pv[row][col];
                    // 更新公式
                    o[row][col] = (l_old * __expf(m_old - m_new_val) * o_old + __expf(m_up[row] - m_new_val) * pv_val) / l_new_val;
                }
            }
             __syncthreads();
             if (tx == 0) {
                 for (int row = ty; row < b_r; row += blockDim.y) {
                    l[row] = l_new[row];
                    m[row] = m_new[row];
                 }
            }
            __syncthreads();

            for (int i = b_idx; i < b_r * d; i += blockDim.x * blockDim.y) {
                O[iter * b_r * d + i] = o[i / d][i % d];
            }
            for (int i = b_idx; i < b_r; i += blockDim.x * blockDim.y) {
                L[iter * b_r + i] = l[i];
                M[iter * b_r + i] = m[i];
            }
            __syncthreads();
        }
        
    }
   
   
}//thread block must bigger than max(b_r,b_c) * d
at::Tensor flash_cuda(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,int n,int m,int d_k,int d_v) {
  // Check that K and V have the same dimensions
  TORCH_CHECK(K.sizes() == V.sizes(), "K and V must have the same dimensions");
  // Check that the last dimension of Q matches the last dimension of K
  TORCH_CHECK(Q.size(-1) == K.size(-1), "Last dimension of Q must match last dimension of K");
  
  TORCH_CHECK(Q.dtype() == at::kFloat, "Q must be float tensor");
  TORCH_CHECK(K.dtype() == at::kFloat, "K must be float tensor");
  TORCH_CHECK(V.dtype() == at::kFloat, "V must be float tensor");
  
  TORCH_INTERNAL_ASSERT(Q.device().type() == at::DeviceType::CUDA, "Q must be CUDA tensor");
  TORCH_INTERNAL_ASSERT(K.device().type() == at::DeviceType::CUDA, "K must be CUDA tensor");
  TORCH_INTERNAL_ASSERT(V.device().type() == at::DeviceType::CUDA, "V must be CUDA tensor");
  
  at::Tensor Q_contig = Q.contiguous();
  at::Tensor K_contig = K.contiguous();
  at::Tensor V_contig = V.contiguous();
  
  // Define the output shape - same as Q for flash attention
  at::Tensor result = torch::empty(Q_contig.sizes(), Q_contig.options());
  
  // Create auxiliary tensors for L and M
  auto batch_size = Q_contig.size(0);
  auto seq_len = Q_contig.size(1);
  
  at::Tensor L = torch::zeros({batch_size, seq_len}, Q_contig.options());
  at::Tensor M = torch::zeros({batch_size, seq_len}, Q_contig.options());
  
  // Extract pointers to the data
  float* Q_ptr = Q_contig.data_ptr<float>();
  float* K_ptr = K_contig.data_ptr<float>();
  float* V_ptr = V_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  float* L_ptr = L.data_ptr<float>();
  float* M_ptr = M.data_ptr<float>();
  
  // Define block sizes
  const int block_m = 4;
  const int block_n = 4;
  
  // Define thread block dimensions
  const int num_threads_x = 128;
  const int num_threads_y = 2;
  dim3 threads(num_threads_x, num_threads_y);
  
  // Calculate grid dimensions based on sequence length
  int grid_dim = (seq_len + block_m - 1) / block_m;
  
  // Launch kernel with template parameters using the provided dimensions
  flash_kernel<num_threads_x, num_threads_y, /*d_model=*/m, /*d_q=*/d_k, block_m, block_n>
      <<<grid_dim, threads>>>(Q_ptr, K_ptr, V_ptr, result_ptr, L_ptr, M_ptr);

  return result;
}

// Registers CUDA implementations for flash
TORCH_LIBRARY_IMPL(apr, CUDA, m) {
  m.impl("flash", &flash_cuda);
  
}

}//namespace apr
