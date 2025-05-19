#include <cooperative_groups.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include <math.h>
namespace apr{

template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}
/*、
    int stride_qm,int stride_qk,
    int stride_kk,int stride_kn,
    int stride_vm,int stride_vn
*/
template<const int num_threads_x,
        const int num_threads_y,
        const int d_model,
        const int d_q,
        const int block_m,
        const int block_n
>
__global__ void flash_kernel(float* Q,float* K,float* V,float* O,float* L,float* M){
    /*
    block_m is b_r
    block_n is b_c
    N is d_model

    m in d_model n in d_q k in d_v

    o_size = d_model x d_q

    q_size = d_model x d_q
    k_size = d_model x d_q
    v_size = d_model x d_q
    */

    __shared__ float q[block_m][d_q];
    __shared__ float k[block_n][d_q];
    __shared__ float v[block_n][d_q];
    __shared__ float o[block_m][d_q];
    __shared__ float s[block_m][block_n];

    __shared__ float l[block_m];
    __shared__ float m[block_m];
    int thread_num_inblock = num_threads_x*num_threads_y; 
    int T_r =  (int)(d_model / block_m);
    int T_c =  (int)(d_model / block_n);
    //global idx
    unsigned int g_idx = threadIdx.x + threadIdx.y*blockDim.x + thread_num_inblock*blockIdx.x;//grid must be a one dimensional vector like <<<1 ,(256,32) >>> 
    unsigned int b_idx = threadIdx.x + threadIdx.y*blockDim.x;
    unsigned int warpId = b_idx / 32;
    unsigned int laneId = b_idx % 32;
    //assumption that is one data for one thread and block_m == block_n
    
    //load gmem to smem
    if(g_idx < d_q*d_model){
        int data_num_block_q = d_q * block_m;//for q and o 
        int data_num_block_k = d_q * block_n;//for k v
        int row_kv = ((g_idx / d_q)%block_n);
        int row_qo = ((g_idx / d_q)%block_m);
        v[row_kv][g_idx % d_q] = V[g_idx];
        k[row_kv][g_idx % d_q] = K[g_idx];
        for(int iter = 0;iter < T_r;iter++){
            q[row_qo][g_idx%d_q] = Q[g_idx % data_num_block_q + iter*block_m*d_q];
            o[row_kv][g_idx%d_q] = O[g_idx % data_num_block_q + iter*block_m*d_q];
            if(b_idx < block_m){
                l[b_idx % block_m] = L[b_idx % block_m + iter*block_m];
                m[b_idx % block_m] = M[b_idx % block_m + iter*block_m];
            }
            //mat multiply
            //semm
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            if(b_idx < d_q*block_m){
                __shared__ float tmp_sum[block_m][d_q];
                for(int z = 0;z < block_n;z++){
                    tmp_sum[ty][tx] = q[ty][tx]*k[z][tx];
                    
                    if(tx == 0){
                        float sum = 0;
                        for(int i=0;i < d_q;i++){
                            sum += tmp_sum[ty][i];
                        }
                        s[ty][z] = sum;//s_ij
                        
                        
                    }
                    __syncthreads();
                    tmp_sum[ty][tx] = 0;
                }
            }
            
            //step 10
            //calculate the max
            __shared__ float m_up[block_m];
            __shared__ float l_up[block_m];
            if(tx == 0)
            {
                float big = INT_MIN;
                for(int z=0;z < block_n;z++){
                    big = max(big,s[ty][z]);
                }
                m_up[ty] = big;
            }
            __syncthreads();
            //calculate the p_ij
            if(b_idx < block_m * block_n){
                int row = b_idx / block_n;
                int col = b_idx % block_n;
                s[row][col] = __expf(s[row][col]-m_up[row]);//p_ij
            }
            __syncthreads();
            //calculate the l_up_ij
            if(tx == 0)
            {
                float sum = 0;
                for(int z=0;z < block_n;z++){
                    sum += s[ty][z];//l_up_ij
                }
                l_up[ty] = sum;
            }
            __syncthreads();
            //step 11
            __shared__ float m_new[block_m];
            __shared__ float l_new[block_m];
            if(b_idx == 0){
                for(int i=0;i < block_m;i++){
                    m_new[i] = max(m[i],m_up[i]);
                    l_new[i] = __expf(m[i]-m_new[i])*l[i] + l_up[i]*__expf(m_up[i]-m_new[i]);
                }
                
            }
            __syncthreads();
            //calculate o_i
            __shared__ float pv[block_m][d_q];
            int row_s = b_idx / block_n;
            int col_s = b_idx % block_n;
            __shared__ float acc[block_m][block_n];
            //mat mutliply
            for(int z = 0;z < d_q;z++){
                if(b_idx < block_m*block_n){
                    acc[row_s][col_s] = s[row_s][col_s]*v[col_s][z];//p_ij*v_ij
                }
                
                //__syncthreads();
                //reduce it or just sum it up
                if(b_idx < block_m*block_n && col_s == 0){
                    float sum = 0;
                    for(int i=0;i < block_n;i++){
                        sum += acc[row_s][i];
                    }
                    pv[row_s][z] = sum;
                }
                __syncthreads();
                if(b_idx < block_m*block_n){
                    acc[row_s][col_s] = 0;
                }
            }
            o[row_qo][g_idx%d_q] = l[row_qo]*__expf(m[row_qo]-m_new[row_qo])*o[row_qo][g_idx%d_q] + __expf(m_up[row_qo]-m_new[row_qo])*pv[row_qo][g_idx%d_q];
            //write back L and M
            if(b_idx < block_m){
                L[b_idx % block_m + iter*block_m] = l_new[b_idx];
                M[b_idx % block_m + iter*block_m] = m_new[b_idx];
            }
            
            //write back to o
            O[g_idx % data_num_block_q + iter*block_m*d_q] = o[row_qo][g_idx%d_q];
            

        }
    }
    
    //debug 思路 只保留少数线程进行计算即可
}//线程分配的困难，因为不是一个块无法共享内存，无法使用sram进行通信，所以划分块和数据的索引变成了最大的困难，目前的方法每个块里面在一些计算阶段都会有很多闲置线程
at::Tensor flash_cuda(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                     int64_t n, int64_t m, int64_t d_k, int64_t d_v) {
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
  constexpr int block_m = 32;
  constexpr int block_n = 32;
  
  // Define thread block dimensions
  constexpr int num_threads_x = 16;
  constexpr int num_threads_y = 16;
  dim3 threads(num_threads_x, num_threads_y);
  
  // Calculate grid dimensions based on sequence length
  int grid_dim = (seq_len + block_m - 1) / block_m;
  
  // Launch kernel with template parameters using the provided dimensions
  flash_kernel<num_threads_x, num_threads_y, 32, 32, block_m, block_n>
      <<<grid_dim, threads>>>(Q_ptr, K_ptr, V_ptr, result_ptr, L_ptr, M_ptr);

  return result;
}

// Registers CUDA implementations for flash
TORCH_LIBRARY_IMPL(apr, CUDA, m) {
  m.impl("flash", &flash_cuda);
  
}

}//namespace apr
