#include <torch/all.h>
#include <torch/library.h>
#include <vector>
#include <Python.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace apr {



/*
    Q = {n x d_k}
    K = {m x d_k}
    V = {m x d_v}
    scores = {n x m}
    output = {n x d_v}
*/
at::Tensor flash_cpu(const at::Tensor& Q, const at::Tensor& K, const at::Tensor& V,
                 int64_t n, int64_t m, int64_t d_k, int64_t d_v) {
    // Compute Q*K^T
    auto scores = torch::zeros({n, m}, Q.options());
    auto q_accessor = Q.accessor<float, 2>();
    auto k_accessor = K.accessor<float, 2>();
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d_k; ++k) {
                sum += q_accessor[i][k] * k_accessor[j][k];
            }
            scores[i][j] = sum;
        }
    }
    
    // Scale by sqrt(d_k)
    scores = scores / std::sqrt(static_cast<float>(d_k));
    
    // Apply softmax row-wise
    scores = torch::softmax(scores, 1);
    
    // Compute attention weighted values
    auto output = torch::zeros({n, d_v}, V.options());
    auto v_accessor = V.accessor<float, 2>();
    auto scores_accessor = scores.accessor<float, 2>();
    
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < d_v; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < m; ++j) {
                sum += scores_accessor[i][j] * v_accessor[j][k];
            }
            output[i][k] = sum;
        }
    }
    
    return output;
}

// Defines the operators
TORCH_LIBRARY(apr, m) {
  m.def("flash(Tensor Q, Tensor K, Tensor V, int n, int m, int d_k, int d_v) -> Tensor");
  //h_attention(h_q,h_k,h_v,d_model,d_q,d_q,d_q);
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(apr, CPU, m) {
  m.impl("flash", &flash_cpu);
  
}
}//namespace apr