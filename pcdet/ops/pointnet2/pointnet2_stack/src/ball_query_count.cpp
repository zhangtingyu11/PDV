/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query_count_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
  if (!x.type().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

//**
 * @brief 
 * 
 * @param B batch_size
 * @param M grid_point的个数
 * @param radius 搜索的半径
 * @param nsample 半径内搜索多少个点
 * @param new_xyz_tensor grid_point的坐标
 * @param new_xyz_batch_cnt_tensor 每个batch中有多少个grid point
 * @param xyz_tensor 特征点的坐标
 * @param xyz_batch_cnt_tensor 每个batch中有多少个特征点
 * @param idx_tensor [grid point个数, nsample]
 * @return int 
 */
int ball_query_count_wrapper_stack(int B, int M, float radius, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(new_xyz_batch_cnt_tensor);
    CHECK_INPUT(xyz_batch_cnt_tensor);

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const int *new_xyz_batch_cnt = new_xyz_batch_cnt_tensor.data<int>();
    const int *xyz_batch_cnt = xyz_batch_cnt_tensor.data<int>();
    int *idx = idx_tensor.data<int>();

    ball_query_count_kernel_launcher_stack(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    return 1;
}
