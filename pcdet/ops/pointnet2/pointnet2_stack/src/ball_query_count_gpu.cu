/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"

//**
 * @brief 
 * 
 * @param B batch_size
 * @param M grid_point的个数
 * @param radius 搜索的半径
 * @param nsample 半径内搜索多少个点
 * @param new_xyz grid_point的坐标
 * @param new_xyz_batch_cnt 每个batch中有多少个grid point
 * @param xyz 特征点的坐标
 * @param xyz_batch_cnt 每个batch中有多少个特征点
 * @param idx [grid point个数, nsample]
 * @return int 
 */
__global__ void ball_query_count_kernel_stack(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    //* 获取当前处理的grid_point的索引
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    //* 半径的平方
    float radius2 = radius * radius;
    //* 当前处理的grid point的坐标
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    //* 当前处理的batch的特征点个数
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        //* 如果特征点在这个grid point的半径内, 就把这个特征点的索引赋值给idx
        if (d2 < radius2){
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    //* idx的第一个是-1表示这个grid point半径内没有点
    if (cnt == 0) idx[0] = -1;
}

//**
 * @brief 
 * 
 * @param B batch_size
 * @param M grid_point的个数
 * @param radius 搜索的半径
 * @param nsample 半径内搜索多少个点
 * @param new_xyz grid_point的坐标
 * @param new_xyz_batch_cnt 每个batch中有多少个grid point
 * @param xyz 特征点的坐标
 * @param xyz_batch_cnt 每个batch中有多少个特征点
 * @param idx [grid point个数, nsample]
 * @return int 
 */
void ball_query_count_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_count_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
