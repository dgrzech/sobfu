#pragma once

/* kinfu includes */
#include <kfusion/cuda/tsdf_volume.hpp>

/* sobfu incldues */
#include <sobfu/precomp.hpp>
#include <sobfu/vector_fields.hpp>

/* thrust includes */
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

namespace sobfu {
namespace device {

/*
 * REDUCTOR
 */

struct Reductor {
    /* constructor */
    Reductor(int3 dims_, float vsz_, float trunc_dist_);
    /* destructor */
    ~Reductor();

    /* energy functional */
    float data_energy(float2 *phi_global_data, float2 *phi_n_data);
    float reg_energy_sobolev(Mat4f *J_data);

    float2 max_update_norm();
    float2 voxel_max_energy(float2 *phi_global_data, float2 *phi_n_data, Mat4f *J_data, float w_reg);

    int3 dims;
    float vsz, trunc_dist;
    int no_voxels;

    int blocks,
        threads; /* no. of blocks and threads for the reductions used to calculate value of the energy functional */

    float4 *updates;

    float *h_data_out, *d_data_out;
    float *h_reg_out, *d_reg_out;

    float2 *h_max_out, *d_max_out;
};

/* kernels */
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_data_kernel(float2 *g_idata_global, float2 *g_idata_n, float *g_odata, unsigned int n);
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_reg_sobolev_kernel(Mat4f *g_idata, float *g_odata, unsigned int n);

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_voxel_max_energy_kernel(float2 *d_idata_global, float2 *d_idata_n, Mat4f *d_idata_reg,
                                               float2 *d_o_data, float w_reg, unsigned int n);
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_max_kernel(float4 *updates, float2 *g_o_max_data, unsigned int n);

/* host methods */
void reduce_data(int size, int threads, int blocks, float2 *d_idata_global, float2 *d_idata_n, float *d_odata);
void reduce_reg_sobolev(int size, int threads, int blocks, Mat4f *d_idata, float *d_odata);

void reduce_voxel_max_energy(int size, int threads, int blocks, float2 *d_idata_global, float2 *d_idata_n,
                             Mat4f *d_idata_reg, float w_reg, float2 *d_odata);
void reduce_max(int size, int threads, int blocks, float4 *updates, float2 *d_o_max_data);

/* final reduce on the cpu */
float final_reduce(float *h_odata, float *d_odata, int numBlocks);
float2 final_reduce_max(float2 *h_o_max_data, float2 *d_o_max_data, int numBlocks, int3 dims);

}  // namespace device
}  // namespace sobfu
