/* sobfu incldues */
#include <sobfu/reductor.hpp>

/*
 * REDUCTOR
 */

sobfu::device::Reductor::Reductor(int3 dims_, float vsz_, float trunc_dist_) {
    dims       = dims_;
    vsz        = vsz_;
    trunc_dist = trunc_dist_;

    no_voxels = dims.x * dims.y * dims.z;

    /* own kernels */
    get_num_blocks_and_threads(no_voxels, 65536, 512, blocks, threads);

    h_data_out = new float[blocks];
    h_reg_out  = new float[blocks];
    h_max_out  = new float2[blocks];

    cudaSafeCall(cudaMalloc((void **) &d_data_out, blocks * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **) &d_reg_out, blocks * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **) &d_max_out, blocks * sizeof(float2)));

    cudaSafeCall(cudaMalloc((void **) &updates, no_voxels * sizeof(float4)));
}

sobfu::device::Reductor::~Reductor() {
    delete h_data_out, h_reg_out, h_max_out;

    cudaSafeCall(cudaFree(d_data_out));
    cudaSafeCall(cudaFree(d_reg_out));
    cudaSafeCall(cudaFree(d_max_out));
    cudaSafeCall(cudaFree(updates));
}

float sobfu::device::Reductor::data_energy(float2 *phi_global_data, float2 *phi_n_data) {
    reduce_data(no_voxels, threads, blocks, phi_global_data, phi_n_data, d_data_out);
    cudaSafeCall(cudaGetLastError());

    return 0.5f * final_reduce(h_data_out, d_data_out, blocks);
}

float sobfu::device::Reductor::reg_energy_sobolev(Mat4f *J_data) {
    reduce_reg_sobolev(no_voxels, threads, blocks, J_data, d_reg_out);
    cudaSafeCall(cudaGetLastError());

    return 0.5f * final_reduce(h_reg_out, d_reg_out, blocks);
}

float2 sobfu::device::Reductor::max_update_norm() {
    reduce_max(no_voxels, threads, blocks, updates, d_max_out);
    cudaSafeCall(cudaGetLastError());

    return final_reduce_max(h_max_out, d_max_out, blocks, dims);
}

float2 sobfu::device::Reductor::voxel_max_energy(float2 *phi_global_data, float2 *phi_n_data, Mat4f *J_data,
                                                 float w_reg) {
    reduce_voxel_max_energy(no_voxels, threads, blocks, phi_global_data, phi_n_data, J_data, w_reg, d_max_out);
    cudaSafeCall(cudaGetLastError());

    return final_reduce_max(h_max_out, d_max_out, blocks, dims);
}

/* sum partial sums from each block on cpu */
float sobfu::device::final_reduce(float *h_odata, float *d_odata, int numBlocks) {
    float result = 0.f;

    /* copy result from device to host */
    cudaSafeCall(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBlocks; i++) {
        result += h_odata[i];
    }

    return result;
}

float2 sobfu::device::final_reduce_max(float2 *h_o_max_data, float2 *d_o_max_data, int numBlocks, int3 dims) {
    float2 result = make_float2(0.f, 0.f);

    /* copy result from device to host */
    cudaSafeCall(cudaMemcpy(h_o_max_data, d_o_max_data, numBlocks * sizeof(float2), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numBlocks; i++) {
        if (h_o_max_data[i].x > result.x) {
            result = h_o_max_data[i];
        }
    }

    return result;
}
