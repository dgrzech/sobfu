/* sobfu includes */
#include <sobfu/cuda/utils.hpp>
#include <sobfu/reductor.hpp>

#define FULL_MASK 0xffffffff

/*
 * OWN KERNELS
 */

template <unsigned int blockSize, bool nIsPow2>
__global__ void sobfu::device::reduce_data_kernel(float2 *g_idata_global, float2 *g_idata_n, float *g_odata,
                                                  unsigned int n) {
    float *sdata = SharedMemory<float>();

    /* perform first level of reduction, reading from global memory, writing to shared memory */
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0.f;

    /* we reduce multiple elements per thread;  the number is determined by the umber of active thread blocks (via
     * gridDim); more blocks will result in a larger gridSize and therefore fewer elements per thread */
    while (i < n) {
        mySum += (g_idata_global[i].x - g_idata_n[i].x) * (g_idata_global[i].x - g_idata_n[i].x);

        /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */
        if (nIsPow2 || i + blockSize < n) {
            mySum += (g_idata_global[i + blockSize].x - g_idata_n[i + blockSize].x) *
                     (g_idata_global[i + blockSize].x - g_idata_n[i + blockSize].x);
        }

        i += gridSize;
    }

    /* each thread puts its local sum into shared memory */
    sdata[tid] = mySum;
    __syncthreads();

    /* do reduction in shared mem */
    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (tid < 32) {
        /* fetch final intermediate sum from 2nd warp */
        if (blockSize >= 64)
            mySum += sdata[tid + 32];
        /* reduce final warp using shuffle */
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            mySum += __shfl_down_sync(FULL_MASK, mySum, offset);
        }
    }
#else
    /* fully unroll reduction within a single warp */
    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = mySum = mySum + sdata[tid + 8];
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = mySum = mySum + sdata[tid + 4];
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = mySum = mySum + sdata[tid + 2];
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = mySum = mySum + sdata[tid + 1];
    }

    __syncthreads();
#endif

    /* write result for this block to global mem */
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void sobfu::device::reduce_reg_sobolev_kernel(Mat4f *g_idata, float *g_odata, unsigned int n) {
    float *sdata = SharedMemory<float>();

    /* perform first level of reduction, reading from global memory, writing to shared memory */
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0.f;

    /* we reduce multiple elements per thread;  the number is determined by the umber of active thread blocks (via
     * gridDim); more blocks will result in a larger gridSize and therefore fewer elements per thread */
    while (i < n) {
        mySum += norm_sq(g_idata[i].data[0]) + norm_sq(g_idata[i].data[1]) + norm_sq(g_idata[i].data[2]);

        /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */
        if (nIsPow2 || i + blockSize < n) {
            mySum += norm_sq(g_idata[i + blockSize].data[0]) + norm_sq(g_idata[i + blockSize].data[1]) +
                     norm_sq(g_idata[i + blockSize].data[2]);
        }

        i += gridSize;
    }

    /* each thread puts its local sum into shared memory */
    sdata[tid] = mySum;
    __syncthreads();

    /* do reduction in shared mem */
    if ((blockSize >= 512) && (tid < 256)) {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300)
    if (tid < 32) {
        /* fetch final intermediate sum from 2nd warp */
        if (blockSize >= 64)
            mySum += sdata[tid + 32];
        /* reduce final warp using shuffle */
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            mySum += __shfl_down_sync(FULL_MASK, mySum, offset);
        }
    }
#else
    /* fully unroll reduction within a single warp */
    if ((blockSize >= 64) && (tid < 32)) {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        sdata[tid] = mySum = mySum + sdata[tid + 8];
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        sdata[tid] = mySum = mySum + sdata[tid + 4];
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        sdata[tid] = mySum = mySum + sdata[tid + 2];
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        sdata[tid] = mySum = mySum + sdata[tid + 1];
    }

    __syncthreads();
#endif

    /* write result for this block to global mem */
    if (tid == 0)
        g_odata[blockIdx.x] = mySum;
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void sobfu::device::reduce_voxel_max_energy_kernel(float2 *d_idata_global, float2 *d_idata_n,
                                                              Mat4f *d_idata_reg, float2 *d_o_data, float w_reg,
                                                              unsigned int n) {
    float2 *sdata = SharedMemory<float2>();

    /* perform first level of reduction, reading from global memory, writing to shared memory */
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float2 local_max;
    local_max.x = 0.f;
    local_max.y = 0.f;

    /* we reduce multiple elements per thread;  the number is determined by the umber of active thread blocks (via
     * gridDim); more blocks will result in a larger gridSize and therefore fewer elements per thread */
    while (i < n) {
        float temp = (d_idata_global[i].x - d_idata_n[i].x) * (d_idata_global[i].x - d_idata_n[i].x) +
                     w_reg * (norm_sq(d_idata_reg[i].data[0]) + norm_sq(d_idata_reg[i].data[1]) +
                              norm_sq(d_idata_reg[i].data[2]));

        if (temp > local_max.x) {
            local_max.x = temp;
            local_max.y = (float) i;
        }

        /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */
        if (nIsPow2 || i + blockSize < n) {
            temp = (d_idata_global[i + blockSize].x - d_idata_n[i + blockSize].x) *
                       (d_idata_global[i + blockSize].x - d_idata_n[i + blockSize].x) +
                   w_reg * (norm_sq(d_idata_reg[i + blockSize].data[0]) + norm_sq(d_idata_reg[i + blockSize].data[1]) +
                            norm_sq(d_idata_reg[i + blockSize].data[2]));

            if (temp > local_max.x) {
                local_max.x = temp;
                local_max.y = (float) i + blockSize;
            }
        }

        i += gridSize;
    }

    /* each thread puts its local sum into shared memory */
    sdata[tid] = local_max;
    __syncthreads();

    /* do reduction in shared mem */
    if ((blockSize >= 512) && (tid < 256)) {
        if (sdata[tid + 256].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 256];
        }
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        if (sdata[tid + 128].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 128];
        }
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        if (sdata[tid + 64].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 64];
        }
    }

    __syncthreads();

    if ((blockSize >= 64) && (tid < 32)) {
        if (sdata[tid + 32].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 32];
        }
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        if (sdata[tid + 16].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 16];
        }
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        if (sdata[tid + 8].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 8];
        }
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        if (sdata[tid + 4].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 4];
        }
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        if (sdata[tid + 2].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 2];
        }
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        if (sdata[tid + 1].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 1];
        }
    }

    __syncthreads();

    /* write result for this block to global mem */
    if (tid == 0) {
        d_o_data[blockIdx.x] = local_max;
    }
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void sobfu::device::reduce_max_kernel(float4 *updates, float2 *g_o_max_data, unsigned int n) {
    float2 *sdata = SharedMemory<float2>();

    /* perform first level of reduction, reading from global memory, writing to shared memory */
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float2 local_max;
    local_max.x = 0.f;
    local_max.y = 0.f;

    /* we reduce multiple elements per thread;  the number is determined by the umber of active thread blocks (via
     * gridDim); more blocks will result in a larger gridSize and therefore fewer elements per thread */
    while (i < n) {
        if (norm(updates[i]) > local_max.x) {
            local_max.x = norm(updates[i]);
            local_max.y = (float) i;
        }

        /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */
        if (nIsPow2 || i + blockSize < n) {
            if (norm(updates[i + blockSize]) > local_max.x) {
                local_max.x = norm(updates[i + blockSize]);
                local_max.y = (float) i + blockSize;
            }
        }

        i += gridSize;
    }

    /* each thread puts its local sum into shared memory */
    sdata[tid] = local_max;
    __syncthreads();

    /* do reduction in shared mem */
    if ((blockSize >= 512) && (tid < 256)) {
        if (sdata[tid + 256].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 256];
        }
    }

    __syncthreads();

    if ((blockSize >= 256) && (tid < 128)) {
        if (sdata[tid + 128].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 128];
        }
    }

    __syncthreads();

    if ((blockSize >= 128) && (tid < 64)) {
        if (sdata[tid + 64].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 64];
        }
    }

    __syncthreads();

    if ((blockSize >= 64) && (tid < 32)) {
        if (sdata[tid + 32].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 32];
        }
    }

    __syncthreads();

    if ((blockSize >= 32) && (tid < 16)) {
        if (sdata[tid + 16].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 16];
        }
    }

    __syncthreads();

    if ((blockSize >= 16) && (tid < 8)) {
        if (sdata[tid + 8].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 8];
        }
    }

    __syncthreads();

    if ((blockSize >= 8) && (tid < 4)) {
        if (sdata[tid + 4].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 4];
        }
    }

    __syncthreads();

    if ((blockSize >= 4) && (tid < 2)) {
        if (sdata[tid + 2].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 2];
        }
    }

    __syncthreads();

    if ((blockSize >= 2) && (tid < 1)) {
        if (sdata[tid + 1].x > local_max.x) {
            sdata[tid] = local_max = sdata[tid + 1];
        }
    }

    __syncthreads();

    /* write result for this block to global mem */
    if (tid == 0) {
        g_o_max_data[blockIdx.x] = local_max;
    }
}

/* wrapper function for kernel launch */
void sobfu::device::reduce_data(int size, int threads, int blocks, float2 *d_idata_global, float2 *d_idata_n,
                                float *d_odata) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    /* when there is only one warp per block, we need to allocate two warps worth of shared memory so that we don't
     * index shared memory out of bounds */
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    if (isPow2(size)) {
        switch (threads) {
            case 512:
                reduce_data_kernel<512, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 256:
                reduce_data_kernel<256, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 128:
                reduce_data_kernel<128, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 64:
                reduce_data_kernel<64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 32:
                reduce_data_kernel<32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 16:
                reduce_data_kernel<16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 8:
                reduce_data_kernel<8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 4:
                reduce_data_kernel<4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 2:
                reduce_data_kernel<2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 1:
                reduce_data_kernel<1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;
        }
    } else {
        switch (threads) {
            case 512:
                reduce_data_kernel<512, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 256:
                reduce_data_kernel<256, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 128:
                reduce_data_kernel<128, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 64:
                reduce_data_kernel<64, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 32:
                reduce_data_kernel<32, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 16:
                reduce_data_kernel<16, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 8:
                reduce_data_kernel<8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 4:
                reduce_data_kernel<4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 2:
                reduce_data_kernel<2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;

            case 1:
                reduce_data_kernel<1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_odata, size);
                break;
        }
    }
}

void sobfu::device::reduce_reg_sobolev(int size, int threads, int blocks, Mat4f *d_idata, float *d_odata) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    /* when there is only one warp per block, we need to allocate two warps worth of shared memory so that we don't
     * index shared memory out of bounds */
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    if (isPow2(size)) {
        switch (threads) {
            case 512:
                reduce_reg_sobolev_kernel<512, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce_reg_sobolev_kernel<256, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce_reg_sobolev_kernel<128, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce_reg_sobolev_kernel<64, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce_reg_sobolev_kernel<32, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce_reg_sobolev_kernel<16, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce_reg_sobolev_kernel<8, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce_reg_sobolev_kernel<4, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce_reg_sobolev_kernel<2, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce_reg_sobolev_kernel<1, true><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
        }
    } else {
        switch (threads) {
            case 512:
                reduce_reg_sobolev_kernel<512, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce_reg_sobolev_kernel<256, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce_reg_sobolev_kernel<128, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce_reg_sobolev_kernel<64, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce_reg_sobolev_kernel<32, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce_reg_sobolev_kernel<16, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 8:
                reduce_reg_sobolev_kernel<8, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 4:
                reduce_reg_sobolev_kernel<4, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 2:
                reduce_reg_sobolev_kernel<2, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;

            case 1:
                reduce_reg_sobolev_kernel<1, false><<<dimGrid, dimBlock, smemSize>>>(d_idata, d_odata, size);
                break;
        }
    }
}

void sobfu::device::reduce_voxel_max_energy(int size, int threads, int blocks, float2 *d_idata_global,
                                            float2 *d_idata_n, Mat4f *d_idata_reg, float w_reg, float2 *d_odata) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    /* when there is only one warp per block, we need to allocate two warps worth of shared memory so that we don't
     * index shared memory out of bounds */
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float2) : threads * sizeof(float2);

    if (isPow2(size)) {
        switch (threads) {
            case 512:
                reduce_voxel_max_energy_kernel<512, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 256:
                reduce_voxel_max_energy_kernel<256, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 128:
                reduce_voxel_max_energy_kernel<128, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 64:
                reduce_voxel_max_energy_kernel<64, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 32:
                reduce_voxel_max_energy_kernel<32, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 16:
                reduce_voxel_max_energy_kernel<16, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 8:
                reduce_voxel_max_energy_kernel<8, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 4:
                reduce_voxel_max_energy_kernel<4, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 2:
                reduce_voxel_max_energy_kernel<2, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 1:
                reduce_voxel_max_energy_kernel<1, true>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;
        }
    } else {
        switch (threads) {
            case 512:
                reduce_voxel_max_energy_kernel<512, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 256:
                reduce_voxel_max_energy_kernel<256, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 128:
                reduce_voxel_max_energy_kernel<128, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 64:
                reduce_voxel_max_energy_kernel<64, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 32:
                reduce_voxel_max_energy_kernel<32, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 16:
                reduce_voxel_max_energy_kernel<16, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 8:
                reduce_voxel_max_energy_kernel<8, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 4:
                reduce_voxel_max_energy_kernel<4, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 2:
                reduce_voxel_max_energy_kernel<2, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;

            case 1:
                reduce_voxel_max_energy_kernel<1, false>
                    <<<dimGrid, dimBlock, smemSize>>>(d_idata_global, d_idata_n, d_idata_reg, d_odata, w_reg, size);
                break;
        }
    }
}

void sobfu::device::reduce_max(int size, int threads, int blocks, float4 *updates, float2 *d_o_max_data) {
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    /* when there is only one warp per block, we need to allocate two warps worth of shared memory so that we don't
     * index shared memory out of bounds */
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float2) : threads * sizeof(float2);

    if (isPow2(size)) {
        switch (threads) {
            case 512:
                reduce_max_kernel<512, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 256:
                reduce_max_kernel<256, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 128:
                reduce_max_kernel<128, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 64:
                reduce_max_kernel<64, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 32:
                reduce_max_kernel<32, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 16:
                reduce_max_kernel<16, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 8:
                reduce_max_kernel<8, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 4:
                reduce_max_kernel<4, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 2:
                reduce_max_kernel<2, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 1:
                reduce_max_kernel<1, true><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;
        }
    } else {
        switch (threads) {
            case 512:
                reduce_max_kernel<512, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 256:
                reduce_max_kernel<256, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 128:
                reduce_max_kernel<128, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 64:
                reduce_max_kernel<64, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 32:
                reduce_max_kernel<32, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 16:
                reduce_max_kernel<16, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 8:
                reduce_max_kernel<8, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 4:
                reduce_max_kernel<4, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 2:
                reduce_max_kernel<2, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;

            case 1:
                reduce_max_kernel<1, false><<<dimGrid, dimBlock, smemSize>>>(updates, d_o_max_data, size);
                break;
        }
    }
}
