/* sobfu includes */
#include <sobfu/cuda/utils.hpp>
#include <sobfu/scalar_fields.hpp>

#define FULL_MASK 0xffffffff

/*
 * SCALAR FIELD
 */

__device__ __forceinline__ float* sobfu::device::ScalarField::beg(int x, int y) const { return data + y * dims.x + x; }

__device__ __forceinline__ float* sobfu::device::ScalarField::zstep(float* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ float* sobfu::device::ScalarField::operator()(int idx) const { return data + idx; }

__device__ __forceinline__ float* sobfu::device::ScalarField::operator()(int x, int y, int z) const {
    return data + z * dims.y * dims.x + y * dims.x + x;
}

__global__ void sobfu::device::clear_kernel(sobfu::device::ScalarField field) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > field.dims.x - 1 || y > field.dims.y - 1) {
        return;
    }

    float* pos = field.beg(x, y);
    for (int i = 0; i <= field.dims.z - 1; pos = field.zstep(pos), ++i) {
        *pos = 0.f;
    }
}

void sobfu::device::clear(sobfu::device::ScalarField& field) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(field.dims.x, block.x), kfusion::device::divUp(field.dims.y, block.y));

    clear_kernel<<<grid, block>>>(field);
    cudaSafeCall(cudaGetLastError());
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void sobfu::device::reduce_sum_kernel(float* g_idata, float* g_odata, unsigned int n) {
    float* sdata = SharedMemory<float>();

    /* perform first level of reduction, reading from global memory, writing to shared memory */
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float mySum = 0.f;

    /* we reduce multiple elements per thread;  the number is determined by the umber of active thread blocks (via
     * gridDim); more blocks will result in a larger gridSize and therefore fewer elements per thread */
    while (i < n) {
        mySum += g_idata[i];

        /* ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays */
        if (nIsPow2 || i + blockSize < n) {
            mySum += g_idata[i + blockSize];
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

float sobfu::device::final_reduce_sum(float* d_odata, int numBlocks) {
    /* copy result from device to host */
    float* h_odata = new float[numBlocks];
    cudaSafeCall(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float result = 0.f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_odata[i];
    }

    delete h_odata;
    return result;
}

float sobfu::device::reduce_sum(sobfu::device::ScalarField& field) {
    int no_voxels = field.dims.x * field.dims.y * field.dims.z;
    int blocks, threads;

    get_num_blocks_and_threads(no_voxels, 65536, 512, blocks, threads);

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    /* when there is only one warp per block, we need to allocate two warps worth of shared memory so that we don't
     * index shared memory out of bounds */
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    float* d_odata;
    cudaSafeCall(cudaMalloc((void**) &d_odata, blocks * sizeof(float)));

    if (isPow2(no_voxels)) {
        switch (threads) {
            case 512:
                reduce_sum_kernel<512, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 256:
                reduce_sum_kernel<256, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 128:
                reduce_sum_kernel<128, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 64:
                reduce_sum_kernel<64, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 32:
                reduce_sum_kernel<32, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 16:
                reduce_sum_kernel<16, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 8:
                reduce_sum_kernel<8, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 4:
                reduce_sum_kernel<4, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 2:
                reduce_sum_kernel<2, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 1:
                reduce_sum_kernel<1, true><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;
        }
    } else {
        switch (threads) {
            case 512:
                reduce_sum_kernel<512, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 256:
                reduce_sum_kernel<256, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 128:
                reduce_sum_kernel<128, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 64:
                reduce_sum_kernel<64, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 32:
                reduce_sum_kernel<32, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 16:
                reduce_sum_kernel<16, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 8:
                reduce_sum_kernel<8, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 4:
                reduce_sum_kernel<4, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 2:
                reduce_sum_kernel<2, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;

            case 1:
                reduce_sum_kernel<1, false><<<dimGrid, dimBlock, smemSize>>>(field.data, d_odata, no_voxels);
                break;
        }
    }

    float result = final_reduce_sum(d_odata, blocks);
    cudaSafeCall(cudaFree(d_odata));
    return result;
}
