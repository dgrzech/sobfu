#pragma once

/* sobfu includes */
#include <sobfu/params.hpp>
#include <sobfu/precomp.hpp>

/* kinfu includes */
#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>
#include <kfusion/types.hpp>

namespace sobfu {
namespace cuda {

/*
 * SCALAR FIELD
 */

class ScalarField {
public:
    /* constructor */
    ScalarField(cv::Vec3i dims_);
    /* destructor */
    ~ScalarField();

    /* get field data*/
    kfusion::cuda::CudaData get_data();
    const kfusion::cuda::CudaData get_data() const;

    /* get dims */
    int3 get_dims();

    /* clear field */
    void clear();
    /* sum values in the field */
    float sum();
    /* print field */
    void print();

private:
    kfusion::cuda::CudaData data; /* field data */
    int3 dims;                    /* field dimensions */
};

}  // namespace cuda

namespace device {

/*
 * SCALAR FIELD
 */

struct ScalarField {
    /* constructor */
    ScalarField(float *const data_, int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ float *beg(int x, int y) const;
    __device__ __forceinline__ float *zstep(float *const ptr) const;

    __device__ __forceinline__ float *operator()(int idx) const;
    __device__ __forceinline__ float *operator()(int x, int y, int z) const;

    float *const data; /* field data */
    const int3 dims;
};

/* clear */
__global__ void clear_kernel(sobfu::device::ScalarField field);
void clear(ScalarField &field);

/* sum */
float reduce_sum(sobfu::device::ScalarField &field);

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce_sum_kernel(float *g_idata, float *g_odata, unsigned int n);
float final_reduce_sum(float *d_odata, int numBlocks);

}  // namespace device
}  // namespace sobfu
