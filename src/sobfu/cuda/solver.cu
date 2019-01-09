/* sobfu includes */
#include <sobfu/cuda/utils.hpp>
#include <sobfu/solver.hpp>

/* kinfu includes */
#include <kfusion/cuda/device.hpp>

using namespace kfusion;
using namespace kfusion::device;

/*
 * POTENTIAL GRADIENT
 */

__global__ void sobfu::device::calculate_potential_gradient_kernel(float2* phi_n_psi, float2* phi_global,
                                                                   float4* nabla_phi_n_o_psi, float4* L,
                                                                   float4* nabla_U, float w_reg, int dim_x, int dim_y,
                                                                   int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > dim_x - 1 || y > dim_y - 1) {
        return;
    }

    int idx = y * dim_x + x;
    for (int i = 0; i <= dim_z - 1; idx += dim_y * dim_x, ++i) {
        float tsdf_n_psi  = phi_n_psi[idx].x;
        float tsdf_global = phi_global[idx].x;

        nabla_U[idx] = (tsdf_n_psi - tsdf_global) * nabla_phi_n_o_psi[idx] + w_reg * L[idx];
    }
}

void sobfu::device::calculate_potential_gradient(kfusion::device::TsdfVolume& phi_n_psi,
                                                 kfusion::device::TsdfVolume& phi_global,
                                                 sobfu::device::TsdfGradient& nabla_phi_n_o_psi,
                                                 sobfu::device::Laplacian& L, sobfu::device::PotentialGradient& nabla_U,
                                                 float w_reg) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(phi_n_psi.dims.x, block.x), kfusion::device::divUp(phi_n_psi.dims.y, block.y));

    calculate_potential_gradient_kernel<<<grid, block>>>(phi_n_psi.data, phi_global.data, nabla_phi_n_o_psi.data,
                                                         L.data, nabla_U.data, w_reg, phi_n_psi.dims.x,
                                                         phi_n_psi.dims.y, phi_n_psi.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * DEFORMATION FIELD
 */

__global__ void sobfu::device::update_psi_kernel(float4* psi, float4* nabla_U_S, float4* updates, float alpha,
                                                 int dim_x, int dim_y, int dim_z) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > dim_x - 1 || y > dim_y - 1) {
        return;
    }

    int global_idx = y * dim_x + x;
    for (int i = 0; i <= dim_z - 1; global_idx += dim_y * dim_x, ++i) {
        float4 update = alpha * nabla_U_S[global_idx];

        updates[global_idx] = update;
        psi[global_idx] -= update;
    }
}

void sobfu::device::update_psi(sobfu::device::DeformationField& psi, sobfu::device::PotentialGradient& nabla_U_S,
                               float4* updates, float alpha) {
    /* integrate in time */
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(psi.dims.x, block.x), kfusion::device::divUp(psi.dims.y, block.y));

    update_psi_kernel<<<grid, block>>>(psi.data, nabla_U_S.data, updates, alpha, psi.dims.x, psi.dims.y, psi.dims.z);
    cudaSafeCall(cudaGetLastError());
}

/*
 * PIPELINE
 */

void sobfu::device::estimate_psi(SDFs& sdfs, sobfu::device::DeformationField& psi,
                                 sobfu::device::DeformationField& psi_inv,
                                 sobfu::device::SpatialGradients* spatial_grads, Differentiators& differentiators,
                                 float* d_S_i, sobfu::device::Reductor* r, SolverParams& params) {
    /* copy sobolev filter to constant memory */
    set_convolution_kernel(d_S_i);

    /* create cuda streams */
    int no_streams = 3;
    cudaStream_t streams[no_streams];
    for (int i = 0; i < no_streams; i++) {
        cudaSafeCall(cudaStreamCreate(&streams[i]));
    }

    /* calculate no. of blocks and no. of threads per block */
    int3 dims = psi.dims;

    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(dims.x, block.x), kfusion::device::divUp(dims.y, block.y));

    /* apply psi to phi_n */
    apply_kernel<<<grid, block, 0, streams[0]>>>(sdfs.phi_n, sdfs.phi_n_psi, psi);
    cudaSafeCall(cudaGetLastError());

    /* run gradient descent */
    float2 curr_max_update_norm;
    float e_curr = std::numeric_limits<float>::infinity();

    int iter = 1;
    while (iter <= params.max_iter) {
        if (iter == 1 || iter % 50 == 0) {
            std::cout << "iter. no. " << iter << std::endl;
        }

        /* calculate the gradient of phi_n */
        estimate_gradient_kernel<<<grid, block, 0, streams[0]>>>(differentiators.tsdf_diff,
                                                                 *(spatial_grads->nabla_phi_n_o_psi));
        cudaSafeCall(cudaGetLastError());
        /* calculate the jacobian of psi */
        estimate_deformation_jacobian_kernel<<<grid, block, 0, streams[1]>>>(differentiators.diff, *(spatial_grads->J));
        cudaSafeCall(cudaGetLastError());
        /* calculate the laplacian of psi */
        estimate_laplacian_kernel<<<grid, block, 0, streams[2]>>>(differentiators.second_order_diff,
                                                                  *(spatial_grads->L));
        cudaSafeCall(cudaGetLastError());

        /* calculate current value of the energy functional */
        if ((params.verbosity == 1 && (iter == 1 || iter % 50 == 0 || iter == params.max_iter) ||
             params.verbosity == 2)) {
            /* data term */
            float e_data = r->data_energy(sdfs.phi_global.data, sdfs.phi_n_psi.data);
            /* regularisation term */
            float e_reg = r->reg_energy_sobolev(spatial_grads->J->data);

            e_curr = e_data + params.w_reg * e_reg;
            std::cout << "data energy + w_reg * reg energy = " << e_data << " + " << params.w_reg << " * " << e_reg
                      << " = " << e_curr << std::endl;
        }

        /*
         * PDE'S
         */

        /* calculate gradient of the potential */
        sobfu::device::calculate_potential_gradient(sdfs.phi_n_psi, sdfs.phi_global,
                                                    *(spatial_grads->nabla_phi_n_o_psi), *(spatial_grads->L),
                                                    *(spatial_grads->nabla_U), params.w_reg);
        cudaSafeCall(cudaGetLastError());

        /* convolve gradient of the potential with a sobolev kernel */
        sobfu::device::convolution_rows((*(spatial_grads->nabla_U_S)).data, (*(spatial_grads->nabla_U)).data, dims.x,
                                        dims.y, dims.z);
        sobfu::device::convolution_columns((*(spatial_grads->nabla_U_S)).data, (*(spatial_grads->nabla_U)).data, dims.x,
                                           dims.y, dims.z);
        sobfu::device::convolution_depth((*(spatial_grads->nabla_U_S)).data, (*(spatial_grads->nabla_U)).data, dims.x,
                                         dims.y, dims.z);

        /* update psi */
        update_psi_kernel<<<grid, block, 0, streams[0]>>>(psi.data, (*(spatial_grads->nabla_U_S)).data, r->updates,
                                                          params.alpha, dims.x, dims.y, dims.z);
        cudaSafeCall(cudaGetLastError());

        /* apply psi to phi_n */
        apply_kernel<<<grid, block, 0, streams[0]>>>(sdfs.phi_n, sdfs.phi_n_psi, psi);
        cudaSafeCall(cudaGetLastError());

        /* get value of the max. update norm at the current iteration of the solver */
        curr_max_update_norm = r->max_update_norm();
        if ((params.verbosity == 1 && (iter == 1 || iter % 50 == 0 || iter == params.max_iter) ||
             params.verbosity == 2)) {
            int idx_x = curr_max_update_norm.y / (psi.dims.x * psi.dims.y);
            int idx_y = (curr_max_update_norm.y - idx_x * psi.dims.x * psi.dims.y) / psi.dims.x;
            int idx_z = curr_max_update_norm.y - psi.dims.x * (idx_y + psi.dims.y * idx_x);

            std::cout << "max. update norm " << curr_max_update_norm.x << " at voxel (" << idx_z << ", " << idx_y
                      << ", " << idx_x << ")" << std::endl;
        }

        if (curr_max_update_norm.x <= params.max_update_norm) {
            std::cout << "SOLVER CONVERGED AFTER " << iter << " ITERATIONS" << std::endl;
            break;
        }

        if (iter == params.max_iter) {
            std::cout << "SOLVER REACHED MAX. NO. OF ITERATIONS WITHOUT CONVERGING" << std::endl;
        }

        iter++;
    }

    /* iteratively estimate the inverse deformation field */
    sobfu::device::init_identity(psi_inv);
    sobfu::device::estimate_inverse(psi, psi_inv);
    /* apply psi_inv to phi_global */
    apply_kernel<<<grid, block>>>(sdfs.phi_global, sdfs.phi_global_psi_inv, psi_inv);
    cudaSafeCall(cudaGetLastError());

    for (int i = 0; i < no_streams; i++) {
        cudaSafeCall(cudaStreamDestroy(streams[i]));
    }
}

/*
 * CONVOLUTIONS
 */

#define KERNEL_RADIUS 3
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

#define ROWS_BLOCKDIM_X 4
#define ROWS_BLOCKDIM_Y 64
#define ROWS_RESULT_STEPS 4
#define ROWS_HALO_STEPS 1

#define COLUMNS_BLOCKDIM_X 64
#define COLUMNS_BLOCKDIM_Y 4
#define COLUMNS_RESULT_STEPS 4
#define COLUMNS_HALO_STEPS 1

#define DEPTH_BLOCKDIM_X 64
#define DEPTH_BLOCKDIM_Z 4
#define DEPTH_RESULT_STEPS 4
#define DEPTH_HALO_STEPS 1

__constant__ float S[KERNEL_LENGTH];

void sobfu::device::set_convolution_kernel(float* d_kernel) {
    cudaSafeCall(cudaMemcpyToSymbol(S, d_kernel, KERNEL_LENGTH * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaGetLastError());
}

/*** ROW CONVOLUTION ***/
__global__ void sobfu::device::convolution_rows_kernel(float4* d_dst, float4* d_src, int image_w, int image_h,
                                                       int image_d) {
    __shared__ float4 s_data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    /* offset to the left halo edge */
    const int base_x = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int base_y = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;
    const int base_z = blockIdx.z;

    const int first_pixel_in_line = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS - threadIdx.x;
    const int last_pixel_in_line  = image_w - base_x - 1;

    d_dst += base_z * image_h * image_w + base_y * image_w + base_x;
    d_src += base_z * image_h * image_w + base_y * image_w + base_x;

    /* load main data */
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (image_w - base_x > i * ROWS_BLOCKDIM_X) ? d_src[i * ROWS_BLOCKDIM_X] : d_src[last_pixel_in_line];
    }

    /* load left halo */
#pragma unroll
    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (base_x >= -i * ROWS_BLOCKDIM_X) ? d_src[i * ROWS_BLOCKDIM_X] : d_src[first_pixel_in_line];
    }

    /* load right halo */
#pragma unroll
    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (image_w - base_x > i * ROWS_BLOCKDIM_X) ? d_src[i * ROWS_BLOCKDIM_X] : d_src[last_pixel_in_line];
    }

    /* compute and store results */
    __syncthreads();

    /* this pixel is not part of the iamge and doesn't need to be convolved */
    if (base_y >= image_h) {
        return;
    }
#pragma unroll
    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        if (image_w - base_x > i * ROWS_BLOCKDIM_X) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                sum += S[KERNEL_RADIUS - j] * s_data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
            }

            d_dst[i * ROWS_BLOCKDIM_X] = sum;
        }
    }
}

void sobfu::device::convolution_rows(float4* d_dst, float4* d_src, int image_w, int image_h, int image_d) {
    int blocks_x =
        image_w / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) + min(1, image_w % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X));
    int blocks_y = image_h / ROWS_BLOCKDIM_Y + min(1, image_h % ROWS_BLOCKDIM_Y);
    int blocks_z = image_d;

    dim3 blocks(blocks_x, blocks_y, blocks_z);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1);

    convolution_rows_kernel<<<blocks, threads>>>(d_dst, d_src, image_w, image_h, image_d);
    cudaSafeCall(cudaGetLastError());
}

/*** COLUMMN CONVOLUTION ***/
__global__ void sobfu::device::convolution_columns_kernel(float4* d_dst, float4* d_src, int image_w, int image_h,
                                                          int image_d) {
    __shared__ float4
        s_data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    /* offset to the upper halo edge */
    const int base_x = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int base_y = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    const int base_z = blockIdx.z;

    const int first_pixel_in_line = (COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS - threadIdx.y) * image_w;
    const int last_pixel_in_line  = (image_h - base_y - 1) * image_w;

    d_dst += base_z * image_h * image_w + base_y * image_w + base_x;
    d_src += base_z * image_h * image_w + base_y * image_w + base_x;

    /* main data */
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (image_h - base_y > i * COLUMNS_BLOCKDIM_Y)
                                                                        ? d_src[i * COLUMNS_BLOCKDIM_Y * image_w]
                                                                        : d_src[last_pixel_in_line];
    }

    /* upper halo */
#pragma unroll
    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (base_y >= -i * COLUMNS_BLOCKDIM_Y) ? d_src[i * COLUMNS_BLOCKDIM_Y * image_w] : d_src[first_pixel_in_line];
    }

    /* lower halo */
#pragma unroll
    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
         i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (image_h - base_y > i * COLUMNS_BLOCKDIM_Y)
                                                                        ? d_src[i * COLUMNS_BLOCKDIM_Y * image_w]
                                                                        : d_src[last_pixel_in_line];
    }

    /* compute and store results */
    __syncthreads();

    /* this pixel isn't part of hte image and doesn't need to be convolved */
    if (base_x >= image_w) {
        return;
    }

#pragma unroll
    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        if (image_h - base_y > i * COLUMNS_BLOCKDIM_Y) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                sum += S[KERNEL_RADIUS - j] * s_data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
            }

            d_dst[i * COLUMNS_BLOCKDIM_Y * image_w] += sum;
        }
    }
}

void sobfu::device::convolution_columns(float4* d_dst, float4* d_src, int image_w, int image_h, int image_d) {
    int blocks_x = image_w / COLUMNS_BLOCKDIM_X + min(1, image_w % COLUMNS_BLOCKDIM_X);
    int blocks_y = image_h / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) +
                   min(1, image_h % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    int blocks_z = image_d;

    dim3 blocks(blocks_x, blocks_y, blocks_z);
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1);

    convolution_columns_kernel<<<blocks, threads>>>(d_dst, d_src, image_w, image_h, image_d);
    cudaSafeCall(cudaGetLastError());
}

/*** DEPTH CONVOLUTION ***/
__global__ void sobfu::device::convolution_depth_kernel(float4* d_dst, float4* d_src, int image_w, int image_h,
                                                        int image_d) {
    /* here it is [x][z] as we leave out y bc it has a size of 1 */
    __shared__ float4 s_data[DEPTH_BLOCKDIM_X][(DEPTH_RESULT_STEPS + 2 * DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + 1];

    /* offset to the upper halo edge */
    const int base_x = blockIdx.x * DEPTH_BLOCKDIM_X + threadIdx.x;
    const int base_y = blockIdx.y;
    const int base_z = (blockIdx.z * DEPTH_RESULT_STEPS - DEPTH_HALO_STEPS) * DEPTH_BLOCKDIM_Z + threadIdx.z;

    const int first_pixel_in_line = (DEPTH_BLOCKDIM_Z * DEPTH_HALO_STEPS - threadIdx.z) * image_w * image_h;
    const int last_pixel_in_line  = (image_d - base_z - 1) * image_w * image_h;

    d_dst += base_z * image_h * image_w + base_y * image_w + base_x;
    d_src += base_z * image_h * image_w + base_y * image_w + base_x;

    /* main data */
#pragma unroll
    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (image_d - base_z > i * DEPTH_BLOCKDIM_Z)
                                                                      ? d_src[i * DEPTH_BLOCKDIM_Z * image_w * image_h]
                                                                      : d_src[last_pixel_in_line];
    }

    /* upper halo */
#pragma unroll
    for (int i = 0; i < DEPTH_HALO_STEPS; i++) {
        s_data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (base_z >= -i * DEPTH_BLOCKDIM_Z)
                                                                      ? d_src[i * DEPTH_BLOCKDIM_Z * image_w * image_h]
                                                                      : d_src[first_pixel_in_line];
    }

    /* lower halo */
#pragma unroll
    for (int i = DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS + DEPTH_HALO_STEPS;
         i++) {
        s_data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z] = (image_d - base_z > i * DEPTH_BLOCKDIM_Z)
                                                                      ? d_src[i * DEPTH_BLOCKDIM_Z * image_w * image_h]
                                                                      : d_src[last_pixel_in_line];
    }

    /* compute and store results */
    __syncthreads();

    /* this pixel is not part of the image and doesn't need to be convolved */
    if (base_x >= image_w) {
        return;
    }

#pragma unroll
    for (int i = DEPTH_HALO_STEPS; i < DEPTH_HALO_STEPS + DEPTH_RESULT_STEPS; i++) {
        if (image_d - base_z > i * DEPTH_BLOCKDIM_Z) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                sum += S[KERNEL_RADIUS - j] * s_data[threadIdx.x][threadIdx.z + i * DEPTH_BLOCKDIM_Z + j];
            }

            d_dst[i * DEPTH_BLOCKDIM_Z * image_w * image_h] += sum;
        }
    }
}

void sobfu::device::convolution_depth(float4* d_dst, float4* d_src, int image_w, int image_h, int image_d) {
    int blocks_x = image_w / DEPTH_BLOCKDIM_X + min(1, image_w % DEPTH_BLOCKDIM_X);
    int blocks_y = image_h;
    int blocks_z =
        image_d / (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z) + min(1, image_d % (DEPTH_RESULT_STEPS * DEPTH_BLOCKDIM_Z));

    dim3 blocks(blocks_x, blocks_y, blocks_z);
    dim3 threads(DEPTH_BLOCKDIM_X, 1, DEPTH_BLOCKDIM_Z);

    convolution_depth_kernel<<<blocks, threads>>>(d_dst, d_src, image_w, image_h, image_d);
    cudaSafeCall(cudaGetLastError());
}
