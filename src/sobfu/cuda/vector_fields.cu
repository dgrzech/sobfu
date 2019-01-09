/* sobfu includes */
#include <sobfu/cuda/utils.hpp>
#include <sobfu/vector_fields.hpp>

/* kinfu includes */
#include <kfusion/cuda/device.hpp>

using namespace kfusion::device;

/*
 * VECTOR FIELD
 */

__device__ __forceinline__ float4* sobfu::device::VectorField::beg(int x, int y) const { return data + x + dims.x * y; }

__device__ __forceinline__ float4* sobfu::device::VectorField::zstep(float4* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ float4* sobfu::device::VectorField::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

__device__ __forceinline__ float4 sobfu::device::VectorField::get_displacement(int x, int y, int z) const {
    return *(data + z * dims.y * dims.x + y * dims.x + x) - make_float4((float) x, (float) y, (float) z, 0.f);
}

void sobfu::device::clear(VectorField& field) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(field.dims.x, block.x), kfusion::device::divUp(field.dims.y, block.y));

    clear_kernel<<<grid, block>>>(field);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sobfu::device::clear_kernel(sobfu::device::VectorField field) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > field.dims.x - 1 || y > field.dims.y - 1) {
        return;
    }

    float4* beg = field.beg(x, y);
    float4* end = beg + field.dims.x * field.dims.y * field.dims.z;

    for (float4* pos = beg; pos != end; pos = field.zstep(pos)) {
        *pos = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

/*
 * DEFORMATION FIELD
 */

void sobfu::device::init_identity(sobfu::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(psi.dims.x, block.x), kfusion::device::divUp(psi.dims.y, block.y));

    init_identity_kernel<<<grid, block>>>(psi);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sobfu::device::init_identity_kernel(sobfu::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float4 idx   = make_float4((float) x, (float) y, 0.f, 0.f);
    float4 zstep = make_float4(0.f, 0.f, 1.f, 0.f);

    float4* pos = psi.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; idx += zstep, pos = psi.zstep(pos), ++i) {
        *pos = idx;
    }
}

__global__ void sobfu::device::apply_kernel(const kfusion::device::TsdfVolume phi,
                                            kfusion::device::TsdfVolume phi_warped,
                                            const sobfu::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > phi_warped.dims.x - 1 || y > phi_warped.dims.y - 1) {
        return;
    }

    float4* psi_ptr        = psi.beg(x, y);
    float2* phi_warped_ptr = phi_warped.beg(x, y);
    for (int i   = 0; i <= phi_warped.dims.z - 1;
         psi_ptr = psi.zstep(psi_ptr), phi_warped_ptr = phi_warped.zstep(phi_warped_ptr), ++i) {
        float4 psi_val = *psi_ptr;

        float2 tsdf_deformed = interpolate_tsdf(phi, trunc(psi_val));
        *phi_warped_ptr      = tsdf_deformed;
    }
}

void sobfu::device::apply(const kfusion::device::TsdfVolume& phi, kfusion::device::TsdfVolume& phi_warped,
                          const sobfu::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(phi.dims.x, block.x), kfusion::device::divUp(phi.dims.y, block.y));

    apply_kernel<<<grid, block>>>(phi, phi_warped, psi);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sobfu::device::estimate_inverse_kernel(sobfu::device::DeformationField psi,
                                                       sobfu::device::DeformationField psi_inv) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi_inv.dims.x - 1 || y > psi_inv.dims.y - 1) {
        return;
    }

    float4* psi_inv_ptr = psi_inv.beg(x, y);
    for (int i = 0; i <= psi_inv.dims.z - 1; psi_inv_ptr = psi_inv.zstep(psi_inv_ptr), ++i) {
        float4 psi_inv_val = *psi_inv_ptr;
        *psi_inv_ptr =
            make_float4((float) x, (float) y, (float) i, 0.f) - 1.f * interpolate_field_inv(psi, trunc(psi_inv_val));
    }
}

void sobfu::device::estimate_inverse(sobfu::device::DeformationField& psi,
                                     sobfu::device::DeformationField& psi_inverse) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(psi_inverse.dims.x, block.x), kfusion::device::divUp(psi_inverse.dims.y, block.y));

    /* estimate inverse */
    for (int iter = 0; iter < 48; ++iter) {
        estimate_inverse_kernel<<<grid, block>>>(psi, psi_inverse);
        cudaSafeCall(cudaGetLastError());
    }
}

/*
 * TSDF DIFFERENTIATOR METHODS
 */

__global__ void sobfu::device::estimate_gradient_kernel(const sobfu::device::TsdfDifferentiator diff,
                                                        sobfu::device::TsdfGradient grad) {
    diff(grad);
}

void sobfu::device::TsdfDifferentiator::calculate(sobfu::device::TsdfGradient& grad) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(grad.dims.x, block.x), kfusion::device::divUp(grad.dims.y, block.y));

    estimate_gradient_kernel<<<grid, block>>>(*this, grad);
    cudaSafeCall(cudaGetLastError());
}

__device__ __forceinline__ void sobfu::device::TsdfDifferentiator::operator()(sobfu::device::TsdfGradient& grad) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > vol.dims.x - 1 || y > vol.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;
    if (x == 0) {
        idx_x_2 = x + 1;
    } else if (x == vol.dims.x - 1) {
        idx_x_1 = x - 1;
    }

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;
    if (y == 0) {
        idx_y_2 = y + 1;
    } else if (y == vol.dims.y - 1) {
        idx_y_1 = y - 1;
    }

    float4* grad_ptr = grad.beg(x, y);

#pragma unroll
    for (int i = 0; i <= vol.dims.z - 1; grad_ptr = grad.zstep(grad_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;
        if (i == 0) {
            idx_z_2 = i + 1;
        } else if (i == vol.dims.z - 1) {
            idx_z_1 = i - 1;
        }

        float Fx1 = (*vol(idx_x_1, y, i)).x;
        float Fx2 = (*vol(idx_x_2, y, i)).x;
        float n_x = __fdividef(Fx1 - Fx2, 2.f);

        float Fy1 = (*vol(x, idx_y_1, i)).x;
        float Fy2 = (*vol(x, idx_y_2, i)).x;
        float n_y = __fdividef(Fy1 - Fy2, 2.f);

        float Fz1 = (*vol(x, y, idx_z_1)).x;
        float Fz2 = (*vol(x, y, idx_z_2)).x;
        float n_z = __fdividef(Fz1 - Fz2, 2.f);

        float4 n  = make_float4(n_x, n_y, n_z, 0.f);
        *grad_ptr = n;
    }
}

__global__ void sobfu::device::interpolate_gradient_kernel(sobfu::device::TsdfGradient nabla_phi_n_psi,
                                                           sobfu::device::TsdfGradient nabla_phi_n_psi_t,
                                                           sobfu::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float3 idx   = make_float3(x, y, 0.f);
    float3 zstep = make_float3(0.f, 0.f, 1.f);

    int global_idx = y * nabla_phi_n_psi.dims.x + x;

    float4* nabla_phi_n_psi_t_ptr = nabla_phi_n_psi_t.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; nabla_phi_n_psi_t_ptr = nabla_phi_n_psi_t.zstep(nabla_phi_n_psi_t_ptr),
             global_idx += nabla_phi_n_psi.dims.x * nabla_phi_n_psi.dims.y, idx += zstep, ++i) {
        float4 psi_val         = psi.data[global_idx];
        *nabla_phi_n_psi_t_ptr = interpolate_field(nabla_phi_n_psi, trunc(psi_val));
    }
}

void sobfu::device::interpolate_gradient(sobfu::device::TsdfGradient& nabla_phi_n_psi,
                                         sobfu::device::TsdfGradient& nabla_phi_n_psi_t,
                                         sobfu::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(psi.dims.x, block.x), kfusion::device::divUp(psi.dims.y, block.y));

    interpolate_gradient_kernel<<<grid, block>>>(nabla_phi_n_psi, nabla_phi_n_psi_t, psi);
    cudaSafeCall(cudaGetLastError());
}

/*
 * LAPLACIAN
 */

__global__ void sobfu::device::interpolate_laplacian_kernel(sobfu::device::Laplacian L,
                                                            sobfu::device::Laplacian L_o_psi,
                                                            sobfu::device::DeformationField psi) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    float4* psi_ptr     = psi.beg(x, y);
    float4* L_o_psi_ptr = L_o_psi.beg(x, y);
    for (int i = 0; i <= psi.dims.z - 1; psi_ptr = psi.zstep(psi_ptr), L_o_psi_ptr = L_o_psi.zstep(L_o_psi_ptr), ++i) {
        float4 psi_val = *psi_ptr;
        *L_o_psi_ptr   = interpolate_field(L, trunc(psi_val));
    }
}

void sobfu::device::interpolate_laplacian(sobfu::device::Laplacian& L, sobfu::device::Laplacian& L_o_psi,
                                          sobfu::device::DeformationField& psi) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(psi.dims.x, block.x), kfusion::device::divUp(psi.dims.y, block.y));

    interpolate_laplacian_kernel<<<grid, block>>>(L, L_o_psi, psi);
    cudaSafeCall(cudaGetLastError());
}

/*
 * SECOND ORDER DIFFERENTIATOR METHODS
 */

void sobfu::device::SecondOrderDifferentiator::calculate(sobfu::device::Laplacian& L) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(L.dims.x, block.x), kfusion::device::divUp(L.dims.y, block.y));

    estimate_laplacian_kernel<<<grid, block>>>(*this, L);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sobfu::device::estimate_laplacian_kernel(const sobfu::device::SecondOrderDifferentiator diff,
                                                         sobfu::device::Laplacian L) {
    diff.laplacian(L);
}

__device__ __forceinline__ void sobfu::device::SecondOrderDifferentiator::laplacian(sobfu::device::Laplacian& L) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > L.dims.x - 1 || y > L.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;
    if (x == 0) {
        idx_x_1 = x;
        idx_x_2 = x;
    } else if (x == L.dims.x - 1) {
        idx_x_1 = x;
        idx_x_2 = x;
    }

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;
    if (y == 0) {
        idx_y_1 = y;
        idx_y_2 = y;
    } else if (y == L.dims.y - 1) {
        idx_y_1 = y;
        idx_y_2 = y;
    }

    float4* L_ptr = L.beg(x, y);

#pragma unroll
    for (int i = 0; i <= L.dims.z - 1; L_ptr = L.zstep(L_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;
        if (i == 0) {
            idx_z_1 = i;
            idx_z_2 = i;
        } else if (i == L.dims.z - 1) {
            idx_z_1 = i;
            idx_z_2 = i;
        }

        float4 L_val = -6.f * *psi(x, y, i) + *psi(idx_x_1, y, i) + *psi(idx_x_2, y, i) + *psi(x, idx_y_1, i) +
                       *psi(x, idx_y_2, i) + *psi(x, y, idx_z_1) + *psi(x, y, idx_z_2);
        *L_ptr = -1.f * L_val;
    }
}

/*
 * JACOBIAN
 */

__device__ __forceinline__ Mat4f* sobfu::device::Jacobian::beg(int x, int y) const { return data + x + dims.x * y; }

__device__ __forceinline__ Mat4f* sobfu::device::Jacobian::zstep(Mat4f* const ptr) const {
    return ptr + dims.x * dims.y;
}

__device__ __forceinline__ Mat4f* sobfu::device::Jacobian::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

void sobfu::device::clear(Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(J.dims.x, block.x), kfusion::device::divUp(J.dims.y, block.y));

    clear_jacobian_kernel<<<grid, block>>>(J);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void sobfu::device::clear_jacobian_kernel(sobfu::device::Jacobian J) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > J.dims.x - 1 || y > J.dims.y - 1) {
        return;
    }

    Mat4f* beg = J.beg(x, y);
    Mat4f* end = beg + J.dims.x * J.dims.y * J.dims.z;

    for (Mat4f* pos = beg; pos != end; pos = J.zstep(pos)) {
        float4 g = make_float4(0.f, 0.f, 0.f, 0.f);

        Mat4f val;
        val.data[0] = g;
        val.data[1] = g;
        val.data[2] = g;

        *pos = val;
    }
}

/*
 * DIFFERENTIATOR METHODS
 */

void sobfu::device::Differentiator::calculate(sobfu::device::Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(J.dims.x, block.x), kfusion::device::divUp(J.dims.y, block.y));

    estimate_jacobian_kernel<<<grid, block>>>(*this, J);
    cudaSafeCall(cudaGetLastError());
}

void sobfu::device::Differentiator::calculate_deformation_jacobian(sobfu::device::Jacobian& J) {
    dim3 block(64, 16);
    dim3 grid(kfusion::device::divUp(J.dims.x, block.x), kfusion::device::divUp(J.dims.y, block.y));

    estimate_deformation_jacobian_kernel<<<grid, block>>>(*this, J);
    cudaSafeCall(cudaGetLastError());
}

__global__ void sobfu::device::estimate_jacobian_kernel(const sobfu::device::Differentiator diff,
                                                        sobfu::device::Jacobian J) {
    diff(J, 0);
}

__global__ void sobfu::device::estimate_deformation_jacobian_kernel(const sobfu::device::Differentiator diff,
                                                                    sobfu::device::Jacobian J) {
    diff(J, 1);
}

__device__ __forceinline__ void sobfu::device::Differentiator::operator()(sobfu::device::Jacobian& J, int mode) const {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > psi.dims.x - 1 || y > psi.dims.y - 1) {
        return;
    }

    int idx_x_1 = x + 1;
    int idx_x_2 = x - 1;
    if (x == 0) {
        idx_x_2 = x + 1;
    } else if (x == psi.dims.x - 1) {
        idx_x_1 = x - 1;
    }

    int idx_y_1 = y + 1;
    int idx_y_2 = y - 1;
    if (y == 0) {
        idx_y_2 = y + 1;
    } else if (y == psi.dims.y - 1) {
        idx_y_1 = y - 1;
    }

    Mat4f* J_ptr = J.beg(x, y);

#pragma unroll
    for (int i = 0; i <= psi.dims.z - 1; J_ptr = J.zstep(J_ptr), ++i) {
        int idx_z_1 = i + 1;
        int idx_z_2 = i - 1;
        if (i == 0) {
            idx_z_2 = i + 1;
        } else if (i == psi.dims.z - 1) {
            idx_z_1 = i - 1;
        }

        float4 J_x;
        float4 J_y;
        float4 J_z;

        if (mode == 0) {
            J_x = (*psi(idx_x_1, y, i) - *psi(idx_x_2, y, i)) / 2.f;
            J_y = (*psi(x, idx_y_1, i) - *psi(x, idx_y_2, i)) / 2.f;
            J_z = (*psi(x, y, idx_z_1) - *psi(x, y, idx_z_2)) / 2.f;
        } else if (mode == 1) {
            J_x = (psi.get_displacement(idx_x_1, y, i) - psi.get_displacement(idx_x_2, y, i)) / 2.f;
            J_y = (psi.get_displacement(x, idx_y_1, i) - psi.get_displacement(x, idx_y_2, i)) / 2.f;
            J_z = (psi.get_displacement(x, y, idx_z_1) - psi.get_displacement(x, y, idx_z_2)) / 2.f;
        }

        Mat4f val;
        val.data[0] = make_float4(J_x.x, J_y.x, J_z.x, 0.f);
        val.data[1] = make_float4(J_x.y, J_y.y, J_z.y, 0.f);
        val.data[2] = make_float4(J_x.z, J_y.z, J_z.z, 0.f);

        *J(x, y, i) = val;
    }
}
