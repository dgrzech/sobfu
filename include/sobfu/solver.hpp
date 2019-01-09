#pragma once

/* kinfu incldues */
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/safe_call.hpp>

/* sobfu includes */
#include <sobfu/reductor.hpp>
#include <sobfu/scalar_fields.hpp>
#include <sobfu/vector_fields.hpp>

/*
 * SOLVER PARAMETERS
 */

struct SolverParams {
    int verbosity, max_iter, s;
    float max_update_norm, lambda, alpha, w_reg;
};

/*
 * SDF'S USED IN THE SOLVER
 */

struct SDFs {
    SDFs(kfusion::device::TsdfVolume &phi_global_, kfusion::device::TsdfVolume &phi_global_psi_inv_,
         kfusion::device::TsdfVolume &phi_n_, kfusion::device::TsdfVolume &phi_n_psi_)
        : phi_global(phi_global_), phi_global_psi_inv(phi_global_psi_inv_), phi_n(phi_n_), phi_n_psi(phi_n_psi_) {}

    kfusion::device::TsdfVolume phi_global, phi_global_psi_inv, phi_n, phi_n_psi;
};

/*
 * SPATIAL DIFFERENTIATORS
 */

struct Differentiators {
    Differentiators(sobfu::device::TsdfDifferentiator &tsdf_diff_, sobfu::device::Differentiator &diff_,
                    sobfu::device::Differentiator &diff_inv_,
                    sobfu::device::SecondOrderDifferentiator &second_order_diff_)

        : tsdf_diff(tsdf_diff_), diff(diff_), diff_inv(diff_inv_), second_order_diff(second_order_diff_) {}

    sobfu::device::TsdfDifferentiator tsdf_diff;
    sobfu::device::Differentiator diff, diff_inv;
    sobfu::device::SecondOrderDifferentiator second_order_diff;
};

namespace sobfu {
namespace cuda {

/*
 * SOLVER
 */

class Solver {
public:
    /* constructor */
    Solver(Params &params);
    /* destructor */
    ~Solver();

    void estimate_psi(const cv::Ptr<kfusion::cuda::TsdfVolume> phi_global,
                      cv::Ptr<kfusion::cuda::TsdfVolume> phi_global_psi_inv,
                      const cv::Ptr<kfusion::cuda::TsdfVolume> phi_n, cv::Ptr<kfusion::cuda::TsdfVolume> phi_n_psi,
                      std::shared_ptr<sobfu::cuda::DeformationField> psi,
                      std::shared_ptr<sobfu::cuda::DeformationField> psi_inv);

private:
    /* volume params */
    int3 dims;
    float3 voxel_sizes;

    int no_voxels;
    float trunc_dist, eta, max_weight;

    /* solver params */
    SolverParams solver_params;

    /* gradients */
    sobfu::cuda::SpatialGradients *spatial_grads;
    sobfu::device::SpatialGradients *spatial_grads_device;

    sobfu::device::TsdfGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    sobfu::device::Jacobian *J, *J_inv;
    sobfu::device::Laplacian *L, *L_o_psi_inv;
    sobfu::device::PotentialGradient *nabla_U, *nabla_U_S;

    /* used to calculate value of the energy functional */
    sobfu::device::Reductor *r;

    /* sobolev filter */
    float *h_S_i, *d_S_i;
};

/* get 3d sobolev filter */
static void get_3d_sobolev_filter(SolverParams &params, float *h_S_i);
/* calculate 1d filters from a separable 3d filter */
static void decompose_sobolev_filter(SolverParams &params, float *h_S_i);

}  // namespace cuda

namespace device {

/* potential gradient */
__global__ void calculate_potential_gradient_kernel(float2 *phi_n_psi, float2 *phi_global, float4 *nabla_phi_n_o_psi,
                                                    float4 *L, float4 *nabla_U, float w_reg, int dim_x, int dim_y,
                                                    int dim_z);
void calculate_potential_gradient(kfusion::device::TsdfVolume &phi_n_psi, kfusion::device::TsdfVolume &phi_global,
                                  sobfu::device::TsdfGradient &nabla_phi_n_o_psi, sobfu::device::Laplacian &L,
                                  sobfu::device::PotentialGradient &nabla_U, float w_reg);

/* estimate psi */
void estimate_psi(SDFs &sdfs, sobfu::device::DeformationField &psi, sobfu::device::DeformationField &psi_inv,
                  sobfu::device::SpatialGradients *spatial_grads, Differentiators &diffs, float *d_S_i,
                  sobfu::device::Reductor *r, SolverParams &params);

/* DEFORMATION FIELD UPDATES */
__global__ void update_psi_kernel(float4 *psi, float4 *nabla_U_S, float4 *updates, float alpha, int dim_x, int dim_y,
                                  int dim_z);
void update_psi(sobfu::device::DeformationField &psi, sobfu::device::PotentialGradient &nabla_U_S, float4 *updates,
                float alpha);

/*
 * CONVOLUTIONS
 */

void set_convolution_kernel(float *d_kernel);

__global__ void convolution_rows_kernel(float4 *d_dst, float4 *d_src, int image_w, int image_h, int image_d);
__global__ void convolution_columns_kernel(float4 *d_dst, float4 *d_src, int image_w, int image_h, int image_d);
__global__ void convolution_depth_kernel(float4 *d_dst, float4 *d_src, int image_w, int image_h, int image_d);

void convolution_rows(float4 *d_dst, float4 *updates, int image_w, int image_h, int image_d);
void convolution_columns(float4 *d_dst, float4 *updates, int image_w, int image_h, int image_d);
void convolution_depth(float4 *d_dst, float4 *updates, int image_w, int image_h, int image_d);

}  // namespace device
}  // namespace sobfu
