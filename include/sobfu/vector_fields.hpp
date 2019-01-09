#pragma once

/* sobfu includes */
#include <sobfu/params.hpp>
#include <sobfu/precomp.hpp>

/* kinfu includes */
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>
#include <kfusion/types.hpp>

namespace sobfu {
namespace cuda {

/*
 * VECTOR FIELD
 */

class VectorField {
public:
    /* constructor */
    VectorField(cv::Vec3i dims_);
    /* destructor */
    ~VectorField();

    /* get dimensions of the field */
    cv::Vec3i get_dims() const;

    /* get the data in the field */
    kfusion::cuda::CudaData get_data();
    const kfusion::cuda::CudaData get_data() const;
    /* set the data in the field */
    void set_data(kfusion::cuda::CudaData &data);

    /* clear field */
    void clear();
    /* print field */
    void print();
    /* get no. of nan's in the field */
    int get_no_nans();

protected:
    kfusion::cuda::CudaData data; /* field data */
    cv::Vec3i dims;               /* field dimensions */
};

/*
 * DEFORMATION FIELD
 */

class DeformationField : public VectorField {
public:
    /* constructor */
    DeformationField(cv::Vec3i dims_);
    /* destructor */
    ~DeformationField();

    /* init field to identity */
    void clear();

    /* apply the field to an sdf */
    void apply(const cv::Ptr<kfusion::cuda::TsdfVolume> phi, cv::Ptr<kfusion::cuda::TsdfVolume> phi_psi);
    /* approximate the inverse of the field */
    void get_inverse(sobfu::cuda::DeformationField &psi_inv);
};

/*
 * TSDF GRADIENT, LAPLACIAN, POTENTIAL GRADIENT
 */

typedef VectorField TsdfGradient;
typedef VectorField Laplacian;
typedef VectorField PotentialGradient;

/*
 * JACOBIAN
 */

class Jacobian {
public:
    /* constructor */
    Jacobian(cv::Vec3i dims_);
    /* destructor */
    ~Jacobian();

    kfusion::cuda::CudaData get_data();
    const kfusion::cuda::CudaData get_data() const;

    /* clear jacobian */
    void clear();

private:
    kfusion::cuda::CudaData data;
    cv::Vec3i dims;
};

/*
 * SPATIAL GRADIENTS
 */

struct SpatialGradients {
    /* constructor */
    SpatialGradients(cv::Vec3i dims_);
    /* destructor */
    ~SpatialGradients();

    TsdfGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    Jacobian *J, *J_inv;
    Laplacian *L, *L_o_psi_inv;
    PotentialGradient *nabla_U, *nabla_U_S;
};

}  // namespace cuda
}  // namespace sobfu

namespace sobfu {
namespace device {

/*
 * VECTOR FIELD
 */

struct VectorField {
    /* constructor */
    VectorField(float4 *const data_, const int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ float4 *beg(int x, int y) const;
    __device__ __forceinline__ float4 *zstep(float4 *const ptr) const;
    __device__ __forceinline__ float4 *operator()(int x, int y, int z) const;

    __device__ __forceinline__ float4 get_displacement(int x, int y, int z) const;

    float4 *const data; /* vector field data */
    const int3 dims;    /* vector field dimensions */
};

/* clear field */
__global__ void clear_kernel(VectorField field);
void clear(VectorField &field);

/*
 * DEFORMATION FIELD
 */

typedef VectorField DeformationField;

/* set deformation field to identity */
__global__ void init_identity_kernel(DeformationField field);
void init_identity(DeformationField &field);
/* apply psi to an sdf */
__global__ void apply_kernel(const kfusion::device::TsdfVolume phi, kfusion::device::TsdfVolume phi_warped,
                             const DeformationField psi);
void apply(const kfusion::device::TsdfVolume &phi, kfusion::device::TsdfVolume &phi_warped,
           const sobfu::device::DeformationField &psi);
/* approximate the inverse of a deformation field */
__global__ void estimate_inverse_kernel(sobfu::device::DeformationField psi, sobfu::device::DeformationField psi_inv);
void estimate_inverse(sobfu::device::DeformationField &psi, sobfu::device::DeformationField &psi_inv);

/*
 * TSDF GRADIENT
 */

typedef VectorField TsdfGradient;

struct TsdfDifferentiator {
    /* constructor */
    TsdfDifferentiator(kfusion::device::TsdfVolume &vol_) : vol(vol_) {}

    /* calculate the gradient */
    __device__ void operator()(sobfu::device::TsdfGradient &grad) const;
    void calculate(sobfu::device::TsdfGradient &grad);

    kfusion::device::TsdfVolume vol;
};

/* estimate tsdf gradient */
__global__ void estimate_gradient_kernel(const sobfu::device::TsdfDifferentiator diff,
                                         sobfu::device::TsdfGradient grad);
/* interpolate tsdf gradient */
__global__ void interpolate_gradient_kernel(sobfu::device::TsdfGradient nabla_phi_n_psi,
                                            sobfu::device::TsdfGradient nabla_phi_n_psi_t,
                                            sobfu::device::DeformationField psi);
void interpolate_gradient(sobfu::device::TsdfGradient &nabla_phi_n_psi, sobfu::device::TsdfGradient &nabla_phi_n_psi_t,
                          sobfu::device::DeformationField &psi);

/*
 * LAPLACIAN
 */

typedef VectorField Laplacian;

struct SecondOrderDifferentiator {
    /* constructor */
    SecondOrderDifferentiator(sobfu::device::DeformationField &psi_) : psi(psi_) {}

    __device__ void laplacian(sobfu::device::Laplacian &L) const;
    void calculate(sobfu::device::Laplacian &L);

    sobfu::device::DeformationField psi;
};

/* estimate laplacian */
__global__ void estimate_laplacian_kernel(const sobfu::device::SecondOrderDifferentiator diff,
                                          sobfu::device::Laplacian L);
/* interpolate laplacian */
__global__ void interpolate_laplacian_kernel(sobfu::device::Laplacian L, sobfu::device::Laplacian L_o_psi,
                                             sobfu::device::DeformationField psi);
void interpolate_laplacian(sobfu::device::Laplacian &L, sobfu::device::Laplacian &L_o_psi,
                           sobfu::device::DeformationField &psi);

/*
 * POTENTIAL GRADIENT
 */

typedef VectorField PotentialGradient;

/*
 * JACOBIAN
 */

struct Jacobian {
    /* constructor */
    Jacobian(Mat4f *const data_, int3 dims_) : data(data_), dims(dims_) {}

    __device__ __forceinline__ Mat4f *beg(int x, int y) const;
    __device__ __forceinline__ Mat4f *zstep(Mat4f *const ptr) const;
    __device__ __forceinline__ Mat4f *operator()(int x, int y, int z) const;

    Mat4f *const data; /* jacobian data */
    const int3 dims;
};

struct Differentiator {
    /* constructor */
    Differentiator(sobfu::device::DeformationField &psi_) : psi(psi_) {}

    /* calculate jacobian */
    __device__ void operator()(sobfu::device::Jacobian &J, int mode) const;
    void calculate(sobfu::device::Jacobian &J);
    void calculate_deformation_jacobian(sobfu::device::Jacobian &J);

    sobfu::device::DeformationField psi;
};

/* clear jacobian */
__global__ void clear_jacobian_kernel(Jacobian J);
void clear(Jacobian &J);
/* estimate jacobian */
__global__ void estimate_jacobian_kernel(const sobfu::device::Differentiator diff, sobfu::device::Jacobian J);
__global__ void estimate_deformation_jacobian_kernel(const sobfu::device::Differentiator diff,
                                                     sobfu::device::Jacobian J);

/*
 * SPATIAL GRADIENTS
 */

struct SpatialGradients {
    SpatialGradients(sobfu::device::TsdfGradient *nabla_phi_n_, sobfu::device::TsdfGradient *nabla_phi_n_o_psi_,
                     sobfu::device::Jacobian *J_, sobfu::device::Jacobian *J_inv_, sobfu::device::Laplacian *L_,
                     sobfu::device::Laplacian *L_o_psi_inv_, sobfu::device::PotentialGradient *nabla_U_,
                     sobfu::device::PotentialGradient *nabla_U_S_)
        : nabla_phi_n(nabla_phi_n_),
          nabla_phi_n_o_psi(nabla_phi_n_o_psi_),
          J(J_),
          J_inv(J_inv_),
          L(L_),
          L_o_psi_inv(L_o_psi_inv_),
          nabla_U(nabla_U_),
          nabla_U_S(nabla_U_S_) {}
    ~SpatialGradients();

    sobfu::device::TsdfGradient *nabla_phi_n, *nabla_phi_n_o_psi;
    sobfu::device::Jacobian *J, *J_inv;
    sobfu::device::Laplacian *L, *L_o_psi_inv;
    sobfu::device::PotentialGradient *nabla_U, *nabla_U_S;
};

}  // namespace device
}  // namespace sobfu
