/* gtest includes */
#include <gtest/gtest.h>

/* sobfu includes */
#include <sobfu/params.hpp>
#include <sobfu/solver.hpp>

/* kinfu includes */
#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>
#include <kfusion/types.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>
#include <iostream>
#include <memory>

class DeformationFieldTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    DeformationFieldTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~DeformationFieldTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {
        /* volume params */
        params.volume_dims = cv::Vec3i::all(64);
        params.volume_size = cv::Vec3f::all(0.25f);

        params.tsdf_trunc_dist       = 10.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);
        params.gradient_delta_factor = 0.1f;

        params.eta = 2.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);

        /* camera params */
        params.intr = kfusion::Intr(1.f, 1.f, 0.f, 0.f);

        dims        = kfusion::device_cast<kfusion::device::Vec3i>(params.volume_dims);
        voxel_sizes = kfusion::device_cast<kfusion::device::Vec3f>(params.voxel_sizes());

        no_voxels = dims.x * dims.y * dims.z;

        /* init tsdf's */
        phi_global         = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_global_psi_inv = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n              = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n_psi          = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        /* init psi */
        psi     = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);
        psi_inv = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {}

    /* Objects declared here can be used by all tests in the test case for Solver. */

    /* sobfu params */
    Params params;

    int3 dims;
    float3 voxel_sizes;

    int no_voxels;

    /* global and live tsdf's */
    cv::Ptr<kfusion::cuda::TsdfVolume> phi_global, phi_global_psi_inv, phi_n, phi_n_psi;
    /* deformation field */
    std::shared_ptr<sobfu::cuda::DeformationField> psi, psi_inv;
    /* solver */
    std::shared_ptr<sobfu::cuda::Solver> solver;

    /* max. error */
    float max_error = 1e-1f;
    /* floating point precision */
    const float epsilon = 1e-5f;
};

/* test that the vector field is correctly initialised to 0 */
TEST_F(DeformationFieldTest, ClearTest) {
    kfusion::cuda::CudaData data = psi->get_data();

    int sizes[3]    = {dims.x, dims.y, dims.z};
    cv::Mat *matrix = new cv::Mat(3, sizes, CV_32FC4);
    data.download(matrix->ptr<float4>());

    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                ASSERT_NEAR(matrix->at<float4>(k, j, i).x, (float) i, epsilon);
                ASSERT_NEAR(matrix->at<float4>(k, j, i).y, (float) j, epsilon);
                ASSERT_NEAR(matrix->at<float4>(k, j, i).z, (float) k, epsilon);
            }
        }
    }
}

/* test that the norm of the gradient of an sdf is 1 */
TEST_F(DeformationFieldTest, TsdfGradientTest) {
    /* init tsdf */
    float3 c = make_float3(0.16f, 0.16f, 0.16f);
    float r  = 0.01f;
    phi_global->initSphere(c, r);

    kfusion::device::TsdfVolume phi_global_device(phi_global->data().ptr<float2>(), dims, voxel_sizes,
                                                  params.tsdf_trunc_dist, params.eta, params.tsdf_max_weight);

    /* init gradient */
    kfusion::cuda::CudaData grad_data;
    grad_data.create(no_voxels * sizeof(float4));
    sobfu::device::TsdfGradient gradient_device(grad_data.ptr<float4>(), dims);

    sobfu::device::TsdfDifferentiator diff(phi_global_device);
    diff.calculate(gradient_device);

    /* test that the gradient magnitude is 0 in the truncated regions and close to 1 elsewhere */
    float2 *tsdf_ptr = new float2[no_voxels];
    phi_global->data().download(tsdf_ptr);

    float4 *grad_ptr = new float4[no_voxels];
    grad_data.download(grad_ptr);

    int sizes[3] = {dims.x, dims.y, dims.z};
    for (int i = 1; i < sizes[0] - 1; i++) {
        for (int j = 1; j < sizes[1] - 1; j++) {
            for (int k = 1; k < sizes[2] - 1; k++) {
                float tsdf_val  = (*(tsdf_ptr + i + j * sizes[0] + k * sizes[1] * sizes[0])).x;
                float4 grad_val = *(grad_ptr + i + j * sizes[0] + k * sizes[1] * sizes[0]);

                float norm = sqrtf(grad_val.x * grad_val.x + grad_val.y * grad_val.y + grad_val.z * grad_val.z);
                if (fabs(tsdf_val) < 1.f) { /* only check for non-truncated voxels */
                    ASSERT_NEAR(norm, voxel_sizes.x / params.tsdf_trunc_dist, 0.15f);
                }
            }
        }
    }
}

/* test that the jacobian of a uniform vector field is null */
TEST_F(DeformationFieldTest, UniformFieldJacobianTest) {
    /* init psi to a uniform field */
    int sizes[3] = {dims.x, dims.y, dims.z};

    cv::Mat *matrix = new cv::Mat(3, sizes, CV_32FC4);
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                matrix->at<float4>(k, j, i) = make_float4(1.f, 1.f, 1.f, 0.f);
            }
        }
    }

    kfusion::cuda::CudaData psi_data;
    psi_data.create(no_voxels * sizeof(float4));
    psi_data.upload(matrix->ptr<float4>(), no_voxels * sizeof(float4));
    sobfu::device::DeformationField psi_device(psi_data.ptr<float4>(), dims);

    /* init & clear the jacobian */
    kfusion::cuda::CudaData jacobian_data;
    jacobian_data.create(no_voxels * sizeof(Mat4f));
    sobfu::device::Jacobian J(jacobian_data.ptr<Mat4f>(), dims);

    /* calculate the jacobian */
    sobfu::device::Differentiator diff(psi_device);
    diff.calculate(J);

    /* test that the jacobian is null */
    Mat4f *ptr = new Mat4f[no_voxels];
    jacobian_data.download(ptr);

    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                Mat4f J_val = *(ptr + i + j * sizes[0] + k * sizes[1] * sizes[0]);

                for (int r = 0; r < 3; r++) {
                    ASSERT_NEAR(J_val.data[r].x, 0.f, epsilon);
                    ASSERT_NEAR(J_val.data[r].y, 0.f, epsilon);
                    ASSERT_NEAR(J_val.data[r].z, 0.f, epsilon);
                }
            }
        }
    }
}

/* test the calculation of the jacobian of a simple vector field */
TEST_F(DeformationFieldTest, JacobianTestSimple) {
    /* init psi */
    int sizes[3] = {dims.x, dims.y, dims.z};

    cv::Mat *matrix = new cv::Mat(3, sizes, CV_32FC4);
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                matrix->at<float4>(k, j, i) = make_float4(i, j, k, 0.f);
            }
        }
    }

    kfusion::cuda::CudaData psi_data;
    psi_data.create(no_voxels * sizeof(float4));
    psi_data.upload(matrix->ptr<float4>(), no_voxels * sizeof(float4));
    sobfu::device::DeformationField psi_device(psi_data.ptr<float4>(), dims);

    /* init the jacobian */
    kfusion::cuda::CudaData jacobian_data;
    jacobian_data.create(no_voxels * sizeof(Mat4f));
    sobfu::device::Jacobian J(jacobian_data.ptr<Mat4f>(), dims);

    /* calculate the jacobian */
    sobfu::device::Differentiator diff(psi_device);
    diff.calculate(J);

    /* test that the jacobian values are correct */
    Mat4f *ptr = new Mat4f[no_voxels];
    jacobian_data.download(ptr);

    for (int i = 1; i < sizes[0] - 1; i++) {
        for (int j = 1; j < sizes[1] - 1; j++) {
            for (int k = 1; k < sizes[2] - 1; k++) {
                Mat4f J_val = *(ptr + i + j * sizes[0] + k * sizes[1] * sizes[0]);

                ASSERT_NEAR(J_val.data[0].x, 1.f, epsilon);
                ASSERT_NEAR(J_val.data[0].y, 0.f, epsilon);
                ASSERT_NEAR(J_val.data[0].z, 0.f, epsilon);

                ASSERT_NEAR(J_val.data[1].x, 0.f, epsilon);
                ASSERT_NEAR(J_val.data[1].y, 1.f, epsilon);
                ASSERT_NEAR(J_val.data[1].z, 0.f, epsilon);

                ASSERT_NEAR(J_val.data[2].x, 0.f, epsilon);
                ASSERT_NEAR(J_val.data[2].y, 0.f, epsilon);
                ASSERT_NEAR(J_val.data[2].z, 1.f, epsilon);
            }
        }
    }
}

/* test the computation of the jacobian and laplacian of a vector field */
TEST_F(DeformationFieldTest, JacobianLaplacianTestComplicated) {
    /* init psi */
    int sizes[3] = {dims.x, dims.y, dims.z};

    cv::Mat *matrix = new cv::Mat(3, sizes, CV_32FC4);
    for (int i = 0; i < sizes[0]; i++) {
        for (int j = 0; j < sizes[1]; j++) {
            for (int k = 0; k < sizes[2]; k++) {
                matrix->at<float4>(k, j, i) = make_float4(i * (1.f - j), exp(-k) + j, k, 0.f);
            }
        }
    }

    kfusion::cuda::CudaData psi_data;
    psi_data.create(no_voxels * sizeof(float4));
    psi_data.upload(matrix->ptr<float4>(), no_voxels * sizeof(float4));
    sobfu::device::DeformationField psi_device(psi_data.ptr<float4>(), dims);

    /* init the jacobian */
    kfusion::cuda::CudaData J_data;
    J_data.create(no_voxels * sizeof(Mat4f));
    sobfu::device::Jacobian J(J_data.ptr<Mat4f>(), dims);

    /* calculate the jacobian */
    sobfu::device::Differentiator diff(psi_device);
    diff.calculate(J);

    /* test that the jacobian values are correct */
    Mat4f *ptr = new Mat4f[no_voxels];
    J_data.download(ptr);

    /*	       (1-y  -x      0    )
     * J_psi = ( 0    1  -exp(-z) )
     * 	       ( 0    0      1    )
     */

    for (int i = 1; i < sizes[0] - 1; i++) {
        for (int j = 1; j < sizes[1] - 1; j++) {
            for (int k = 1; k < sizes[2] - 1; k++) {
                Mat4f J_val = *(ptr + i + j * sizes[0] + k * sizes[1] * sizes[0]);

                ASSERT_NEAR(J_val.data[0].x, 1.f - j, max_error);
                ASSERT_NEAR(J_val.data[0].y, -i, max_error);
                ASSERT_NEAR(J_val.data[0].z, 0.f, max_error);

                ASSERT_NEAR(J_val.data[1].x, 0, max_error);
                ASSERT_NEAR(J_val.data[1].y, 1.f, max_error);
                ASSERT_NEAR(J_val.data[1].z, -exp(-k), max_error);

                ASSERT_NEAR(J_val.data[2].x, 0.f, max_error);
                ASSERT_NEAR(J_val.data[2].y, 0.f, max_error);
                ASSERT_NEAR(J_val.data[2].z, 1.f, max_error);
            }
        }
    }

    /* init the laplacian */
    kfusion::cuda::CudaData laplacian_data;
    laplacian_data.create(no_voxels * sizeof(float4));
    sobfu::device::Laplacian L(laplacian_data.ptr<float4>(), dims);

    /* calculate the laplacian */
    sobfu::device::SecondOrderDifferentiator secondOrderDiff(psi_device);
    secondOrderDiff.calculate(L);

    /* test that the laplacian values are correct */
    float4 *L_ptr = new float4[no_voxels];
    laplacian_data.download(L_ptr);

    /*
     * expecting -(0  exp(-z)  0)--we calculate the negative laplacian for simplicity
     */

    for (int i = 1; i < sizes[0] - 1; i++) {
        for (int j = 1; j < sizes[1] - 1; j++) {
            for (int k = 1; k < sizes[2] - 1; k++) {
                float4 L_val = *(L_ptr + i + j * sizes[0] + k * sizes[1] * sizes[0]);

                ASSERT_NEAR(L_val.x, 0.f, max_error);
                ASSERT_NEAR(L_val.y, -exp(-k), max_error);
                ASSERT_NEAR(L_val.z, 0.f, max_error);
            }
        }
    }
}
