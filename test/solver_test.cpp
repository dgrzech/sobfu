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

class SolverTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    SolverTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~SolverTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {
        /* verbosity */
        params.verbosity = 1;

        /* volume params */
        params.volume_dims = cv::Vec3i::all(64);
        params.volume_size = cv::Vec3f::all(0.25f);

        params.tsdf_trunc_dist       = 10.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);
        params.gradient_delta_factor = 0.1f;

        params.eta = 2.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);

        /* camera params */
        params.intr = kfusion::Intr(1.f, 1.f, 0.f, 0.f);

        /* solver params */
        params.max_iter        = 2048;
        params.max_update_norm = -1.f;

        params.s      = 7;
        params.lambda = 0.1f;

        params.alpha = 0.001f;
        params.w_reg = 0.4f;

        dims        = kfusion::device_cast<kfusion::device::Vec3i>(params.volume_dims);
        voxel_sizes = kfusion::device_cast<kfusion::device::Vec3f>(params.voxel_sizes());

        no_voxels = dims.x * dims.y * dims.z;

        /* init tsdf's */
        phi_global         = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_global_psi_inv = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n              = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n_psi          = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        mc = cv::Ptr<kfusion::cuda::MarchingCubes>(new kfusion::cuda::MarchingCubes());

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
    /* marching cubes--for visualisation purposes */
    cv::Ptr<kfusion::cuda::MarchingCubes> mc;
    /* deformation field */
    std::shared_ptr<sobfu::cuda::DeformationField> psi, psi_inv;
    /* solver */
    std::shared_ptr<sobfu::cuda::Solver> solver;

    /* max. error */
    float max_error = 1e-1f;
    /* floating point precision */
    const float epsilon = 1e-5f;
};

/* test the alignment of 2 spheres */
TEST_F(SolverTest, AlignmentTestSphereTranslation) {
    /* set parameters */
    params.max_iter = 2048;

    params.alpha = 0.01f;
    params.w_reg = 0.4f;

    /* init solver */
    solver = std::make_shared<sobfu::cuda::Solver>(params);

    /* init phi_global and phi_n */
    int sizes[3] = {dims.x, dims.y, dims.z};

    float3 c = make_float3(0.13f, 0.13f, 0.13f);
    float r  = 0.012f;
    phi_global->initSphere(c, r);

    c = make_float3(0.125f, 0.13f, 0.13f);
    phi_n->initSphere(c, r);
    phi_n_psi->initSphere(c, r);

    /* calculate psi and apply it to phi_n */
    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);
}

/* test the alignment of a sphere translating and expanding */
TEST_F(SolverTest, AlignmentTestSphereExpanding) {
    /* set parameters */
    params.max_iter = 2048;

    params.alpha = 0.005f;
    params.w_reg = 0.4f;

    /* init solver */
    solver = std::make_shared<sobfu::cuda::Solver>(params);

    /* init phi_global and phi_n */
    int sizes[3] = {dims.x, dims.y, dims.z};

    float3 c = make_float3(0.13f, 0.13f, 0.13f);
    float r  = 0.012f;
    phi_global->initSphere(c, r);

    c = make_float3(0.125f, 0.13f, 0.13f);
    r = 0.0145f;
    phi_n->initSphere(c, r);
    phi_n_psi->initSphere(c, r);

    /* calculate psi and apply it to phi_n */
    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);
}

/* test the alignment of 2 spheres over 3 frames */
TEST_F(SolverTest, SerialAlignmentTest) {
    /* set parameters */
    params.max_iter = 2048;

    params.alpha = 0.005f;
    params.w_reg = 0.4f;

    /* init solver */
    solver = std::make_shared<sobfu::cuda::Solver>(params);

    /* init phi_global and phi_n */
    int sizes[3] = {dims.x, dims.y, dims.z};

    float3 c = make_float3(0.13f, 0.13f, 0.13f);
    float r  = 0.02f;
    phi_global->initSphere(c, r);

    c = make_float3(0.125f, 0.13f, 0.132f);
    phi_n->initSphere(c, r);
    phi_n_psi->initSphere(c, r);

    /*
     * FRAME 1 to FRAME 0
     *
     */

    std::cout << "\nFRAME 1\n" << std::endl;

    /* calculate psi and apply it to phi_n */
    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);

    /*
     * FRAME 2 to FRAME 0
     *
     */

    std::cout << "\nFRAME 2\n" << std::endl;

    phi_n->clear();
    phi_n_psi->clear();

    c = make_float3(0.123f, 0.13f, 0.132f);
    phi_n->initSphere(c, r);
    psi->apply(phi_n, phi_n_psi);

    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);
}
