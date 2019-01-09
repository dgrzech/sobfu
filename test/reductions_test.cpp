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

class ReductionsTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    ReductionsTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~ReductionsTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {
        /* verbosity */
        params.verbosity = 2;

        /* volume params */
        params.volume_dims = cv::Vec3i::all(64);
        params.volume_size = cv::Vec3f::all(0.25f);

        params.tsdf_trunc_dist = 5.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);
        params.eta             = 2.f * params.volume_size[0] / static_cast<float>(params.volume_dims[0]);

        dims        = kfusion::device_cast<kfusion::device::Vec3i>(params.volume_dims);
        voxel_sizes = kfusion::device_cast<kfusion::device::Vec3f>(params.voxel_sizes());

        no_voxels = dims.x * dims.y * dims.z;

        /* init tsdf's */
        phi_global = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n      = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        /* init reduction class */
        reductor = new sobfu::device::Reductor(dims, voxel_sizes.x, params.tsdf_trunc_dist);
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
    cv::Ptr<kfusion::cuda::TsdfVolume> phi_global, phi_n;

    /* reduction class */
    sobfu::device::Reductor* reductor;

    /* max. error */
    float max_error = 1e-1f;
    /* floating point precision */
    const float epsilon = 1e-5f;
};

/* test calculation of the energy functional data term */
TEST_F(ReductionsTest, DataTermTest) {
    /* init tsdf's */
    kfusion::device::TsdfVolume phi_n_device(phi_n->data().ptr<float2>(), dims, voxel_sizes, params.tsdf_trunc_dist,
                                             params.eta, params.tsdf_max_weight);
    kfusion::device::clear_volume(phi_n_device); /* will contain all 0's */

    float3 c = make_float3(5.f, 5.f, 5.f);
    float r  = 0.01f;
    phi_global->initSphere(c, r); /* will contain all 1's */

    kfusion::device::TsdfVolume phi_global_device(phi_global->data().ptr<float2>(), dims, voxel_sizes,
                                                  params.tsdf_trunc_dist, params.eta, params.tsdf_max_weight);

    float data_energy = reductor->data_energy(phi_global->data().ptr<float2>(), phi_n->data().ptr<float2>());
    ASSERT_NEAR(data_energy, 0.5f * static_cast<float>(no_voxels) * 1.f, max_error);
}
