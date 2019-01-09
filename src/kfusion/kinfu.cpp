#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>

using namespace std;
using namespace kfusion;
using namespace kfusion::cuda;

static inline float deg2rad(float alpha) { return alpha * 0.017453293f; }

kfusion::KinFuParams kfusion::KinFuParams::default_params() {
    const int iters[] = {10, 5, 4, 0};
    const int levels  = sizeof(iters) / sizeof(iters[0]);

    KinFuParams p;

    p.cols = 640;  // pixels
    p.rows = 480;  // pixels
    p.intr = Intr(525.f, 525.f, p.cols / 2 - 0.5f, p.rows / 2 - 0.5f);

    p.volume_dims = Vec3i::all(512);  // number of voxels
    p.volume_size = Vec3f::all(3.f);  // meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0] / 2, -p.volume_size[1] / 2, 0.5f));

    p.bilateral_sigma_depth   = 0.04f;  // meter
    p.bilateral_sigma_spatial = 4.5;    // pixels
    p.bilateral_kernel_size   = 7;      // pixels

    p.icp_truncate_depth_dist = 0.f;            // meters, disabled
    p.icp_dist_thres          = 0.1f;           // meters
    p.icp_angle_thres         = deg2rad(30.f);  // radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f;    // meters, disabled
    p.tsdf_trunc_dist          = 0.04f;  // meters;
    p.tsdf_max_weight          = 64;     // frames

    p.raycast_step_factor   = 0.75f;  // in voxel sizes
    p.gradient_delta_factor = 0.5f;   // in voxel sizes

    // p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f);  // meters

    return p;
}

const kfusion::KinFuParams &kfusion::KinFu::params() const { return params_; }

kfusion::KinFuParams &kfusion::KinFu::params() { return params_; }

const kfusion::cuda::TsdfVolume &kfusion::KinFu::tsdf() const { return *volume_; }

kfusion::cuda::TsdfVolume &kfusion::KinFu::tsdf() { return *volume_; }

const kfusion::cuda::ProjectiveICP &kfusion::KinFu::icp() const { return *icp_; }

kfusion::cuda::ProjectiveICP &kfusion::KinFu::icp() { return *icp_; }

const kfusion::cuda::MarchingCubes &kfusion::KinFu::mc() const { return *mc_; }

kfusion::cuda::MarchingCubes &kfusion::KinFu::mc() { return *mc_; }

void kfusion::KinFu::allocate_buffers() {
    const int LEVELS = cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);

    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);

    for (int i = 0; i < LEVELS; ++i) {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void kfusion::KinFu::reset() {
    if (frame_counter_)
        cout << "Reset" << endl;

    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(30000);
    poses_.push_back(Affine3f::Identity());
    volume_->clear();
}

kfusion::Affine3f kfusion::KinFu::getCameraPose(int time) const {
    if ((time > static_cast<int>(poses_.size())) || (time < 0)) {
        time = static_cast<int>(poses_.size()) - 1;
    }

    return poses_[time];
}
