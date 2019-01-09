#pragma once

/* kinfu includes */
#include <kfusion/types.hpp>

/* sobfu parameters */
struct Params {
    int cols = 640;
    int rows = 480; /* no. of rows and columns in the frames */

    cv::Vec3i volume_dims; /* volume dimensions in voxels */
    cv::Vec3f volume_size; /* volume size in metres */

    cv::Affine3f volume_pose;
    kfusion::Intr intr; /* camera intrinsics */

    float icp_truncate_depth_dist; /* depth truncation distance */

    float bilateral_sigma_depth, bilateral_sigma_spatial;
    int bilateral_kernel_size;

    float tsdf_trunc_dist, eta; /* tsdf truncation distance and expected object thickness */
    float tsdf_max_weight;      /* tsdf max. weight */

    float gradient_delta_factor;

    int start_frame = 0; /* frame when to start registration */
    int verbosity   = 0; /* solver verbosity */

    int s /* filter size */, max_iter /* max. no of iterations of the solver */;
    float max_update_norm /* max. update norm */, lambda /* filter parameter */, alpha /* gradient descent step size */,
        w_reg /* weight of the regularisation term */;

    cv::Vec3f voxel_sizes() {
        return cv::Vec3f(volume_size[0] / volume_dims[0], volume_size[1] / volume_dims[1],
                         volume_size[2] / volume_dims[2]);
    }
};
