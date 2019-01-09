#include <sobfu/solver.hpp>

/*
 * SOLVER
 */

sobfu::cuda::Solver::Solver(Params& params) {
    /*
     * PARAMETERS
     */

    /* volume */
    cv::Vec3i d = params.volume_dims;
    dims        = make_int3(d[0], d[1], d[2]);
    no_voxels   = dims.x * dims.y * dims.z;

    cv::Vec3f vsz = params.voxel_sizes();
    voxel_sizes   = make_float3(vsz[0], vsz[1], vsz[2]);

    trunc_dist = params.tsdf_trunc_dist;
    eta        = params.eta;
    max_weight = params.tsdf_max_weight;

    /* solver */
    solver_params.verbosity = params.verbosity;

    solver_params.max_iter        = params.max_iter;
    solver_params.max_update_norm = params.max_update_norm;

    solver_params.lambda = params.lambda;
    solver_params.s      = params.s;

    solver_params.alpha = params.alpha;
    solver_params.w_reg = params.w_reg;

    /*
     * SOLVER HELPER CLASSES
     */

    /* gradients */
    spatial_grads = new sobfu::cuda::SpatialGradients(d);

    nabla_phi_n = new sobfu::device::TsdfGradient(spatial_grads->nabla_phi_n->get_data().ptr<float4>(), dims);
    nabla_phi_n_o_psi =
        new sobfu::device::TsdfGradient(spatial_grads->nabla_phi_n_o_psi->get_data().ptr<float4>(), dims);
    J           = new sobfu::device::Jacobian(spatial_grads->J->get_data().ptr<Mat4f>(), dims);
    J_inv       = new sobfu::device::Jacobian(spatial_grads->J_inv->get_data().ptr<Mat4f>(), dims);
    L           = new sobfu::device::Laplacian(spatial_grads->L->get_data().ptr<float4>(), dims);
    L_o_psi_inv = new sobfu::device::Laplacian(spatial_grads->L_o_psi_inv->get_data().ptr<float4>(), dims);
    nabla_U     = new sobfu::device::PotentialGradient(spatial_grads->nabla_U->get_data().ptr<float4>(), dims);
    nabla_U_S   = new sobfu::device::PotentialGradient(spatial_grads->nabla_U_S->get_data().ptr<float4>(), dims);

    spatial_grads_device = new sobfu::device::SpatialGradients(nabla_phi_n, nabla_phi_n_o_psi, J, J_inv, L, L_o_psi_inv,
                                                               nabla_U, nabla_U_S);

    /* reductor */
    r = new sobfu::device::Reductor(dims, voxel_sizes.x, trunc_dist);

    /* sobolev filter */
    h_S_i = new float[solver_params.s];
    cudaSafeCall(cudaMalloc((void**) &d_S_i, solver_params.s * sizeof(float)));

    decompose_sobolev_filter(solver_params, h_S_i);
    cudaSafeCall(cudaMemcpy(d_S_i, h_S_i, solver_params.s * sizeof(float), cudaMemcpyHostToDevice));
}

sobfu::cuda::Solver::~Solver() = default;

void sobfu::cuda::Solver::estimate_psi(const cv::Ptr<kfusion::cuda::TsdfVolume> phi_global,
                                       cv::Ptr<kfusion::cuda::TsdfVolume> phi_global_psi_inv,
                                       const cv::Ptr<kfusion::cuda::TsdfVolume> phi_n,
                                       cv::Ptr<kfusion::cuda::TsdfVolume> phi_n_psi,
                                       std::shared_ptr<sobfu::cuda::DeformationField> psi,
                                       std::shared_ptr<sobfu::cuda::DeformationField> psi_inv) {
    /* DEVICE CLASSES */
    sobfu::device::DeformationField psi_device(psi->get_data().ptr<float4>(), dims);
    sobfu::device::DeformationField psi_inv_device(psi_inv->get_data().ptr<float4>(), dims);

    kfusion::device::TsdfVolume phi_global_device(phi_global->data().ptr<float2>(), dims, voxel_sizes, trunc_dist, eta,
                                                  max_weight);
    kfusion::device::TsdfVolume phi_global_psi_inv_device(phi_global_psi_inv->data().ptr<float2>(), dims, voxel_sizes,
                                                          trunc_dist, eta, max_weight);

    kfusion::device::TsdfVolume phi_n_device(phi_n->data().ptr<float2>(), dims, voxel_sizes, trunc_dist, eta,
                                             max_weight);
    kfusion::device::TsdfVolume phi_n_psi_device(phi_n_psi->data().ptr<float2>(), dims, voxel_sizes, trunc_dist, eta,
                                                 max_weight);

    SDFs sdfs(phi_global_device, phi_global_psi_inv_device, phi_n_device, phi_n_psi_device);

    sobfu::device::TsdfDifferentiator tsdf_diff(phi_n_psi_device);
    sobfu::device::Differentiator diff(psi_device);
    sobfu::device::Differentiator diff_inv(psi_inv_device);
    sobfu::device::SecondOrderDifferentiator second_order_diff(psi_device);

    Differentiators differentiators(tsdf_diff, diff, diff_inv, second_order_diff);

    /* run the solver */
    sobfu::device::estimate_psi(sdfs, psi_device, psi_inv_device, spatial_grads_device, differentiators, d_S_i, r,
                                solver_params);
}

/*
 * SOBOLEV FILTER
 */

static void sobfu::cuda::get_3d_sobolev_filter(SolverParams& params, float* h_S_i) {
    int s3 = params.s * params.s * params.s;

    /* init identity and laplacian matrices */
    cv::Mat Id    = cv::Mat::eye(s3, s3, CV_32FC1);
    cv::Mat L_mat = -6.f * cv::Mat::eye(s3, s3, CV_32FC1);

    /* calculate laplacian matrix */
    for (int i = 0; i <= static_cast<int>(pow(params.s, 3)) - 1; ++i) {
        int idx_z = i / (params.s * params.s);
        int idx_y = (i - idx_z * params.s * params.s) / params.s;
        int idx_x = i - params.s * (idx_y + params.s * idx_z);

        if (idx_x + 1 < params.s) {
            int pos                 = (idx_x + 1) + idx_y * params.s + idx_z * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }
        if (idx_x - 1 >= 0) {
            int pos                 = (idx_x - 1) + idx_y * params.s + idx_z * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }

        if (idx_y + 1 < params.s) {
            int pos                 = idx_x + (idx_y + 1) * params.s + idx_z * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }
        if (idx_y - 1 >= 0) {
            int pos                 = idx_x + (idx_y - 1) * params.s + idx_z * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }

        if (idx_z + 1 < params.s) {
            int pos                 = idx_x + idx_y * params.s + (idx_z + 1) * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }
        if (idx_z - 1 >= 0) {
            int pos                 = idx_x + idx_y * params.s + (idx_z - 1) * params.s * params.s;
            L_mat.at<float>(i, pos) = 1.f;
        }
    }

    /* init one-hot vector v */
    cv::Mat v                       = cv::Mat::zeros(s3, 1, CV_32FC1);
    v.at<float>(floor(s3 / 2.f), 0) = 1.f;

    /* init sobolev filter S */
    cv::Mat S = cv::Mat::zeros(s3, 1, CV_32FC1);
    /* solve for S */
    cv::solve((Id - params.lambda * L_mat), v, S, cv::DECOMP_SVD);

    std::cout << S << std::endl;
}

static void sobfu::cuda::decompose_sobolev_filter(SolverParams& params, float* h_S_i) {
    if (params.s == 3) {
        if (params.lambda == 0.1f) {
            h_S_i[0] = 0.06537f;
            h_S_i[1] = 0.99572f;
            h_S_i[2] = h_S_i[0];
        }
    }

    if (params.s == 7) {
        if (params.lambda == 0.05f) {
            h_S_i[0] = 0.00006f;
            h_S_i[1] = 0.00015f;
            h_S_i[2] = 0.03917f;
            h_S_i[3] = 0.99846f;
            h_S_i[4] = h_S_i[2];
            h_S_i[5] = h_S_i[1];
            h_S_i[6] = h_S_i[0];
        }

        if (params.lambda == 0.1f) {
            h_S_i[0] = 0.00030f;
            h_S_i[1] = 0.00441f;
            h_S_i[2] = 0.06571f;
            h_S_i[3] = 0.99565f;
            h_S_i[4] = h_S_i[2];
            h_S_i[5] = h_S_i[1];
            h_S_i[6] = h_S_i[0];
        }

        if (params.lambda == 0.2f) {
            h_S_i[0] = 0.00120f;
            h_S_i[1] = 0.01094f;
            h_S_i[2] = 0.10204f;
            h_S_i[3] = 0.98941f;
            h_S_i[4] = h_S_i[2];
            h_S_i[5] = h_S_i[1];
            h_S_i[6] = h_S_i[0];
        }

        if (params.lambda == 0.4f) {
            h_S_i[0] = 0.00169f;
            h_S_i[1] = 0.01312f;
            h_S_i[2] = 0.10927f;
            h_S_i[3] = 0.98781f;
            h_S_i[4] = h_S_i[2];
            h_S_i[5] = h_S_i[1];
            h_S_i[6] = h_S_i[0];
        }
    }

    if (params.s == 9) {
        if (params.lambda == 0.05f) {
            h_S_i[0] = 0.000003f;
            h_S_i[1] = 0.00006f;
            h_S_i[2] = 0.00155f;
            h_S_i[3] = 0.03917f;
            h_S_i[4] = 0.99846f;
            h_S_i[5] = 0.03917f;
            h_S_i[6] = 0.00155f;
            h_S_i[7] = 0.00006f;
            h_S_i[8] = 0.000003f;
        }

        if (params.lambda == 0.1f) {
            h_S_i[0] = 0.00002f;
            h_S_i[1] = 0.00030f;
            h_S_i[2] = 0.00441f;
            h_S_i[3] = 0.06571f;
            h_S_i[4] = 0.99565f;
            h_S_i[5] = h_S_i[3];
            h_S_i[6] = h_S_i[2];
            h_S_i[7] = h_S_i[1];
            h_S_i[8] = h_S_i[0];
        }
    }

    if (params.s == 11) {
        if (params.lambda == 0.1f) {
            h_S_i[0]  = 0.0000015f;
            h_S_i[1]  = 0.00002f;
            h_S_i[2]  = 0.00030f;
            h_S_i[3]  = 0.00441f;
            h_S_i[4]  = 0.06571f;
            h_S_i[5]  = 0.99565f;
            h_S_i[6]  = h_S_i[4];
            h_S_i[7]  = h_S_i[3];
            h_S_i[8]  = h_S_i[2];
            h_S_i[9]  = h_S_i[1];
            h_S_i[10] = h_S_i[0];
        }
    }

    /* normalise filter to unit sum */
    float sum = 0.f;
    for (int i = 0; i < params.s; ++i) {
        sum += h_S_i[i];
    }

    for (int i = 0; i < params.s; ++i) {
        h_S_i[i] /= sum;
    }
}
