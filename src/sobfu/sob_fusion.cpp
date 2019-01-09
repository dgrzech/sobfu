#include <sobfu/sob_fusion.hpp>

SobFusion::SobFusion(const Params &params) : frame_counter_(0), params(params) {
    int cols = params.cols;
    int rows = params.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(1);
    curr_.normals_pyr.resize(1);

    prev_.depth_pyr.resize(1);
    prev_.normals_pyr.resize(1);

    curr_.points_pyr.resize(1);
    prev_.points_pyr.resize(1);

    curr_.depth_pyr[0].create(rows, cols);
    curr_.normals_pyr[0].create(rows, cols);

    prev_.depth_pyr[0].create(rows, cols);
    prev_.normals_pyr[0].create(rows, cols);

    curr_.points_pyr[0].create(rows, cols);
    prev_.points_pyr[0].create(rows, cols);

    depths_.create(rows, cols);
    normals_.create(rows, cols);
    points_.create(rows, cols);

    poses_.clear();
    poses_.reserve(4096);
    poses_.push_back(cv::Affine3f::Identity());

    mc = cv::Ptr<kfusion::cuda::MarchingCubes>(new kfusion::cuda::MarchingCubes());
    mc->setPose(params.volume_pose);
}

SobFusion::~SobFusion() = default;

Params &SobFusion::getParams() { return params; }

pcl::PolygonMesh::Ptr SobFusion::get_phi_global_mesh() { return get_mesh(phi_global); }

pcl::PolygonMesh::Ptr SobFusion::get_phi_global_psi_inv_mesh() { return get_mesh(phi_global_psi_inv); }

pcl::PolygonMesh::Ptr SobFusion::get_phi_n_mesh() { return get_mesh(phi_n); }

pcl::PolygonMesh::Ptr SobFusion::get_phi_n_psi_mesh() { return get_mesh(phi_n_psi); }

std::shared_ptr<sobfu::cuda::DeformationField> SobFusion::getDeformationField() { return this->psi; }

/* PIPELINE
 *
 * --- frame 0 ---
 *
 * 1. bilateral filter
 * 2. depth truncation
 * 3. initailisation of phi_global and phi_n
 *
 * --- frames n + 1 ---
 * 1. bilateral filter
 * 2. depth truncation
 * 3. initialisation of phi_n
 * 4. estimation of psi
 * 5. fusion of phi_n(psi)
 * 6. warp of phi_global with psi^-1
 *
 */

bool SobFusion::operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image & /*image*/) {
    std::cout << "--- FRAME NO. " << frame_counter_ << " ---" << std::endl;

    /*
     *  bilateral filter
     */

    kfusion::cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], params.bilateral_kernel_size,
                                        params.bilateral_sigma_spatial, params.bilateral_sigma_depth);

    /*
     * depth truncation
     */

    kfusion::cuda::depthTruncation(curr_.depth_pyr[0], params.icp_truncate_depth_dist);

    /*
     *  compute distances using depth map
     */

    kfusion::cuda::computeDists(curr_.depth_pyr[0], dists_, params.intr);

    if (frame_counter_ == 0) {
        /*
         * INITIALISATION OF PHI_GLOBAL
         */

        phi_global = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_global->integrate(dists_, poses_.back(), params.intr);

        /*
         * INITIALISATION OF PHI_GLOBAL(PSI_INV), PHI_N, AND PHI_N(PSI)
         */

        phi_global_psi_inv = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n              = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));
        phi_n_psi          = cv::Ptr<kfusion::cuda::TsdfVolume>(new kfusion::cuda::TsdfVolume(params));

        /*
         * INITIALISATION OF PSI AND PSI_INV
         */

        this->psi     = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);
        this->psi_inv = std::make_shared<sobfu::cuda::DeformationField>(params.volume_dims);

        /*
         * INITIALISATION OF THE SOLVER
         */

        this->solver = std::make_shared<sobfu::cuda::Solver>(params);

        return ++frame_counter_, true;
    }

    /*
     * UPDATE OF PHI_N
     */

    phi_n->clear();
    this->phi_n->integrate(dists_, poses_.back(), params.intr);

    /*
     * ESTIMATION OF DEFORMATION FIELD AND SURFACE FUSION
     */

    if (frame_counter_ < params.start_frame) {
        this->phi_global->integrate(*phi_n);
        return ++frame_counter_, true;
    }

    solver->estimate_psi(phi_global, phi_global_psi_inv, phi_n, phi_n_psi, psi, psi_inv);
    this->phi_global->integrate(*phi_n_psi);

    return ++frame_counter_, true;
}

pcl::PolygonMesh::Ptr SobFusion::get_mesh(cv::Ptr<kfusion::cuda::TsdfVolume> vol) {
    kfusion::device::DeviceArray<pcl::PointXYZ> vertices_buffer_device;
    kfusion::device::DeviceArray<pcl::Normal> normals_buffer_device;

    /* run marching cubes */
    std::shared_ptr<kfusion::cuda::Surface> model =
        std::make_shared<kfusion::cuda::Surface>(mc->run(*vol, vertices_buffer_device, normals_buffer_device));
    kfusion::cuda::waitAllDefaultStream();

    pcl::PolygonMesh::Ptr mesh = convert_to_mesh(model->vertices);
    return mesh;
}

pcl::PolygonMesh::Ptr SobFusion::convert_to_mesh(const kfusion::cuda::DeviceArray<pcl::PointXYZ> &triangles) {
    if (triangles.empty()) {
        return pcl::PolygonMesh::Ptr(new pcl::PolygonMesh());
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width  = static_cast<int>(triangles.size());
    cloud.height = 1;
    triangles.download(cloud.points);

    pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
    pcl::toPCLPointCloud2(cloud, mesh->cloud);

    mesh->polygons.resize(triangles.size() / 3);
    for (size_t i = 0; i < mesh->polygons.size(); ++i) {
        pcl::Vertices v;
        v.vertices.push_back(i * 3 + 0);
        v.vertices.push_back(i * 3 + 1);
        v.vertices.push_back(i * 3 + 2);
        mesh->polygons[i] = v;
    }

    return mesh;
}
