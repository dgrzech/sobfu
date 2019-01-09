#pragma once

/* sobfu includes */
#include <sobfu/params.hpp>
#include <sobfu/solver.hpp>

/* kinfu includes */
#include <kfusion/internal.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/precomp.hpp>

/* pcl includes */
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>

/* sys headers */
#include <math.h>
#include <thread>

/* */
class SobFusion {
public:
    /* default constructor */
    SobFusion(const Params &params);
    /* default destructor */
    ~SobFusion();

    /* get sobfu params */
    Params &getParams();

    /* get the canonical model vertices stored as pcl triangle mesh */
    pcl::PolygonMesh::Ptr get_phi_global_mesh();
    /* get the canonical model mesh warped to live */
    pcl::PolygonMesh::Ptr get_phi_global_psi_inv_mesh();
    /* get the live model mesh */
    pcl::PolygonMesh::Ptr get_phi_n_mesh();
    /* get the live model mesh warped with psi */
    pcl::PolygonMesh::Ptr get_phi_n_psi_mesh();

    /* get the deformation field */
    std::shared_ptr<sobfu::cuda::DeformationField> getDeformationField();

    /* run algorithm on all frames */
    bool operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image &image = kfusion::cuda::Image());

private:
    std::vector<cv::Affine3f> poses_;
    kfusion::cuda::Dists dists_;
    kfusion::cuda::Frame curr_, prev_;

    kfusion::cuda::Cloud points_;
    kfusion::cuda::Normals normals_;
    kfusion::cuda::Depth depths_;

    int frame_counter_;

    /* system parameters */
    Params params;

    /* tsdf's */
    cv::Ptr<kfusion::cuda::TsdfVolume> phi_global, phi_global_psi_inv, phi_n, phi_n_psi;
    /* deformation field warps phi_n to phi_global */
    std::shared_ptr<sobfu::cuda::DeformationField> psi, psi_inv;
    /* solver */
    std::shared_ptr<sobfu::cuda::Solver> solver;

    /* marching cubes */
    cv::Ptr<kfusion::cuda::MarchingCubes> mc;

    /* run marching cubes on vol */
    pcl::PolygonMesh::Ptr get_mesh(cv::Ptr<kfusion::cuda::TsdfVolume> vol);
    /* convert the canonical model to pcl polygon mesh */
    static pcl::PolygonMesh::Ptr convert_to_mesh(const kfusion::cuda::DeviceArray<pcl::PointXYZ> &triangles);
};
