#include <kfusion/precomp.hpp>

void kfusion::cuda::depthBilateralFilter(const Depth &in, Depth &out, int kernel_size, float sigma_spatial,
                                         float sigma_depth) {
    out.create(in.rows(), in.cols());
    device::bilateralFilter(in, out, kernel_size, sigma_spatial, sigma_depth);
}

void kfusion::cuda::depthTruncation(Depth &depth, float threshold) { device::truncateDepth(depth, threshold); }

void kfusion::cuda::depthBuildPyramid(const Depth &depth, Depth &pyramid, float sigma_depth) {
    pyramid.create(depth.rows() / 2, depth.cols() / 2);
    device::depthPyr(depth, pyramid, sigma_depth);
}

void kfusion::cuda::waitAllDefaultStream() { cudaSafeCall(cudaDeviceSynchronize()); }

void kfusion::cuda::computeNormalsAndMaskDepth(const Intr &intr, Depth &depth, Normals &normals) {
    normals.create(depth.rows(), depth.cols());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Normals &n = (device::Normals &) normals;
    device::computeNormalsAndMaskDepth(reproj, depth, n);
}

void kfusion::cuda::computePointNormals(const Intr &intr, const Depth &depth, Cloud &points, Normals &normals) {
    points.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Points &p  = (device::Points &) points;
    device::Normals &n = (device::Normals &) normals;
    device::computePointNormals(reproj, depth, p, n);
}

void kfusion::cuda::computeDists(const Depth &depth, Dists &dists, const Intr &intr) {
    dists.create(depth.rows(), depth.cols());
    device::compute_dists(depth, dists, make_float2(intr.fx, intr.fy), make_float2(intr.cx, intr.cy));
}

void kfusion::cuda::resizeDepthNormals(const Depth &depth, const Normals &normals, Depth &depth_out,
                                       Normals &normals_out) {
    depth_out.create(depth.rows() / 2, depth.cols() / 2);
    normals_out.create(normals.rows() / 2, normals.cols() / 2);

    device::Normals &nsrc = (device::Normals &) normals;
    device::Normals &ndst = (device::Normals &) normals_out;

    device::resizeDepthNormals(depth, nsrc, depth_out, ndst);
}

void kfusion::cuda::resizePointsNormals(const Cloud &points, const Normals &normals, Cloud &points_out,
                                        Normals &normals_out) {
    points_out.create(points.rows() / 2, points.cols() / 2);
    normals_out.create(normals.rows() / 2, normals.cols() / 2);

    device::Points &pi  = (device::Points &) points;
    device::Normals &ni = (device::Normals &) normals;

    device::Points &po  = (device::Points &) points_out;
    device::Normals &no = (device::Normals &) normals_out;

    device::resizePointsNormals(pi, ni, po, no);
}

void kfusion::cuda::rasteriseSurface(const Intr &intr, const Affine3f &vol2cam, const Surface &s, Cloud &vertices_out,
                                     Normals &normals_out) {
    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);
    device::Aff3f dev_vol2cam = device_cast<device::Aff3f>(vol2cam);

    /* convert warped triangle vertices & normals to point clouds */
    pcl::PointCloud<pcl::PointXYZ> vertices_warped;
    s.vertices.download(vertices_warped.points);

    pcl::PointCloud<pcl::Normal> normals_warped;
    s.normals.download(normals_warped.points);

    /* lay out the warped triangle vertices and normals in memory */
    std::vector<pcl::PointXYZ>
        triangle_vertices; /* the index of each entry in the vector, modulo 3, stores the 1st, 2nd, and 3rd vertex of a
                             triangle */
    std::vector<pcl::Normal> triangle_normals;
    for (size_t i = 0; i < vertices_warped.size() / 3; i++) {
        triangle_vertices.emplace_back(vertices_warped[i * 3 + 0]);
        triangle_vertices.emplace_back(vertices_warped[i * 3 + 1]);
        triangle_vertices.emplace_back(vertices_warped[i * 3 + 2]);

        triangle_normals.emplace_back(normals_warped[i * 3 + 0]);
        triangle_normals.emplace_back(normals_warped[i * 3 + 1]);
        triangle_normals.emplace_back(normals_warped[i * 3 + 2]);
    }

    kfusion::cuda::Vertices vertices;
    vertices.upload(triangle_vertices);

    kfusion::cuda::Norms normals;
    normals.upload(triangle_normals);

    kfusion::cuda::Surface surface;
    surface.vertices = vertices;
    surface.normals  = normals;

    /* convert to classes that can be used in cuda */
    const device::Surface &dev_surface = (const device::Surface &) surface;
    device::Points &vo                 = (device::Points &) vertices_out;
    device::Normals &no                = (device::Normals &) normals_out;

    device::rasteriseSurface(proj, dev_vol2cam, dev_surface, vo, no);
    waitAllDefaultStream();
}
