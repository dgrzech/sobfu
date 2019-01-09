#include <kfusion/cuda/device.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Depth bilateral filter

namespace kfusion {
namespace device {
__global__ void bilateral_kernel(const PtrStepSz<ushort> src, PtrStep<ushort> dst, const int ksz,
                                 const float sigma_spatial2_inv_half, const float sigma_depth2_inv_half) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)
        return;

    int value = src(y, x);

    int tx = min(x - ksz / 2 + ksz, src.cols - 1);
    int ty = min(y - ksz / 2 + ksz, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max(y - ksz / 2, 0); cy < ty; ++cy) {
        for (int cx = max(x - ksz / 2, 0); cx < tx; ++cx) {
            int depth = src(cy, cx);

            float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            float color2 = (value - depth) * (value - depth);

            float weight = __expf(-(space2 * sigma_spatial2_inv_half + color2 * sigma_depth2_inv_half));

            sum1 += depth * weight;
            sum2 += weight;
        }
    }
    dst(y, x) = __float2int_rn(sum1 / sum2);
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::bilateralFilter(const Depth& src, Depth& dst, int kernel_size, float sigma_spatial,
                                      float sigma_depth) {
    sigma_depth *= 1000;  // meters -> mm

    dim3 block(64, 16);
    dim3 grid(divUp(src.cols(), block.x), divUp(src.rows(), block.y));

    cudaSafeCall(cudaFuncSetCacheConfig(bilateral_kernel, cudaFuncCachePreferL1));
    bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial),
                                      0.5f / (sigma_depth * sigma_depth));
    cudaSafeCall(cudaGetLastError());
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Depth truncation

namespace kfusion {
namespace device {
__global__ void truncate_depth_kernel(PtrStepSz<ushort> depth, ushort max_dist /*mm*/) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < depth.cols && y < depth.rows)
        if (depth(y, x) > max_dist)
            depth(y, x) = 0;
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::truncateDepth(Depth& depth, float max_dist /*meters*/) {
    dim3 block(64, 16);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));

    truncate_depth_kernel<<<grid, block>>>(depth, static_cast<ushort>(max_dist * 1000.f));
    cudaSafeCall(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Build depth pyramid

namespace kfusion {
namespace device {
__global__ void pyramid_kernel(const PtrStepSz<ushort> src, PtrStepSz<ushort> dst, float sigma_depth_mult3) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;
    int center  = src(2 * y, 2 * x);

    int tx = min(2 * x - D / 2 + D, src.cols - 1);
    int ty = min(2 * y - D / 2 + D, src.rows - 1);
    int cy = max(0, 2 * y - D / 2);

    int sum   = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max(0, 2 * x - D / 2); cx < tx; ++cx) {
            int val = src(cy, cx);
            if (abs(val - center) < sigma_depth_mult3) {
                sum += val;
                ++count;
            }
        }
    dst(y, x) = (count == 0) ? 0 : sum / count;
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::depthPyr(const Depth& source, Depth& pyramid, float sigma_depth) {
    sigma_depth *= 1000;  // meters -> mm

    dim3 block(64, 16);
    dim3 grid(divUp(pyramid.cols(), block.x), divUp(pyramid.rows(), block.y));

    pyramid_kernel<<<grid, block>>>(source, pyramid, sigma_depth * 3);
    cudaSafeCall(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute normals

namespace kfusion {
namespace device {
__global__ void compute_normals_kernel(const PtrStepSz<ushort> depth, const Reprojector reproj,
                                       PtrStep<Normal> normals) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    const float qnan = numeric_limits<float>::quiet_NaN();

    Normal n_out = make_float4(qnan, qnan, qnan, 0.f);

    if (x < depth.cols - 1 && y < depth.rows - 1) {
        // mm -> meters
        float z00 = depth(y, x) * 0.001f;
        float z01 = depth(y, x + 1) * 0.001f;
        float z10 = depth(y + 1, x) * 0.001f;

        if (z00 * z01 * z10 != 0) {
            float3 v00 = reproj(x, y, z00);
            float3 v01 = reproj(x + 1, y, z01);
            float3 v10 = reproj(x, y + 1, z10);

            float3 n = normalized(cross(v01 - v00, v10 - v00));
            n_out    = make_float4(-n.x, -n.y, -n.z, 0.f);
        }
    }
    normals(y, x) = n_out;
}

__global__ void mask_depth_kernel(const PtrStep<Normal> normals, PtrStepSz<ushort> depth) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < depth.cols || y < depth.rows) {
        float4 n = normals(y, x);
        if (isnan(n.x))
            depth(y, x) = 0;
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::computeNormalsAndMaskDepth(const Reprojector& reproj, Depth& depth, Normals& normals) {
    dim3 block(64, 16);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));

    compute_normals_kernel<<<grid, block>>>(depth, reproj, normals);
    cudaSafeCall(cudaGetLastError());

    mask_depth_kernel<<<grid, block>>>(normals, depth);
    cudaSafeCall(cudaGetLastError());
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute computePointNormals

namespace kfusion {
namespace device {
__global__ void points_normals_kernel(const Reprojector reproj, const PtrStepSz<ushort> depth, PtrStep<Point> points,
                                      PtrStep<Normal> normals) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    const float qnan = numeric_limits<float>::quiet_NaN();
    points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

    if (x >= depth.cols - 1 || y >= depth.rows - 1)
        return;

    // mm -> meters
    float z00 = depth(y, x) * 0.001f;
    float z01 = depth(y, x + 1) * 0.001f;
    float z10 = depth(y + 1, x) * 0.001f;

    if (z00 * z01 * z10 != 0) {
        float3 v00 = reproj(x, y, z00);
        float3 v01 = reproj(x + 1, y, z01);
        float3 v10 = reproj(x, y + 1, z10);

        float3 n      = normalized(cross(v01 - v00, v10 - v00));
        normals(y, x) = make_float4(-n.x, -n.y, -n.z, 0.f);
        points(y, x)  = make_float4(v00.x, v00.y, v00.z, 0.f);
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::computePointNormals(const Reprojector& reproj, const Depth& depth, Points& points,
                                          Normals& normals) {
    dim3 block(64, 16);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));

    points_normals_kernel<<<grid, block>>>(reproj, depth, points, normals);
    cudaSafeCall(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute dists

namespace kfusion {
namespace device {
__global__ void compute_dists_kernel(const PtrStepSz<ushort> depth, Dists dists, float2 finv, float2 c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < depth.cols || y < depth.rows) {
        float xl     = (x - c.x) * finv.x;
        float yl     = (y - c.y) * finv.y;
        float lambda = sqrtf(xl * xl + yl * yl + 1);

        dists(y, x) = depth(y, x) * lambda * 0.001f;  // meters
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::compute_dists(const Depth& depth, Dists dists, float2 f, float2 c) {
    dim3 block(64, 16);
    dim3 grid(divUp(depth.cols(), block.x), divUp(depth.rows(), block.y));

    compute_dists_kernel<<<grid, block>>>(depth, dists, make_float2(1.f / f.x, 1.f / f.y), c);
    cudaSafeCall(cudaGetLastError());
}

namespace kfusion {
namespace device {
__global__ void resize_depth_normals_kernel(const PtrStep<ushort> dsrc, const PtrStep<float4> nsrc,
                                            PtrStepSz<ushort> ddst, PtrStep<float4> ndst) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= ddst.cols || y >= ddst.rows)
        return;

    const float qnan = numeric_limits<float>::quiet_NaN();

    ushort d = 0;
    float4 n = make_float4(qnan, qnan, qnan, qnan);

    int xs = x * 2;
    int ys = y * 2;

    int d00 = dsrc(ys + 0, xs + 0);
    int d01 = dsrc(ys + 0, xs + 1);
    int d10 = dsrc(ys + 1, xs + 0);
    int d11 = dsrc(ys + 1, xs + 1);

    if (d00 * d01 != 0 && d10 * d11 != 0) {
        d = (d00 + d01 + d10 + d11) / 4;

        float4 n00 = nsrc(ys + 0, xs + 0);
        float4 n01 = nsrc(ys + 0, xs + 1);
        float4 n10 = nsrc(ys + 1, xs + 0);
        float4 n11 = nsrc(ys + 1, xs + 1);

        n.x = (n00.x + n01.x + n10.x + n11.x) * 0.25;
        n.y = (n00.y + n01.y + n10.y + n11.y) * 0.25;
        n.z = (n00.z + n01.z + n10.z + n11.z) * 0.25;
    }
    ddst(y, x) = d;
    ndst(y, x) = n;
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out,
                                         Normals& normals_out) {
    int in_cols = depth.cols();
    int in_rows = depth.rows();

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    dim3 block(64, 16);
    dim3 grid(divUp(out_cols, block.x), divUp(out_rows, block.y));

    resize_depth_normals_kernel<<<grid, block>>>(depth, normals, depth_out, normals_out);
    cudaSafeCall(cudaGetLastError());
}

namespace kfusion {
namespace device {
__global__ void resize_points_normals_kernel(const PtrStep<Point> vsrc, const PtrStep<Normal> nsrc,
                                             PtrStepSz<Point> vdst, PtrStep<Normal> ndst) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= vdst.cols || y >= vdst.rows)
        return;

    const float qnan = numeric_limits<float>::quiet_NaN();
    vdst(y, x) = ndst(y, x) = make_float4(qnan, qnan, qnan, 0.f);

    int xs = x * 2;
    int ys = y * 2;

    float3 d00 = tr(vsrc(ys + 0, xs + 0));
    float3 d01 = tr(vsrc(ys + 0, xs + 1));
    float3 d10 = tr(vsrc(ys + 1, xs + 0));
    float3 d11 = tr(vsrc(ys + 1, xs + 1));

    if (!isnan(d00.x * d01.x * d10.x * d11.x)) {
        float3 d   = (d00 + d01 + d10 + d11) * 0.25f;
        vdst(y, x) = make_float4(d.x, d.y, d.z, 0.f);

        float3 n00 = tr(nsrc(ys + 0, xs + 0));
        float3 n01 = tr(nsrc(ys + 0, xs + 1));
        float3 n10 = tr(nsrc(ys + 1, xs + 0));
        float3 n11 = tr(nsrc(ys + 1, xs + 1));

        float3 n   = (n00 + n01 + n10 + n11) * 0.25f;
        ndst(y, x) = make_float4(n.x, n.y, n.z, 0.f);
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::resizePointsNormals(const Points& points, const Normals& normals, Points& points_out,
                                          Normals& normals_out) {
    int out_cols = points.cols() / 2;
    int out_rows = points.rows() / 2;

    dim3 block(64, 16);
    dim3 grid(divUp(out_cols, block.x), divUp(out_rows, block.y));

    resize_points_normals_kernel<<<grid, block>>>(points, normals, points_out, normals_out);
    cudaSafeCall(cudaGetLastError());
}

namespace kfusion {
namespace device {
/* calculate for the projected triangle the bounding box in the image domain */
__host__ __device__ void get_bounding_box(float2 v1, float2 v2, float2 v3, int2& min, int2& max) {
    min.x = static_cast<int>(fmin(v1.x, fmin(v2.x, v3.x)));
    min.y = static_cast<int>(fmin(v1.y, fmin(v2.y, v3.y)));

    max.x = static_cast<int>(fmax(v1.x, fmax(v2.x, v3.x)));
    max.y = static_cast<int>(fmax(v1.y, fmax(v2.y, v3.y)));
}

__host__ __device__ __forceinline__ float edge_function(const float2& a, const float2& b, const float2& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

/* rasterise surface triangles */
__global__ void rasterise_surface_kernel(const Projector proj, const Aff3f vol2cam, const PtrSz<Point> vsrc,
                                         const PtrSz<Normal> nsrc, PtrStepSz<Point> points_out,
                                         PtrStep<Normal> normals_out) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 3;

    if ((x + 2) >= vsrc.size)
        return;

    /* get vertices and normals */
    float3 v1 = vol2cam * tr(vsrc[x]);
    float3 v2 = vol2cam * tr(vsrc[x + 1]);
    float3 v3 = vol2cam * tr(vsrc[x + 2]);

    /* project vertices onto the image plane */
    float2 coos1 = proj(v1);
    float2 coos2 = proj(v2);
    float2 coos3 = proj(v3);

    /* get 2-d triangle bounding box */
    int2 min;
    int2 max;
    get_bounding_box(coos1, coos2, coos3, min, max);

    /* check for validity of coordinates */
    if (min.x < 0 || min.y < 0 || max.x >= points_out.cols || max.y >= points_out.rows)
        return;

    /* used for smooth interpoation */
    float area = edge_function(coos1, coos2, coos3);

    /* shade pixels */
    for (int i = min.x; i < max.x; i++) {
        for (int j = min.y; j < max.y; j++) {
            /* coordinates of the centre of the pixel */
            float2 p = make_float2(i + 0.5f, j + 0.5f);

            float w0 = edge_function(coos2, coos3, p) / area;
            float w1 = edge_function(coos3, coos1, p) / area;
            float w2 = edge_function(coos1, coos2, p) / area;

            float z = w0 * v1.z + w1 * v2.z + w2 * v3.z;

            if (z < points_out(j, i).z || fabs(points_out(j, i).z) < 1e-7f) {
                points_out(j, i) =
                    make_float4(w0 * v1.x + w1 * v2.x + w2 * v3.x, w0 * v1.y + w1 * v2.y + w2 * v3.y, z, 0.f);
            }
        }
    }

    for (int i = min.x; i < max.x; i++) {
        for (int j = min.y; j < max.y; j++) {
            float3 v0 = tr(points_out(j, i));
            float3 v1 = tr(points_out(j + 1, i));
            float3 v2 = tr(points_out(j, i + 1));

            float3 n          = normalized(cross(v1 - v0, v2 - v0));
            normals_out(j, i) = make_float4(n.x, n.y, n.z, 1.f);
        }
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::rasteriseSurface(const Projector& proj, const Aff3f& vol2cam, const Surface& surface,
                                       Points& points_out, Normals& normals_out) {
    dim3 block(256);
    dim3 grid(divUp(surface.vertices.size() / 3, block.x));

    rasterise_surface_kernel<<<grid, block>>>(proj, vol2cam, surface.vertices, surface.normals, points_out,
                                              normals_out);
    cudaSafeCall(cudaGetLastError());
}
