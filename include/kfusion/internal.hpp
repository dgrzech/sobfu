#pragma once

/* kinfu includes */
#include <kfusion/cuda/device_array.hpp>
#include <kfusion/safe_call.hpp>

/* pcl includes */
#include <pcl/point_types.h>

//#define USE_DEPTH

struct Mat4f {
    float4 data[4];
};

namespace kfusion {
namespace device {
typedef float4 Normal;
typedef float4 Point;

typedef unsigned short ushort;
typedef unsigned char uchar;

typedef PtrStepSz<float> Dists;
typedef DeviceArray2D<ushort> Depth;
typedef DeviceArray2D<Normal> Normals;
typedef DeviceArray2D<Point> Points;
typedef DeviceArray2D<uchar4> Image;

typedef float3 *Displacements;

typedef DeviceArray<float4> Vertices;
typedef Vertices Norms;

/*
 * struct which holds a surface as triangles
 */
struct Surface {
    Vertices vertices;
    Norms normals;
};

struct float8 {
    float x, y, z, w, c1, c2, c3, c4;
};

typedef float4 PointType;

typedef int3 Vec3i;
typedef float3 Vec3f;
struct Mat3f {
    float3 data[3];
};
struct Aff3f {
    Mat3f R;
    Vec3f t;
};

struct TsdfVolume {
public:
    float2 *const data;
    const int3 dims;
    const float3 voxel_size;
    const float trunc_dist;
    const float eta;
    const float max_weight;

    TsdfVolume(float2 *const data, int3 dims, float3 voxel_size, float trunc_dist, float eta, float max_weight);
    // TsdfVolume(const TsdfVolume &);

    TsdfVolume &operator=(const TsdfVolume &);

    __kf_device__ float2 *operator()(int x, int y, int z);
    __kf_device__ const float2 *operator()(int x, int y, int z) const;

    __kf_device__ float2 *beg(int x, int y) const;
    __kf_device__ float2 *zstep(float2 *const ptr) const;
};

struct Projector {
    float2 f, c;
    Projector() {}
    Projector(float fx, float fy, float cx, float cy);
    __kf_device__ float2 operator()(const float3 &p) const;
};

struct Reprojector {
    Reprojector() {}
    Reprojector(float fx, float fy, float cx, float cy);
    float2 finv, c;
    __kf_device__ float3 operator()(int x, int y, float z) const;
};

struct CubeIndexEstimator {
    CubeIndexEstimator(const TsdfVolume &vol) : volume(vol) { isoValue = 0.f; }

    const TsdfVolume volume;
    float isoValue;

    __kf_device__ int computeCubeIndex(int x, int y, int z, float f[8]) const;
    __kf_device__ void readTsdf(int x, int y, int z, float &f, float &weight) const;
};

struct OccupiedVoxels : public CubeIndexEstimator {
    OccupiedVoxels(const TsdfVolume &vol) : CubeIndexEstimator({vol}) {}

    enum {
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,
        CTA_SIZE   = CTA_SIZE_X * CTA_SIZE_Y,

        LOG_WARP_SIZE = 5,
        WARP_SIZE     = 1 << LOG_WARP_SIZE,
        WARPS_COUNT   = CTA_SIZE / WARP_SIZE,
    };

    mutable int *voxels_indices;
    mutable int *vertices_number;
    int max_size;

    __kf_device__ void operator()() const;
};

struct TrianglesGenerator : public CubeIndexEstimator {
    TrianglesGenerator(const TsdfVolume &vol) : CubeIndexEstimator({vol}) {}

    enum { CTA_SIZE = 256, MAX_GRID_SIZE_X = 65536 };

    const int *occupied_voxels;
    const int *vertex_ofssets;
    int voxels_count;
    float3 cell_size;

    Aff3f pose;

    mutable device::PointType *outputVertices;
    mutable device::PointType *outputNormals;

    __kf_device__ void operator()() const;

    __kf_device__ void store_point(float4 *ptr, int index, const float3 &vertex) const;

    __kf_device__ float3 get_node_coo(int x, int y, int z) const;
    __kf_device__ float3 vertex_interp(float3 p0, float3 p1, float f0, float f1) const;

};  // namespace device

struct ComputeIcpHelper {
    struct Policy;
    struct PageLockHelper {
        float *data;
        PageLockHelper();
        ~PageLockHelper();
    };

    float min_cosine;
    float dist2_thres;

    Aff3f aff;

    float rows, cols;
    float2 f, c, finv;

    PtrStep<ushort> dcurr;
    PtrStep<Normal> ncurr;
    PtrStep<Point> vcurr;

    ComputeIcpHelper(float dist_thres, float angle_thres);
    void setLevelIntr(int level_index, float fx, float fy, float cx, float cy);

    void operator()(const Depth &dprev, const Normals &nprev, DeviceArray2D<float> &buffer, float *data,
                    cudaStream_t stream);
    void operator()(const Points &vprev, const Normals &nprev, DeviceArray2D<float> &buffer, float *data,
                    cudaStream_t stream);

    static void allocate_buffer(DeviceArray2D<float> &buffer, int partials_count = -1);

    // private:
    __kf_device__ int find_coresp(int x, int y, float3 &n, float3 &d, float3 &s) const;
    __kf_device__ void partial_reduce(const float row[7], PtrStep<float> &partial_buffer) const;
    __kf_device__ float2 proj(const float3 &p) const;
    __kf_device__ float3 reproj(float x, float y, float z) const;
};

/*
 * TSDF FUNCTIONS
 */

void clear_volume(TsdfVolume &volume);

void integrate(TsdfVolume &phi_global, TsdfVolume &phi_n_psi);
void integrate(const Dists &depth, TsdfVolume &volume, const Aff3f &aff, const Projector &proj);

void init_box(TsdfVolume &volume, const float3 &b);
void init_ellipsoid(TsdfVolume &volume, const float3 &r);
void init_plane(TsdfVolume &volume, const float &z);
void init_sphere(TsdfVolume &volume, const float3 &centre, const float &radius);
void init_torus(TsdfVolume &volume, const float2 &t);

/*
 * MARCHING CUBES FUNCTIONS
 */

/** \brief binds marching cubes tables to texture references */
void bindTextures(const int *edgeBuf, const int *triBuf, const int *numVertsBuf);

/** \brief unbinds */
void unbindTextures();

/** \brief scans TSDF volume and retrieves occuped voxes
 * \param[in] volume TSDF volume
 * \param[out] occupied_voxels buffer for occupied voxels; the function fills the first row with voxel id's and second
 * row with the no. of vertices
 * \return number of voxels in the buffer
 */
int getOccupiedVoxels(const TsdfVolume &volume, DeviceArray2D<int> &occupied_voxels);

/** \brief computes total number of vertices for all voxels and offsets of vertices in final triangle array
 * \param[out] occupied_voxels buffer with occupied voxels; the function fills 3rd only with offsets
 * \return total number of vertexes
 */
int computeOffsetsAndTotalVertices(DeviceArray2D<int> &occupied_voxels);

/** \brief generates final triangle array
 * \param[in] volume TSDF volume
 * \param[in] occupied_voxels occupied voxel ids (1st row), number of vertices (2nd row), offsets (3rd row).
 * \param[in] volume_size volume size in meters
 * \param[out] output triangle array
 */
void generateTriangles(const TsdfVolume &volume, const DeviceArray2D<int> &occupied_voxels, const float3 &volume_size,
                       const Aff3f &pose, DeviceArray<PointType> &outputVertices,
                       DeviceArray<PointType> &outputNormals);

/*
 *  IMAGE PROCESSING FUNCTIONS
 */
void compute_dists(const Depth &depth, Dists dists, float2 f, float2 c);

void truncateDepth(Depth &depth, float max_dist /*meters*/);
void bilateralFilter(const Depth &src, Depth &dst, int kernel_size, float sigma_spatial, float sigma_depth);
void depthPyr(const Depth &source, Depth &pyramid, float sigma_depth);

void resizeDepthNormals(const Depth &depth, const Normals &normals, Depth &depth_out, Normals &normals_out);
void resizePointsNormals(const Points &points, const Normals &normals, Points &points_out, Normals &normals_out);

void computeNormalsAndMaskDepth(const Reprojector &reproj, Depth &depth, Normals &normals);
void computePointNormals(const Reprojector &reproj, const Depth &depth, Points &points, Normals &normals);

/** \brief rasterises the surface extracted via marching cubes
 *  \param[in] proj
 *  \param[in] surface
 *  \param[out] vertices_out
 *  \paraim[out] normals_out
 */
void rasteriseSurface(const Projector &proj, const Aff3f &vol2cam, const Surface &surface, Points &vertices_out,
                      Normals &normals_out);

}  // namespace device
}  // namespace kfusion
