#include <kfusion/cuda/device.hpp>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define FULL_MASK 0xffffffff

namespace kfusion {
namespace device {
// texture<int, 1, cudaReadModeElementType> edgeTex;
texture<int, 1, cudaReadModeElementType> triTex;
texture<int, 1, cudaReadModeElementType> numVertsTex;
}  // namespace device
}  // namespace kfusion

void kfusion::device::bindTextures(const int* /*edgeBuf*/, const int* triBuf, const int* numVertsBuf) {
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    // cudaSafeCall(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
    cudaSafeCall(cudaBindTexture(0, triTex, triBuf, desc));
    cudaSafeCall(cudaBindTexture(0, numVertsTex, numVertsBuf, desc));
}
void kfusion::device::unbindTextures() {
    // cudaSafeCall( cudaUnbindTexture(edgeTex) );
    cudaSafeCall(cudaUnbindTexture(numVertsTex));
    cudaSafeCall(cudaUnbindTexture(triTex));
}

namespace kfusion {
namespace device {
__device__ int global_count = 0;
__device__ int output_count;
__device__ unsigned int blocks_done = 0;

__kf_device__ void kfusion::device::CubeIndexEstimator::readTsdf(int x, int y, int z, float& f, float& weight) const {
    float2 aux = *volume(x, y, z);
    f          = aux.x;
    weight     = aux.y;
}

__kf_device__ int kfusion::device::CubeIndexEstimator::computeCubeIndex(int x, int y, int z, float f[8]) const {
    float weight;
    readTsdf(x, y, z, f[0], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x + 1, y, z, f[1], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x + 1, y + 1, z, f[2], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x, y + 1, z, f[3], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x, y, z + 1, f[4], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x + 1, y, z + 1, f[5], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x + 1, y + 1, z + 1, f[6], weight);
    if (weight == 0.f)
        return 0;
    readTsdf(x, y + 1, z + 1, f[7], weight);
    if (weight == 0.f)
        return 0;

    // calculate flag indicating if each vertex is inside or outside isosurface
    int cubeindex = 0;
    cubeindex     = int(f[0] < isoValue);
    cubeindex += int(f[1] < isoValue) * 2;
    cubeindex += int(f[2] < isoValue) * 4;
    cubeindex += int(f[3] < isoValue) * 8;
    cubeindex += int(f[4] < isoValue) * 16;
    cubeindex += int(f[5] < isoValue) * 32;
    cubeindex += int(f[6] < isoValue) * 64;
    cubeindex += int(f[7] < isoValue) * 128;

    return cubeindex;
}

__kf_device__ void kfusion::device::OccupiedVoxels::operator()() const {
    int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
    int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

    if (__all_sync(FULL_MASK, x >= volume.dims.x) || __all_sync(FULL_MASK, y >= volume.dims.y)) {
        return;
    }

    int ftid    = Block::flattenedThreadId();
    int warp_id = Warp::id();
    int lane_id = Warp::laneId();

    volatile __shared__ int warps_buffer[WARPS_COUNT];

    for (int z = 0; z < volume.dims.z - 1; z++) {
        int numVerts = 0;
        ;
        if (x + 1 < volume.dims.x && y + 1 < volume.dims.y) {
            float field[8];
            int cubeindex = computeCubeIndex(x, y, z, field);

            // read number of vertices from texture
            numVerts = (cubeindex == 0 || cubeindex == 255) ? 0 : tex1Dfetch(numVertsTex, cubeindex);
        }

        int total = __popc(__ballot_sync(FULL_MASK, numVerts > 0));

        if (total == 0)
            continue;

        if (lane_id == 0) {
            int old               = atomicAdd(&global_count, total);
            warps_buffer[warp_id] = old;
        }
        int old_global_voxels_count = warps_buffer[warp_id];

        int offs = Warp::binaryExclScan(__ballot_sync(FULL_MASK, numVerts > 0));

        if (old_global_voxels_count + offs < max_size && numVerts > 0) {
            voxels_indices[old_global_voxels_count + offs]  = volume.dims.y * volume.dims.x * z + volume.dims.x * y + x;
            vertices_number[old_global_voxels_count + offs] = numVerts;
        }

        bool full = old_global_voxels_count + total >= max_size;

        if (full)
            break;

    } /* for(int z = 0; z < 128 - 1; z++) */

    /////////////////////////
    // prepare for future scans
    if (ftid == 0) {
        unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
        unsigned int value        = atomicInc(&blocks_done, total_blocks);

        // last block
        if (value == total_blocks - 1) {
            output_count = min(max_size, global_count);
            blocks_done  = 0;
            global_count = 0;
        }
    }
} /* operator () */

__global__ void getOccupiedVoxelsKernel(const OccupiedVoxels ov) { ov(); }

int getOccupiedVoxels(const TsdfVolume& volume, DeviceArray2D<int>& occupied_voxels) {
    OccupiedVoxels ov(volume);

    ov.voxels_indices  = occupied_voxels.ptr(0);
    ov.vertices_number = occupied_voxels.ptr(1);
    ov.max_size        = occupied_voxels.cols();

    dim3 block(OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    getOccupiedVoxelsKernel<<<grid, block>>>(ov);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    int size;
    cudaSafeCall(cudaMemcpyFromSymbol(&size, output_count, sizeof(size)));
    return size;
}

int computeOffsetsAndTotalVertices(DeviceArray2D<int>& occupied_voxels) {
    thrust::device_ptr<int> beg(occupied_voxels.ptr(1));
    thrust::device_ptr<int> end = beg + occupied_voxels.cols();

    thrust::device_ptr<int> out(occupied_voxels.ptr(2));
    thrust::exclusive_scan(beg, end, out);

    int lastElement, lastScanElement;

    DeviceArray<int> last_elem(occupied_voxels.ptr(1) + occupied_voxels.cols() - 1, 1);
    DeviceArray<int> last_scan(occupied_voxels.ptr(2) + occupied_voxels.cols() - 1, 1);

    last_elem.download(&lastElement);
    last_scan.download(&lastScanElement);

    return lastElement + lastScanElement;
}

__kf_device__ float3 kfusion::device::TrianglesGenerator::get_node_coo(int x, int y, int z) const {
    float3 coo = make_float3(x, y, z);
    coo += 0.5f;  // shift to volume cell center;

    coo.x *= cell_size.x;
    coo.y *= cell_size.y;
    coo.z *= cell_size.z;

    return coo;
}

__kf_device__ float3 kfusion::device::TrianglesGenerator::vertex_interp(float3 p0, float3 p1, float f0,
                                                                        float f1) const {
    float t = (isoValue - f0) / (f1 - f0 + 1e-15f);
    float x = p0.x + t * (p1.x - p0.x);
    float y = p0.y + t * (p1.y - p0.y);
    float z = p0.z + t * (p1.z - p0.z);
    return make_float3(x, y, z);
}

__kf_device__ void kfusion::device::TrianglesGenerator::operator()() const {
    int tid = threadIdx.x;
    int idx = (blockIdx.y * MAX_GRID_SIZE_X + blockIdx.x) * CTA_SIZE + tid;

    if (idx >= voxels_count)
        return;

    int voxel = occupied_voxels[idx];

    int z = voxel / (volume.dims.x * volume.dims.y);
    int y = (voxel - z * volume.dims.x * volume.dims.y) / volume.dims.x;
    int x = (voxel - z * volume.dims.x * volume.dims.y) - y * volume.dims.x;

    float f[8];
    int cubeindex = computeCubeIndex(x, y, z, f);

    /* calculate cell vertex positions */
    float3 v[8];
    v[0] = get_node_coo(x, y, z);
    v[1] = get_node_coo(x + 1, y, z);
    v[2] = get_node_coo(x + 1, y + 1, z);
    v[3] = get_node_coo(x, y + 1, z);
    v[4] = get_node_coo(x, y, z + 1);
    v[5] = get_node_coo(x + 1, y, z + 1);
    v[6] = get_node_coo(x + 1, y + 1, z + 1);
    v[7] = get_node_coo(x, y + 1, z + 1);

    /* find vertices where surface intersects the cube; use shared memory to avoid using local */
    __shared__ float3 vertlist[12][CTA_SIZE];

    vertlist[0][tid]  = vertex_interp(v[0], v[1], f[0], f[1]);
    vertlist[1][tid]  = vertex_interp(v[1], v[2], f[1], f[2]);
    vertlist[2][tid]  = vertex_interp(v[2], v[3], f[2], f[3]);
    vertlist[3][tid]  = vertex_interp(v[3], v[0], f[3], f[0]);
    vertlist[4][tid]  = vertex_interp(v[4], v[5], f[4], f[5]);
    vertlist[5][tid]  = vertex_interp(v[5], v[6], f[5], f[6]);
    vertlist[6][tid]  = vertex_interp(v[6], v[7], f[6], f[7]);
    vertlist[7][tid]  = vertex_interp(v[7], v[4], f[7], f[4]);
    vertlist[8][tid]  = vertex_interp(v[0], v[4], f[0], f[4]);
    vertlist[9][tid]  = vertex_interp(v[1], v[5], f[1], f[5]);
    vertlist[10][tid] = vertex_interp(v[2], v[6], f[2], f[6]);
    vertlist[11][tid] = vertex_interp(v[3], v[7], f[3], f[7]);
    __syncthreads();

    /* output triangle vertices and normals */
    int numVerts = tex1Dfetch(numVertsTex, cubeindex);

    for (int i = 0; i < numVerts; i += 3) {
        int index = vertex_ofssets[idx] + i;

        int v1 = tex1Dfetch(triTex, (cubeindex * 16) + i + 0);
        int v2 = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
        int v3 = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);

        /* NOTE (dig15): the surface could be smoother if the normal weren't the same for each vertex of the triangle */
        float3 n = normalized(cross(vertlist[v3][tid] - vertlist[v1][tid], vertlist[v2][tid] - vertlist[v1][tid]));

        store_point(outputVertices, index + 0, pose * vertlist[v1][tid]);
        store_point(outputNormals, index + 0, n);

        store_point(outputVertices, index + 1, pose * vertlist[v2][tid]);
        store_point(outputNormals, index + 1, n);

        store_point(outputVertices, index + 2, pose * vertlist[v3][tid]);
        store_point(outputNormals, index + 2, n);
    }
}

__kf_device__ void kfusion::device::TrianglesGenerator::store_point(float4* ptr, int index,
                                                                    const float3& vertex) const {
    ptr[index] = make_float4(vertex.x, -vertex.y, -vertex.z, 1.f);
}

__global__ void trianglesGeneratorKernel(const TrianglesGenerator tg) { tg(); }

void generateTriangles(const TsdfVolume& volume, const DeviceArray2D<int>& occupied_voxels, const float3& volume_size,
                       const Aff3f& pose, DeviceArray<PointType>& outputVertices,
                       DeviceArray<PointType>& outputNormals) {
    int device;
    cudaSafeCall(cudaGetDevice(&device));

    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, device));

    typedef TrianglesGenerator Tg;
    Tg tg(volume);

    tg.occupied_voxels = occupied_voxels.ptr(0);
    tg.vertex_ofssets  = occupied_voxels.ptr(2);
    tg.voxels_count    = occupied_voxels.cols();
    tg.cell_size.x     = volume_size.x / volume.dims.x;
    tg.cell_size.y     = volume_size.y / volume.dims.y;
    tg.cell_size.z     = volume_size.z / volume.dims.z;
    tg.outputVertices  = outputVertices;
    tg.outputNormals   = outputNormals;

    tg.pose = pose;

    int block_size = 256;
    int blocks_num = divUp(tg.voxels_count, block_size);

    dim3 block(block_size);
    dim3 grid(min(blocks_num, Tg::MAX_GRID_SIZE_X), divUp(blocks_num, Tg::MAX_GRID_SIZE_X));

    trianglesGeneratorKernel<<<grid, block>>>(tg);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}
}  // namespace device
}  // namespace kfusion
