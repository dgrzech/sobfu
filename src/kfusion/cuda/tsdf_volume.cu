/* kfusion includes */
#include <kfusion/cuda/device.hpp>
#include <kfusion/cuda/texture_binder.hpp>

/* sobfu includes */
#include <sobfu/cuda/utils.hpp>

/* cuda includes */
#include <curand.h>
#include <curand_kernel.h>

/* thrust includes */
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

using namespace kfusion::device;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume initialization

namespace kfusion {
namespace device {
__global__ void clear_volume_kernel(TsdfVolume tsdf) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < tsdf.dims.x && y < tsdf.dims.y) {
        float2* beg = tsdf.beg(x, y);
        float2* end = beg + tsdf.dims.x * tsdf.dims.y * tsdf.dims.z;

        for (float2* pos = beg; pos != end; pos = tsdf.zstep(pos))
            *pos = make_float2(0.f, 0.f);
    }
}
}  // namespace device
}  // namespace kfusion

void kfusion::device::clear_volume(TsdfVolume& volume) {
    dim3 block(64, 16);
    dim3 grid(1, 1, 1);
    grid.x = divUp(volume.dims.x, block.x);
    grid.y = divUp(volume.dims.y, block.y);

    clear_volume_kernel<<<grid, block>>>(volume);
    cudaSafeCall(cudaGetLastError());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume integration

namespace kfusion {
namespace device {
texture<float, 2> dists_tex(0, cudaFilterModePoint, cudaAddressModeBorder,
                            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));

struct TsdfIntegrator {
    Projector proj;
    Aff3f vol2cam;

    int2 dists_size;

    __kf_device__ void operator()(TsdfVolume& volume) const {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= volume.dims.x || y >= volume.dims.y) {
            return;
        }

        float3 vc     = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                                y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f);
        float3 vc_cam = vol2cam * vc; /* transfrom from volume coos to camera coos */

        float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);
        float2* vptr = volume.beg(x, y);
        for (int i = 0; i <= volume.dims.z - 1; ++i, vc_cam += zstep, vptr = volume.zstep(vptr)) {
            /* project the voxel centre onto the depth map */
            float2 coo = proj(vc_cam);
            if (coo.x < 0 || coo.y < 0 || coo.x >= dists_size.x || coo.y >= dists_size.y) {
                continue;
            }

            float Dp = tex2D(dists_tex, coo.x, coo.y);
            if (Dp <= 0.f || vc_cam.z <= 0) {
                continue;
            }

            /* get the psdf value */
            float psdf = Dp - vc_cam.z;
            /* get the weight */
            float weight = (psdf > -volume.eta) ? 1.f : 0.f;

            if (psdf >= volume.trunc_dist) {
                *vptr = make_float2(1.f, weight);
            } else if (psdf <= -volume.trunc_dist) {
                *vptr = make_float2(-1.f, weight);
            } else {
                *vptr = make_float2(__fdividef(psdf, volume.trunc_dist), weight);
            }
        }
    }

    __kf_device__ void operator()(TsdfVolume& phi_global, TsdfVolume& phi_n_psi) const {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= phi_global.dims.x || y >= phi_global.dims.y) {
            return;
        }

        float2* pos_global = phi_global.beg(x, y);
        float2* pos_n_psi  = phi_n_psi.beg(x, y);

        for (int i = 0; i <= phi_global.dims.z - 1;
             ++i, pos_global = phi_global.zstep(pos_global), pos_n_psi = phi_n_psi.zstep(pos_n_psi)) {
            float2 tsdf = *pos_n_psi;

            if (tsdf.y == 0.f || (tsdf.y == 1.f && (tsdf.x == 0.f || tsdf.x == -1.f))) {
                continue;
            }

            float2 tsdf_prev = *pos_global;

            float tsdf_new   = __fdividef(__fmaf_rn(tsdf_prev.y, tsdf_prev.x, tsdf.x), tsdf_prev.y + 1.f);
            float weight_new = fminf(tsdf_prev.y + 1.f, (float) phi_global.max_weight);

            /* pack and write */
            *pos_global = make_float2(tsdf_new, weight_new);
        }
    }
};  // namespace device

__global__ void integrate_kernel(const TsdfIntegrator integrator, TsdfVolume volume) { integrator(volume); }

__global__ void integrate_kernel(const TsdfIntegrator integrator, TsdfVolume phi_global, TsdfVolume phi_n_psi) {
    integrator(phi_global, phi_n_psi);
};
}  // namespace device
}  // namespace kfusion

void kfusion::device::integrate(const PtrStepSz<float>& dists, TsdfVolume& volume, const Aff3f& aff,
                                const Projector& proj) {
    /* init tsdf */
    TsdfIntegrator ti;
    ti.dists_size = make_int2(dists.cols, dists.rows);
    ti.vol2cam    = aff;
    ti.proj       = proj;

    dists_tex.filterMode     = cudaFilterModePoint;
    dists_tex.addressMode[0] = cudaAddressModeBorder;
    dists_tex.addressMode[1] = cudaAddressModeBorder;
    dists_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder binder(dists, dists_tex, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));
    (void) binder;

    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    integrate_kernel<<<grid, block>>>(ti, volume);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void kfusion::device::integrate(TsdfVolume& phi_global, TsdfVolume& phi_n_psi) {
    TsdfIntegrator ti;

    dim3 block(64, 16);
    dim3 grid(divUp(phi_global.dims.x, block.x), divUp(phi_global.dims.y, block.y));

    integrate_kernel<<<grid, block>>>(ti, phi_global, phi_n_psi);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

namespace kfusion {
namespace device {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// signed distance fields for various primitives

__global__ void init_box_kernel(TsdfVolume volume, const float3 b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y)
        return;

    /* centering */
    float3 c = make_float3(volume.dims.x / 2.f * volume.voxel_size.x, volume.dims.y / 2.f * volume.voxel_size.y,
                           volume.dims.z / 2.f * volume.voxel_size.z);

    float3 vc = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                            y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f) -
                c;
    float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i < volume.dims.z; vc += zstep, vptr = volume.zstep(vptr), ++i) {
        float3 d = make_float3(fabs(vc.x), fabs(vc.y), fabs(vc.z)) - b;

        float sdf =
            fmin(fmax(d.x, fmax(d.y, d.z)), 0.f) + norm(make_float3(fmax(d.x, 0.f), fmax(d.y, 0.f), fmax(d.z, 0.f)));
        float weight = 1.f;

        if (sdf >= volume.trunc_dist) {
            *vptr = make_float2(1.f, weight);
        } else if (sdf <= -volume.trunc_dist) {
            *vptr = make_float2(-1.f, weight);
        } else {
            *vptr = make_float2(__fdividef(sdf, volume.trunc_dist), weight);
        }
    }
}

__global__ void init_ellipsoid_kernel(TsdfVolume volume, const float3 r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y)
        return;

    /* centering */
    float3 c = make_float3(volume.dims.x / 2.f * volume.voxel_size.x, volume.dims.y / 2.f * volume.voxel_size.y,
                           volume.dims.z / 2.f * volume.voxel_size.z);

    float3 vc = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                            y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f) -
                c;
    float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i < volume.dims.z; vc += zstep, vptr = volume.zstep(vptr), ++i) {
        float k0 = norm(make_float3(vc.x / r.x, vc.y / r.y, vc.z / r.z));
        float k1 = norm(make_float3(vc.x / (r.x * r.x), vc.y / (r.y * r.y), vc.z / (r.z * r.z)));

        float sdf    = k0 * (k0 - 1.f) / k1;
        float weight = 1.f;

        if (sdf >= volume.trunc_dist) {
            *vptr = make_float2(1.f, weight);
        } else if (sdf <= -volume.trunc_dist) {
            *vptr = make_float2(-1.f, weight);
        } else {
            *vptr = make_float2(__fdividef(sdf, volume.trunc_dist), weight);
        }
    }
}

__global__ void init_sphere_kernel(TsdfVolume volume, float3 centre, float radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y)
        return;

    float3 vc    = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                            y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f);
    float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i < volume.dims.z; vc += zstep, vptr = volume.zstep(vptr), ++i) {
        float d = sqrtf(powf(vc.x - centre.x, 2) + powf(vc.y - centre.y, 2) + powf(vc.z - centre.z, 2));

        float sdf    = d - radius;
        float weight = (sdf > -volume.eta) ? 1.f : 0.f;

        if (sdf >= volume.trunc_dist) {
            *vptr = make_float2(1.f, weight);
        } else if (sdf <= -volume.trunc_dist) {
            *vptr = make_float2(-1.f, weight);
        } else {
            *vptr = make_float2(__fdividef(sdf, volume.trunc_dist), weight);
        }
    }
}

__global__ void init_plane_kernel(TsdfVolume volume, float z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y)
        return;

    float3 vc    = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                            y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f);
    float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i < volume.dims.z; vc += zstep, vptr = volume.zstep(vptr), ++i) {
        float sdf    = vc.z - z;
        float weight = 1.f;

        if (sdf >= volume.trunc_dist) {
            *vptr = make_float2(1.f, weight);
        } else if (sdf <= -volume.trunc_dist) {
            *vptr = make_float2(-1.f, weight);
        } else {
            *vptr = make_float2(__fdividef(sdf, volume.trunc_dist), weight);
        }
    }
}

__global__ void init_torus_kernel(TsdfVolume volume, const float2 t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= volume.dims.x || y >= volume.dims.y)
        return;

    /* centering */
    float3 c = make_float3(volume.dims.x / 2.f * volume.voxel_size.x, volume.dims.y / 2.f * volume.voxel_size.y,
                           volume.dims.z / 2.f * volume.voxel_size.z);

    float3 vc = make_float3(x * volume.voxel_size.x + volume.voxel_size.x / 2.f,
                            y * volume.voxel_size.y + volume.voxel_size.y / 2.f, volume.voxel_size.z / 2.f) -
                c;
    float3 zstep = make_float3(0.f, 0.f, volume.voxel_size.z);

    float2* vptr = volume.beg(x, y);
    for (int i = 0; i < volume.dims.z; vc += zstep, vptr = volume.zstep(vptr), ++i) {
        float2 q = make_float2(norm(make_float2(vc.x, vc.z)) - t.x, vc.y);

        float sdf    = norm(q) - t.y;
        float weight = 1.f;

        if (sdf >= volume.trunc_dist) {
            *vptr = make_float2(1.f, weight);
        } else if (sdf <= -volume.trunc_dist) {
            *vptr = make_float2(-1.f, weight);
        } else {
            *vptr = make_float2(__fdividef(sdf, volume.trunc_dist), weight);
        }
    }
}

}  // namespace device
}  // namespace kfusion

void kfusion::device::init_box(TsdfVolume& volume, const float3& b) {
    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    init_box_kernel<<<grid, block>>>(volume, b);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void kfusion::device::init_ellipsoid(TsdfVolume& volume, const float3& r) {
    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    init_ellipsoid_kernel<<<grid, block>>>(volume, r);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void kfusion::device::init_plane(TsdfVolume& volume, const float& z) {
    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    init_plane_kernel<<<grid, block>>>(volume, z);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void kfusion::device::init_sphere(TsdfVolume& volume, const float3& centre, const float& radius) {
    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    init_sphere_kernel<<<grid, block>>>(volume, centre, radius);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

void kfusion::device::init_torus(TsdfVolume& volume, const float2& t) {
    dim3 block(64, 16);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    init_torus_kernel<<<grid, block>>>(volume, t);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}
