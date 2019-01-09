#pragma once

#include <cuda_fp16.h>
#include <kfusion/cuda/temp_utils.hpp>
#include <kfusion/precomp.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

// __kf_device__ kfusion::device::TsdfVolume::TsdfVolume(float2 *_data, int3 _dims, float3 _voxel_size,
//                                                       float _trunc_dist, float _max_weight)
//     : data(_data), dims(_dims), voxel_size(_voxel_size), trunc_dist(_trunc_dist), max_weight(_max_weight) {}
//
// __kf_device__ kfusion::device::TsdfVolume::TsdfVolume(const TsdfVolume &other)
//     : data(other.data),
//       dims(other.dims),
//       voxel_size(other.voxel_size),
//       trunc_dist(other.trunc_dist),
//       max_weight(other.max_weight) {}

__kf_device__ float2 *kfusion::device::TsdfVolume::operator()(int x, int y, int z) {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

__kf_device__ const float2 *kfusion::device::TsdfVolume::operator()(int x, int y, int z) const {
    return data + x + y * dims.x + z * dims.y * dims.x;
}

__kf_device__ float2 *kfusion::device::TsdfVolume::beg(int x, int y) const { return data + x + dims.x * y; }

__kf_device__ float2 *kfusion::device::TsdfVolume::zstep(float2 *const ptr) const { return ptr + dims.x * dims.y; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Projector

__kf_device__ float2 kfusion::device::Projector::operator()(const float3 &p) const {
    float2 coo;
    coo.x = __fmaf_rn(f.x, __fdividef(p.x, p.z), c.x);
    coo.y = __fmaf_rn(f.y, __fdividef(p.y, p.z), c.y);
    return coo;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Reprojector

__kf_device__ float3 kfusion::device::Reprojector::operator()(int u, int v, float z) const {
    float x = z * (u - c.x) * finv.x;
    float y = z * (v - c.y) * finv.y;
    return make_float3(x, y, z);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Utility

namespace kfusion {
namespace device {
__kf_device__ Vec3f operator*(const Mat3f &m, const Vec3f &v) {
    return make_float3(dot(m.data[0], v), dot(m.data[1], v), dot(m.data[2], v));
}

__kf_device__ Vec3f operator*(const Aff3f &a, const Vec3f &v) { return a.R * v + a.t; }

/* operator for multiplication of an affine transform by a float */
__kf_device__ Aff3f operator*(const float &s, const Aff3f &a) {
    Aff3f b;
    b.R = a.R;
    b.t = s * a.t;
    return b;
}

/* 3-d matrix multiplication */
__kf_device__ Mat3f operator*(const Mat3f &m, const Mat3f &n) {
    Mat3f r;
    r.data[0] = m.data[0] * make_float3(n.data[0].x, n.data[1].x, n.data[2].x);
    r.data[1] = m.data[1] * make_float3(n.data[0].y, n.data[1].y, n.data[2].y);
    r.data[2] = m.data[2] * make_float3(n.data[0].z, n.data[1].z, n.data[2].z);
    return r;
}

/* affine transform composition */
__kf_device__ Aff3f operator*(const Aff3f &a, const Aff3f &b) {
    Aff3f c;
    c.R = a.R * b.R;
    c.t = a.t + b.t;
    return c;
}

__kf_device__ Vec3f tr(const float4 &v) { return make_float3(v.x, v.y, v.z); }

struct plus {
    __kf_device__ float operator()(float l, float r) const { return l + r; }
    __kf_device__ double operator()(double l, double r) const { return l + r; }
};
}  // namespace device
}  // namespace kfusion
