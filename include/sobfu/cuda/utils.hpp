#pragma once

/* cuda includes */
#include <cuda.h>
#include <math_constants.h>

/* kinfu includes */
#include <kfusion/internal.hpp>

/* utility class used to avoid linker errors with extern unsized shared memory arrays with templated type */
template <class T>
struct SharedMemory {
    __device__ inline operator T *() {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }

    __device__ inline operator const T *() const {
        extern __shared__ int __smem[];
        return (T *) __smem;
    }
};

/* get 1d index from 3d coordinates */
__device__ __forceinline__ int get_global_idx(int x, int y, int z, int dim_x, int dim_y) {
    return z * dim_y * dim_y + y * dim_x + x;
}

/*
 * LERP
 */

template <typename T>
__host__ __device__ __forceinline__ T lerp(T v0, T v1, T t) {
    return fma(t, v0, fma(-t, v1, v1));
}

__host__ __device__ __forceinline__ float3 lerp3f(float3 v0, float3 v1, float t) {
    return make_float3(lerp<float>(v0.x, v1.x, t), lerp<float>(v0.y, v1.y, t), lerp<float>(v0.z, v1.z, t));
}

__host__ __device__ __forceinline__ float4 lerp4f(float4 v0, float4 v1, float t) {
    return make_float4(lerp<float>(v0.x, v1.x, t), lerp<float>(v0.y, v1.y, t), lerp<float>(v0.z, v1.z, t), 0.f);
}

/*
 * INTERPOLATION
 */

template <typename Vol>
__device__ __forceinline__ float2 interpolate_tsdf(const Vol &volume, const float3 &p_voxels) {
    float3 cf = p_voxels;

    cf.x = fminf(fmaxf(0.f, cf.x), (float) volume.dims.x - 1);
    cf.y = fminf(fmaxf(0.f, cf.y), (float) volume.dims.y - 1);
    cf.z = fminf(fmaxf(0.f, cf.z), (float) volume.dims.z - 1);

    /* rounding to negative infinity */
    int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

    int idx_x_1 = g.x + 1;
    if (cf.x == 0.f || cf.x == (float) volume.dims.x - 1) {
        idx_x_1--;
    }
    int idx_y_1 = g.y + 1;
    if (cf.y == 0.f || cf.y == (float) volume.dims.y - 1) {
        idx_y_1--;
    }
    int idx_z_1 = g.z + 1;
    if (cf.z == 0.f || cf.z == (float) volume.dims.z - 1) {
        idx_z_1--;
    }

    float a = cf.x - g.x;
    float b = cf.y - g.y;
    float c = cf.z - g.z;

    float tsdf   = lerp(lerp(lerp((*volume(idx_x_1, idx_y_1, idx_z_1)).x, (*volume(idx_x_1, idx_y_1, g.z + 0)).x, c),
                           lerp((*volume(idx_x_1, g.y + 0, idx_z_1)).x, (*volume(idx_x_1, g.y + 0, g.z + 0)).x, c), b),
                      lerp(lerp((*volume(g.x + 0, idx_y_1, idx_z_1)).x, (*volume(g.x + 0, idx_y_1, g.z + 0)).x, c),
                           lerp((*volume(g.x + 0, g.y + 0, idx_z_1)).x, (*volume(g.x + 0, g.y + 0, g.z + 0)).x, c), b),
                      a);
    float weight = (*volume(g.x, g.y, g.z)).y;

    return make_float2(tsdf, weight);
}

template <typename Field>
__device__ __forceinline__ float4 interpolate_field(const Field &field, const float3 &p_voxels) {
    float3 cf = p_voxels;

    cf.x = fminf(fmaxf(0.f, cf.x), (float) field.dims.x - 1);
    cf.y = fminf(fmaxf(0.f, cf.y), (float) field.dims.y - 1);
    cf.z = fminf(fmaxf(0.f, cf.z), (float) field.dims.z - 1);

    /* rounding to negative infinity */
    int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

    int idx_x_1 = g.x + 1;
    if (cf.x == 0.f || cf.x == (float) field.dims.x - 1) {
        idx_x_1--;
    }
    int idx_y_1 = g.y + 1;
    if (cf.y == 0.f || cf.y == (float) field.dims.y - 1) {
        idx_y_1--;
    }
    int idx_z_1 = g.z + 1;
    if (cf.z == 0.f || cf.z == (float) field.dims.z - 1) {
        idx_z_1--;
    }

    float a = cf.x - g.x;
    float b = cf.y - g.y;
    float c = cf.z - g.z;

    float4 result = lerp4f(lerp4f(lerp4f(*field(idx_x_1, idx_y_1, idx_z_1), *field(idx_x_1, idx_y_1, g.z + 0), c),
                                  lerp4f(*field(idx_x_1, g.y + 0, idx_z_1), *field(idx_x_1, g.y + 0, g.z + 0), c), b),
                           lerp4f(lerp4f(*field(g.x + 0, idx_y_1, idx_z_1), *field(g.x + 0, idx_y_1, g.z + 0), c),
                                  lerp4f(*field(g.x + 0, g.y + 0, idx_z_1), *field(g.x + 0, g.y + 0, g.z + 0), c), b),
                           a);
    return result;
}

template <typename Field>
__device__ __forceinline__ float4 interpolate_field_inv(const Field &field, const float3 &p_voxels) {
    float3 cf = p_voxels;

    cf.x = fminf(fmaxf(0.f, cf.x), (float) field.dims.x - 1);
    cf.y = fminf(fmaxf(0.f, cf.y), (float) field.dims.y - 1);
    cf.z = fminf(fmaxf(0.f, cf.z), (float) field.dims.z - 1);

    /* rounding to negative infinity */
    int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

    int idx_x_1 = g.x + 1;
    if (cf.x == 0.f || cf.x == (float) field.dims.x - 1) {
        idx_x_1--;
    }
    int idx_y_1 = g.y + 1;
    if (cf.y == 0.f || cf.y == (float) field.dims.y - 1) {
        idx_y_1--;
    }
    int idx_z_1 = g.z + 1;
    if (cf.z == 0.f || cf.z == (float) field.dims.z - 1) {
        idx_z_1--;
    }

    float a = cf.x - g.x;
    float b = cf.y - g.y;
    float c = cf.z - g.z;

    float4 result = lerp4f(lerp4f(lerp4f(field.get_displacement(idx_x_1, idx_y_1, idx_z_1),
                                         field.get_displacement(idx_x_1, idx_y_1, g.z + 0), c),
                                  lerp4f(field.get_displacement(idx_x_1, g.y + 0, idx_z_1),
                                         field.get_displacement(idx_x_1, g.y + 0, g.z + 0), c),
                                  b),
                           lerp4f(lerp4f(field.get_displacement(g.x + 0, idx_y_1, idx_z_1),
                                         field.get_displacement(g.x + 0, idx_y_1, g.z + 0), c),
                                  lerp4f(field.get_displacement(g.x + 0, g.y + 0, idx_z_1),
                                         field.get_displacement(g.x + 0, g.y + 0, g.z + 0), c),
                                  b),
                           a);
    return result;
}

__device__ __forceinline__ float4 interpolate_field(const float4 *field, const float3 &p_voxels, int dim_x, int dim_y,
                                                    int dim_z) {
    float3 cf = p_voxels;

    cf.x = fminf(fmaxf(0.f, cf.x), (float) dim_x - 1);
    cf.y = fminf(fmaxf(0.f, cf.y), (float) dim_y - 1);
    cf.z = fminf(fmaxf(0.f, cf.z), (float) dim_z - 1);

    /* rounding to negative infinity */
    int3 g = make_int3(__float2int_rd(cf.x), __float2int_rd(cf.y), __float2int_rd(cf.z));

    int idx_x_1 = g.x + 1;
    if (cf.x == 0.f || cf.x == (float) dim_x - 1) {
        idx_x_1--;
    }
    int idx_y_1 = g.y + 1;
    if (cf.y == 0.f || cf.y == (float) dim_y - 1) {
        idx_y_1--;
    }
    int idx_z_1 = g.z + 1;
    if (cf.z == 0.f || cf.z == (float) dim_z - 1) {
        idx_z_1--;
    }

    float a = cf.x - g.x;
    float b = cf.y - g.y;
    float c = cf.z - g.z;

    float4 result = lerp4f(lerp4f(lerp4f(field[get_global_idx(idx_x_1, idx_y_1, idx_z_1, dim_x, dim_y)],
                                         field[get_global_idx(idx_x_1, idx_y_1, g.z + 0, dim_x, dim_y)], c),
                                  lerp4f(field[get_global_idx(idx_x_1, g.y + 0, idx_z_1, dim_x, dim_y)],
                                         field[get_global_idx(idx_x_1, g.y + 0, g.z + 0, dim_x, dim_y)], c),
                                  b),
                           lerp4f(lerp4f(field[get_global_idx(g.x + 0, idx_y_1, idx_z_1, dim_x, dim_y)],
                                         field[get_global_idx(g.x + 0, idx_y_1, g.z + 0, dim_x, dim_y)], c),
                                  lerp4f(field[get_global_idx(g.x + 0, g.y + 0, idx_z_1, dim_x, dim_y)],
                                         field[get_global_idx(g.x + 0, g.y + 0, g.z + 0, dim_x, dim_y)], c),
                                  b),
                           a);
    return result;
}

/*
 * FLOAT2
 */

__device__ __forceinline__ float norm(float2 v) {
    return __fsqrt_rn(__fadd_rn(__fmul_rn(v.x, v.x), __fmul_rn(v.y, v.y)));
}

/*
 * FLOAT3
 */

__host__ __device__ __forceinline__ float3 &operator-=(float3 &v1, const float3 &v2) {
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}

__device__ __forceinline__ float3 operator/(const float3 &v, const float &d) {
    return make_float3(__fdividef(v.x, d), __fdividef(v.y, d), __fdividef(v.z, d));
}

__device__ __forceinline__ float n(const float3 &v) {
    return __fsqrt_rd(__fmul_rn(v.x, v.x) + __fmul_rn(v.y, v.y) + __fmul_rn(v.z, v.z));
}

__device__ __forceinline__ float3 sq(const float3 &v) {
    return make_float3(__fmul_rn(v.x, v.x), __fmul_rn(v.y, v.y), __fmul_rn(v.z, v.z));
}

__device__ __forceinline__ float rms_epsilon(const float3 &v) { return __fsqrt_rn(v.x + v.y + v.z + 1e-5f); }

/*
 * FLOAT4
 */

__device__ __forceinline__ float4 operator+(const float4 &v1, const float4 &v2) {
    return make_float4(__fadd_rn(v1.x, v2.x), __fadd_rn(v1.y, v2.y), __fadd_rn(v1.z, v2.z), 0.f);
}

__device__ __forceinline__ float4 operator-(const float4 &v1, const float4 &v2) {
    return make_float4(__fadd_rn(v1.x, -v2.x), __fadd_rn(v1.y, -v2.y), __fadd_rn(v1.z, -v2.z), 0.f);
}

__host__ __device__ __forceinline__ float4 &operator+=(float4 &v1, const float4 &v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}

__host__ __device__ __forceinline__ float4 &operator-=(float4 &v1, const float4 &v2) {
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}

__device__ __forceinline__ float4 operator*(const float4 &v, const float &m) {
    return make_float4(__fmul_rn(v.x, m), __fmul_rn(v.y, m), __fmul_rn(v.z, m), 0.f);
}

__device__ __forceinline__ float4 operator*(const float &m, const float4 &v) { return v * m; }

__device__ __forceinline__ float4 operator/(const float4 &v, const float &d) {
    return make_float4(__fdividef(v.x, d), __fdividef(v.y, d), __fdividef(v.z, d), 0.f);
}

__device__ __forceinline__ float4 operator/(const float &d, const float4 &v) { return v / d; }

__device__ __forceinline__ float norm(const float4 &v) {
    return __fsqrt_rd(__fmul_rn(v.x, v.x) + __fmul_rn(v.y, v.y) + __fmul_rn(v.z, v.z));
}

__device__ __forceinline__ float norm_sq(const float4 &v) {
    return __fmul_rn(v.x, v.x) + __fmul_rn(v.y, v.y) + __fmul_rn(v.z, v.z);
}

__device__ __forceinline__ float4 normalised(const float4 &v) {
    float n = norm(v);
    return (n > 1e-5f) ? v / n : v;
}

__host__ __device__ __forceinline__ float3 trunc(const float4 &f) { return make_float3(f.x, f.y, f.z); }

__device__ __forceinline__ float4 sq(const float4 &v) {
    return make_float4(__fmul_rn(v.x, v.x), __fmul_rn(v.y, v.y), __fmul_rn(v.z, v.z), 0.f);
}

__device__ __forceinline__ float rms_epsilon(const float4 &v) {
    return __fsqrt_rd(__fadd_rn(__fadd_rn(v.x, __fadd_rn(v.y, v.z)), 1e-5f));
}

/*
 * MAT3F
 */

__device__ __forceinline__ kfusion::device::Mat3f operator+(kfusion::device::Mat3f M1, kfusion::device::Mat3f M2) {
    kfusion::device::Mat3f N;

    N.data[0].x = __fadd_rn(M1.data[0].x, M2.data[0].x);
    N.data[0].y = __fadd_rn(M1.data[0].y, M2.data[0].y);
    N.data[0].z = __fadd_rn(M1.data[0].z, M2.data[0].z);

    N.data[1].x = __fadd_rn(M1.data[1].x, M2.data[1].x);
    N.data[1].y = __fadd_rn(M1.data[1].y, M2.data[1].y);
    N.data[1].z = __fadd_rn(M1.data[1].z, M2.data[1].z);

    N.data[2].x = __fadd_rn(M1.data[2].x, M2.data[2].x);
    N.data[2].y = __fadd_rn(M1.data[2].y, M2.data[2].y);
    N.data[2].z = __fadd_rn(M1.data[2].z, M2.data[2].z);

    return N;
}

__device__ __forceinline__ kfusion::device::Mat3f operator-(kfusion::device::Mat3f M1, kfusion::device::Mat3f M2) {
    kfusion::device::Mat3f N;

    N.data[0].x = __fsub_rn(M1.data[0].x, M2.data[0].x);
    N.data[0].y = __fsub_rn(M1.data[0].y, M2.data[0].y);
    N.data[0].z = __fsub_rn(M1.data[0].z, M2.data[0].z);

    N.data[1].x = __fsub_rn(M1.data[1].x, M2.data[1].x);
    N.data[1].y = __fsub_rn(M1.data[1].y, M2.data[1].y);
    N.data[1].z = __fsub_rn(M1.data[1].z, M2.data[1].z);

    N.data[2].x = __fsub_rn(M1.data[2].x, M2.data[2].x);
    N.data[2].y = __fsub_rn(M1.data[2].y, M2.data[2].y);
    N.data[2].z = __fsub_rn(M1.data[2].z, M2.data[2].z);

    return N;
}

__device__ __forceinline__ kfusion::device::Mat3f operator*(kfusion::device::Mat3f M, float m) {
    kfusion::device::Mat3f N;

    N.data[0].x = __fmul_rn(M.data[0].x, m);
    N.data[0].y = __fmul_rn(M.data[0].y, m);
    N.data[0].z = __fmul_rn(M.data[0].z, m);

    N.data[1].x = __fmul_rn(M.data[1].x, m);
    N.data[1].y = __fmul_rn(M.data[1].y, m);
    N.data[1].z = __fmul_rn(M.data[1].z, m);

    N.data[2].x = __fmul_rn(M.data[2].x, m);
    N.data[2].y = __fmul_rn(M.data[2].y, m);
    N.data[2].z = __fmul_rn(M.data[2].z, m);

    return N;
}

__device__ __forceinline__ kfusion::device::Mat3f operator/(kfusion::device::Mat3f M, float d) { return M * 1.f / d; }

/*
 * MAT4F
 */

__device__ __forceinline__ float4 operator*(Mat4f M, float4 v) {
    return make_float4(
        __fadd_rn(__fadd_rn(__fmul_rn(M.data[0].x, v.x), __fmul_rn(M.data[0].y, v.y)), __fmul_rn(M.data[0].z, v.z)),
        __fadd_rn(__fadd_rn(__fmul_rn(M.data[1].x, v.x), __fmul_rn(M.data[1].y, v.y)), __fmul_rn(M.data[1].z, v.z)),
        __fadd_rn(__fadd_rn(__fmul_rn(M.data[2].x, v.x), __fmul_rn(M.data[2].y, v.y)), __fmul_rn(M.data[2].z, v.z)),
        0.f);
}

__device__ __forceinline__ float det(Mat4f M) {
    return __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(M.data[0].x, M.data[1].y), M.data[2].z),
                                         __fmul_rn(__fmul_rn(M.data[1].x, M.data[2].y), M.data[0].z)),
                               __fmul_rn(__fmul_rn(M.data[2].x, M.data[0].y), M.data[1].z)),
                     __fadd_rn(__fmul_rn(__fadd_rn(__fmul_rn(M.data[0].z, M.data[1].y), M.data[2].x),
                                         __fmul_rn(__fmul_rn(M.data[1].z, M.data[2].y), M.data[0].x)),
                               __fmul_rn(__fmul_rn(M.data[2].z, M.data[0].y), M.data[1].x)));
}

/*
 * OTHER
 */

__host__ __device__ __forceinline__ float sign(float a) {
    if (a > 0.f) {
        return 1.f;
    } else if (a < 0.f) {
        return -1.f;
    } else {
        return 0.f;
    }
}

__host__ __device__ __forceinline__ bool is_truncated(float tsdf) {
    if (fabs(tsdf) >= 1.f) {
        return true;
    }

    return false;
}

__host__ __device__ __forceinline__ float heaviside_smooth(float phi, float epsilon) {
    return 1.f / CUDART_PI_F * (epsilon / (epsilon * epsilon + phi * phi));
}

__host__ __device__ __forceinline__ void vec(kfusion::device::Mat3f J, float *v) {
    v[0] = J.data[0].x;
    v[1] = J.data[1].x;
    v[2] = J.data[2].x;

    v[3] = J.data[0].y;
    v[4] = J.data[1].y;
    v[5] = J.data[2].y;

    v[6] = J.data[0].z;
    v[7] = J.data[1].z;
    v[8] = J.data[2].z;
}

__host__ __device__ __forceinline__ kfusion::device::Mat3f transpose(kfusion::device::Mat3f J) {
    kfusion::device::Mat3f J_T;

    float3 row_1  = J.data[0];
    J_T.data[0].x = row_1.x;
    J_T.data[1].x = row_1.y;
    J_T.data[2].x = row_1.z;

    float3 row_2  = J.data[1];
    J_T.data[0].y = row_2.x;
    J_T.data[1].y = row_2.y;
    J_T.data[2].y = row_2.z;

    float3 row_3  = J.data[2];
    J_T.data[0].z = row_3.x;
    J_T.data[1].z = row_3.y;
    J_T.data[2].z = row_3.z;

    return J_T;
}
