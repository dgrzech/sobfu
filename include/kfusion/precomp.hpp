#pragma once

/* cuda includes */
#include <vector_functions.h>

/* kinfu includes */
#include <kfusion/cuda/imgproc.hpp>
#include <kfusion/cuda/marching_cubes.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/internal.hpp>
#include <kfusion/kinfu.hpp>
#include <kfusion/types.hpp>

/* sys headers */
#include <iostream>

namespace kfusion {
template <typename D, typename S>
inline D device_cast(const S &source) {
    return *reinterpret_cast<const D *>(source.val);
}

template <>
inline device::Aff3f device_cast<device::Aff3f, Affine3f>(const Affine3f &source) {
    device::Aff3f aff;
    Mat3f R = source.rotation();
    Vec3f t = source.translation();
    aff.R   = device_cast<device::Mat3f>(R);
    aff.t   = device_cast<device::Vec3f>(t);
    return aff;
}
}  // namespace kfusion
