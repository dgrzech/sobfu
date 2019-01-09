#pragma once

/* kinfu includes */
#include <kfusion/types.hpp>

/* pcl includes */
#include <pcl/common/geometry.h>

/* sobfu includes */
#include <sobfu/params.hpp>

/* sys headers */
#include <memory>

namespace kfusion {
namespace cuda {
class KF_EXPORTS TsdfVolume {
public:
    TsdfVolume(const Params &params);
    virtual ~TsdfVolume();

    void create(const Vec3i &dims);

    Vec3i getDims() const;
    Vec3f getVoxelSize() const;

    const CudaData data() const;
    CudaData data();

    Vec3f getSize() const;
    void setSize(const Vec3f &size);

    float getTruncDist() const;
    void setTruncDist(float &distance);

    float getEta() const;
    void setEta(float &eta);

    float getMaxWeight() const;
    void setMaxWeight(float &weight);

    Affine3f getPose() const;
    void setPose(const Affine3f &pose);

    float getRaycastStepFactor() const;
    void setRaycastStepFactor(float &factor);

    float getGradientDeltaFactor() const;
    void setGradientDeltaFactor(float &factor);

    Vec3i getGridOrigin() const;
    void setGridOrigin(const Vec3i &origin);

    virtual void clear();
    void swap(CudaData &data);

    virtual void applyAffine(const Affine3f &affine);

    virtual void integrate(const TsdfVolume &phi_n_psi);
    virtual void integrate(const Dists &dists, const Affine3f &camera_pose, const Intr &intr);

    virtual void initBox(const float3 &b);
    virtual void initEllipsoid(const float3 &r);
    virtual void initPlane(const float &z);
    virtual void initSphere(const float3 &centre, const float &radius);
    virtual void initTorus(const float2 &t);

    void print_sdf_values();

    struct Entry {
        typedef unsigned short half;

        half tsdf;
        int weight;

        static float half2float(half value);
        static half float2half(float value);
    };

private:
    CudaData data_;

    float trunc_dist_;
    float eta_;
    float max_weight_;
    Vec3i dims_;
    Vec3f size_;
    Affine3f pose_;

    float gradient_delta_factor_;
    float raycast_step_factor_;
};
}  // namespace cuda
}  // namespace kfusion
