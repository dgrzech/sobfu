#include <kfusion/precomp.hpp>

using namespace kfusion;
using namespace kfusion::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume::Entry

float kfusion::cuda::TsdfVolume::Entry::half2float(half) { throw "Not implemented"; }

kfusion::cuda::TsdfVolume::Entry::half kfusion::cuda::TsdfVolume::Entry::float2half(float value) {
    throw "Not implemented";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

kfusion::cuda::TsdfVolume::TsdfVolume(const Params &params) {
    dims_ = params.volume_dims;
    size_ = params.volume_size;
    pose_ = params.volume_pose;

    trunc_dist_ = params.tsdf_trunc_dist;
    eta_        = params.eta;
    max_weight_ = params.tsdf_max_weight;

    gradient_delta_factor_ = params.gradient_delta_factor;

    create(dims_);
}

kfusion::cuda::TsdfVolume::~TsdfVolume() {}

void kfusion::cuda::TsdfVolume::create(const Vec3i &dims) {
    dims_ = dims;

    int no_voxels = dims_[0] * dims_[1] * dims_[2];
    data_.create(no_voxels * 2 * sizeof(float));

    clear();
}

Vec3i kfusion::cuda::TsdfVolume::getDims() const { return dims_; }

Vec3f kfusion::cuda::TsdfVolume::getVoxelSize() const {
    return Vec3f(size_[0] / dims_[0], size_[1] / dims_[1], size_[2] / dims_[2]);
}

const CudaData kfusion::cuda::TsdfVolume::data() const { return data_; }
CudaData kfusion::cuda::TsdfVolume::data() { return data_; }

Vec3f kfusion::cuda::TsdfVolume::getSize() const { return size_; }
void kfusion::cuda::TsdfVolume::setSize(const Vec3f &size) { size_ = size; }

float kfusion::cuda::TsdfVolume::getTruncDist() const { return trunc_dist_; }
void kfusion::cuda::TsdfVolume::setTruncDist(float &distance) { trunc_dist_ = distance; }

float kfusion::cuda::TsdfVolume::getEta() const { return eta_; }
void kfusion::cuda::TsdfVolume::setEta(float &eta) { eta_ = eta; }

float kfusion::cuda::TsdfVolume::getMaxWeight() const { return max_weight_; }
void kfusion::cuda::TsdfVolume::setMaxWeight(float &weight) { max_weight_ = weight; }

Affine3f kfusion::cuda::TsdfVolume::getPose() const { return pose_; }
void kfusion::cuda::TsdfVolume::setPose(const Affine3f &pose) { pose_ = pose; }

float kfusion::cuda::TsdfVolume::getRaycastStepFactor() const { return raycast_step_factor_; }
void kfusion::cuda::TsdfVolume::setRaycastStepFactor(float &factor) { raycast_step_factor_ = factor; }

float kfusion::cuda::TsdfVolume::getGradientDeltaFactor() const { return gradient_delta_factor_; }
void kfusion::cuda::TsdfVolume::setGradientDeltaFactor(float &factor) { gradient_delta_factor_ = factor; }

void kfusion::cuda::TsdfVolume::clear() {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);
    device::clear_volume(volume);
}

void kfusion::cuda::TsdfVolume::swap(CudaData &data) { data_.swap(data); }

void kfusion::cuda::TsdfVolume::applyAffine(const Affine3f &affine) { pose_ = affine * pose_; }

void kfusion::cuda::TsdfVolume::integrate(const TsdfVolume &phi_n_psi) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume phi_global_device(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);
    device::TsdfVolume phi_n_psi_device(phi_n_psi.data().ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::integrate(phi_global_device, phi_n_psi_device);
}

void kfusion::cuda::TsdfVolume::integrate(const Dists &dists, const Affine3f &camera_pose, const Intr &intr) {
    Affine3f vol2cam  = camera_pose.inv() * pose_;
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::integrate(dists, volume, aff, proj);
}

void kfusion::cuda::TsdfVolume::initBox(const float3 &b) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::init_box(volume, b);
}

void kfusion::cuda::TsdfVolume::initEllipsoid(const float3 &r) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::init_ellipsoid(volume, r);
}

void kfusion::cuda::TsdfVolume::initPlane(const float &z) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::init_plane(volume, z);
}

void kfusion::cuda::TsdfVolume::initSphere(const float3 &centre, const float &radius) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::init_sphere(volume, centre, radius);
}

void kfusion::cuda::TsdfVolume::initTorus(const float2 &t) {
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::TsdfVolume volume(data_.ptr<float2>(), dims, vsz, trunc_dist_, eta_, max_weight_);

    device::init_torus(volume, t);
}

void kfusion::cuda::TsdfVolume::print_sdf_values() {
    cv::Vec3i dims = getDims();

    float2 *phi_data_ptr = new float2[dims[0] * dims[1] * dims[2]];
    data().download(phi_data_ptr);
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float2 val = phi_data_ptr[k * dims[1] * dims[0] + j * dims[0] + i];
                if (val.x != 0.f) {
                    std::cout << val.x << std::endl;
                }
            }
        }
    }
}
