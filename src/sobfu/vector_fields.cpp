/* sobfu includes */
#include <sobfu/vector_fields.hpp>

/*
 * VECTOR FIELD
 */

sobfu::cuda::VectorField::VectorField(cv::Vec3i dims_) : dims(dims_) {
    int no_voxels = dims[0] * dims[1] * dims[2];
    data.create(no_voxels * sizeof(float4));
    clear();
}

sobfu::cuda::VectorField::~VectorField() = default;

cv::Vec3i sobfu::cuda::VectorField::get_dims() const { return dims; }

kfusion::cuda::CudaData sobfu::cuda::VectorField::get_data() { return data; }

const kfusion::cuda::CudaData sobfu::cuda::VectorField::get_data() const { return data; }

void sobfu::cuda::VectorField::set_data(kfusion::cuda::CudaData& data) { data = data; }

void sobfu::cuda::VectorField::clear() {
    int3 d = make_int3(dims[0], dims[1], dims[2]);

    sobfu::device::VectorField field(data.ptr<float4>(), d);
    sobfu::device::clear(field);
}

void sobfu::cuda::VectorField::print() {
    int sizes[3] = {dims[0], dims[1], dims[2]};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC4);
    get_data().download(mat->ptr<float4>());

    std::cout << "--- FIELD ---" << std::endl;
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float u = mat->at<float4>(k, j, i).x;
                float v = mat->at<float4>(k, j, i).y;
                float w = mat->at<float4>(k, j, i).z;

                if (fabs(u) > 1e-5f || fabs(v) > 1e-5f || fabs(w) > 1e-5f) {
                    std::cout << "(x,y,z)=(" << i << ", " << j << ", " << k << "), (u,v,w)=(" << u << ", " << v << ","
                              << w << ")" << std::endl;
                }
            }
        }
    }

    delete mat;
}

int sobfu::cuda::VectorField::get_no_nans() {
    int sizes[3] = {dims[0], dims[1], dims[2]};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC4);
    get_data().download(mat->ptr<float4>());

    int no_nan = 0;
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; k++) {
                float u = mat->at<float4>(k, j, i).x;
                float v = mat->at<float4>(k, j, i).y;
                float w = mat->at<float4>(k, j, i).z;

                if (std::isnan(u) || std::isnan(v) || std::isnan(w)) {
                    no_nan++;
                }
            }
        }
    }

    delete mat;
    return no_nan;
}

/*
 * DEFORMATION FIELD
 */

sobfu::cuda::DeformationField::DeformationField(cv::Vec3i dims_) : VectorField(dims_) { clear(); }

sobfu::cuda::DeformationField::~DeformationField() = default;

void sobfu::cuda::DeformationField::clear() {
    cv::Vec3f dims = get_dims();
    int3 d         = make_int3(dims[0], dims[1], dims[2]);

    sobfu::device::DeformationField psi(get_data().ptr<float4>(), d);
    sobfu::device::init_identity(psi);
}

void sobfu::cuda::DeformationField::get_inverse(sobfu::cuda::DeformationField& psi_inv) {
    kfusion::device::Vec3i d = kfusion::device_cast<kfusion::device::Vec3i>(get_dims());

    sobfu::device::DeformationField psi_device(get_data().ptr<float4>(), d);
    sobfu::device::DeformationField psi_inverse_device(psi_inv.get_data().ptr<float4>(), d);

    sobfu::device::estimate_inverse(psi_device, psi_inverse_device);
}

void sobfu::cuda::DeformationField::apply(const cv::Ptr<kfusion::cuda::TsdfVolume> phi_n,
                                          cv::Ptr<kfusion::cuda::TsdfVolume> phi_n_psi) {
    kfusion::device::Vec3i d   = kfusion::device_cast<kfusion::device::Vec3i>(phi_n->getDims());
    kfusion::device::Vec3f vsz = kfusion::device_cast<kfusion::device::Vec3f>(phi_n->getVoxelSize());

    float3 voxel_sizes = kfusion::device_cast<float3>(phi_n->getVoxelSize());
    float trunc_dist   = phi_n->getTruncDist();
    float eta          = phi_n->getEta();
    float max_weight   = phi_n->getMaxWeight();

    kfusion::device::TsdfVolume phi_device(phi_n->data().ptr<float2>(), d, vsz, trunc_dist, eta, max_weight);
    kfusion::device::TsdfVolume phi_warped_device(phi_n_psi->data().ptr<float2>(), d, vsz, trunc_dist, eta, max_weight);

    sobfu::device::DeformationField psi_device(get_data().ptr<float4>(), d);

    sobfu::device::apply(phi_device, phi_warped_device, psi_device);
    kfusion::cuda::waitAllDefaultStream();
}

/*
 * JACOBIAN
 */

sobfu::cuda::Jacobian::Jacobian(cv::Vec3i dims_) : dims(dims_) {
    int no_voxels = dims[0] * dims[1] * dims[2];
    data.create(no_voxels * sizeof(Mat4f));
    clear();
}

sobfu::cuda::Jacobian::~Jacobian() = default;

kfusion::cuda::CudaData sobfu::cuda::Jacobian::get_data() { return data; }

const kfusion::cuda::CudaData sobfu::cuda::Jacobian::get_data() const { return data; }

void sobfu::cuda::Jacobian::clear() {
    int3 d = make_int3(dims[0], dims[1], dims[2]);

    sobfu::device::Jacobian J(data.ptr<Mat4f>(), d);
    sobfu::device::clear(J);
}

/*
 * SPATIAL GRADIENTS
 */

sobfu::cuda::SpatialGradients::SpatialGradients(cv::Vec3i dims_) {
    nabla_phi_n       = new sobfu::cuda::TsdfGradient(dims_);
    nabla_phi_n_o_psi = new sobfu::cuda::TsdfGradient(dims_);
    J                 = new sobfu::cuda::Jacobian(dims_);
    J_inv             = new sobfu::cuda::Jacobian(dims_);
    L                 = new sobfu::cuda::Laplacian(dims_);
    L_o_psi_inv       = new sobfu::cuda::Laplacian(dims_);
    nabla_U           = new sobfu::cuda::PotentialGradient(dims_);
    nabla_U_S         = new sobfu::cuda::PotentialGradient(dims_);
}

sobfu::cuda::SpatialGradients::~SpatialGradients() {
    delete nabla_phi_n, nabla_phi_n_o_psi, J, J_inv, L, L_o_psi_inv, nabla_U, nabla_U_S;
}
