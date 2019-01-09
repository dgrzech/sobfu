#include <sobfu/scalar_fields.hpp>

/*
 * SCALAR FIELD
 */

sobfu::cuda::ScalarField::ScalarField(cv::Vec3i dims_) {
    dims = make_int3(dims_[0], dims_[1], dims_[2]);

    int no_voxels = dims.x * dims.y * dims.z;
    data.create(no_voxels * sizeof(float));
    clear();
}

sobfu::cuda::ScalarField::~ScalarField() = default;

kfusion::cuda::CudaData sobfu::cuda::ScalarField::get_data() { return data; }

const kfusion::cuda::CudaData sobfu::cuda::ScalarField::get_data() const { return data; }

int3 sobfu::cuda::ScalarField::get_dims() { return dims; }

void sobfu::cuda::ScalarField::clear() {
    sobfu::device::ScalarField field(data.ptr<float>(), dims);
    sobfu::device::clear(field);
}

float sobfu::cuda::ScalarField::sum() {
    sobfu::device::ScalarField field(data.ptr<float>(), dims);

    float result = sobfu::device::reduce_sum(field);
    return result;
}

void sobfu::cuda::ScalarField::print() {
    int sizes[3] = {dims.x, dims.y, dims.z};

    cv::Mat* mat = new cv::Mat(3, sizes, CV_32FC1);
    data.download(mat->ptr<float>());

    std::cout << "--- FIELD ---" << std::endl;
    for (int i = 0; i < dims.x; i++) {
        for (int j = 0; j < dims.y; j++) {
            for (int k = 0; k < dims.z; k++) {
                float val = mat->at<float>(k, j, i);

                std::cout << "(x,y,z)=(" << i << ", " << j << ", " << k << "), val=(" << val << std::endl;
            }
        }
    }

    delete mat;
}
