#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GramianLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  normalize_scale_ = 1.0;
}

template <typename Dtype>
void GramianLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(3);
  top_shape[0] = bottom[0]->shape(0);
  top_shape[1] = bottom[0]->shape(1);
  top_shape[2] = bottom[0]->shape(1);
  top[0]->Reshape(top_shape);
  if (this->layer_param_.gramian_param().normalize_output()) {
    normalize_scale_ = 1.0 / bottom[0]->count(1);
  }
}

template <typename Dtype>
void GramianLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int batch = 0; batch < bottom[0]->shape(0); ++batch) {
    const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(batch);
    Dtype* mutable_top_data = top[0]->mutable_cpu_data() + top[0]->offset(batch);
    caffe_cpu_gemm<Dtype>(
        CblasNoTrans, 
        CblasTrans, 
        bottom[0]->shape(1),
        bottom[0]->shape(1),
        bottom[0]->count(2),
        normalize_scale_,
        bottom_data,
        bottom_data,
        (Dtype)0.,
        mutable_top_data);
  }
}

template <typename Dtype>
void GramianLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  for (int batch = 0; batch < bottom[0]->shape(0); ++batch) {
    Dtype* mutable_bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(batch);
    const Dtype* bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(batch);
    const Dtype* top_diff = top[0]->cpu_diff() + top[0]->offset(batch);
    // Input gradient
    if (propagate_down[0]) {
      caffe_cpu_gemm<Dtype>(
          CblasNoTrans,
          CblasNoTrans,
          bottom[0]->shape(1),
          bottom[0]->count(2),
          bottom[0]->shape(1),
          normalize_scale_,
          top_diff,
          bottom_data,
          (Dtype)0.,
          mutable_bottom_diff);
      caffe_cpu_gemm<Dtype>(
          CblasTrans,
          CblasNoTrans,
          bottom[0]->shape(1),
          bottom[0]->count(2),
          bottom[0]->shape(1),
          normalize_scale_,
          top_diff,
          bottom_data,
          (Dtype)1.,
          mutable_bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GramianLayer);
#endif

INSTANTIATE_CLASS(GramianLayer);
REGISTER_LAYER_CLASS(Gramian);

}  // namespace caffe
