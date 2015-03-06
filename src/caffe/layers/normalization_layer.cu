#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* in,
    const int num, const int channels, const int height,
    const int width, const int size, const Dtype alpha_over_size,
    Dtype* scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = 1. + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
   int num = bottom[0]->num();
   const Dtype* bottom_data = bottom[0]->gpu_data();
   CHECK(squared_length_.count());
   Dtype* squared_length_data = squared_length_.mutable_gpu_data();
   int count = bottom[0]->count() / num;

   // compute the square of the length
   for (int n = 0; n < num; n++) {
        const Dtype* curr_bottom_data = bottom_data + bottom[0]->offset(n);
        Dtype* curr_squared_length_data = squared_length_data + n;
        caffe_gpu_dot(count, curr_bottom_data, curr_bottom_data, curr_squared_length_data);
   }
   //caffe_gpu_powx(
}


template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
}

INSTANTIATE_CLASS(NormalizationLayer);

}  // namespace caffe
