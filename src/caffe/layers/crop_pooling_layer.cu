#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    const int num_image, const int num_crops, const int each_crop_count,  
    Dtype* top_data, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index % each_crop_count;
    int n = index / each_crop_count;
    bottom_data += n * num_crops * each_crop_count;
    bottom_data += i;

    Dtype max_value = bottom_data[0];
    int max_idx = 0;
    for (int c = 1; c < num_crops; c++) {
        Dtype v = *(bottom_data + c * each_crop_count);
        if (v > max_value) {
            max_value = v;
            max_idx = c;
        }
    }
    top_data += n * each_crop_count + i;
    mask += n * each_crop_count + i;
    *top_data = max_value;
    *mask = max_idx;
  }
}

template <typename Dtype>
void CropPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int count = (*top)[0]->count();
  switch (this->layer_param_.crop_pooling_param().pool()) {
  case CropPoolingParameter_PoolMethod_MAX:
      {
    int* mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[1]->num(), num_crops_, 
        bottom[0]->channels() * bottom[0]->height() * bottom[0]->width(),
        top_data, mask);
      }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const int num_image, int num_crops, int each_crop_count,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int i = index % each_crop_count;
    int n = index / each_crop_count;
    top_diff += n * each_crop_count;
    mask += n * each_crop_count;
    bottom_diff += n * each_crop_count * num_crops;
    int index = mask[i]; 
    bottom_diff[index * each_crop_count + i] = top_diff[i];
  }
}

template <typename Dtype>
void CropPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  CHECK(!propagate_down[1]);

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  const int count = (*bottom)[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  switch (this->layer_param_.crop_pooling_param().pool()) {
  case CropPoolingParameter_PoolMethod_MAX:
      {
      const int* mask = max_idx_.gpu_data();
      int nthreads = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
      int each_crop_count = top[0]->channels() * top[0]->height() * top[0]->width();
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, mask, top[0]->num(),
        num_crops_, each_crop_count,
        bottom_diff);
      }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_CLASS(CropPoolingLayer);


}  // namespace caffe
