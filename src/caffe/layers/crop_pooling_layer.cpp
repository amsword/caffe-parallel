#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CropPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void CropPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    int total = bottom[0]->num();
    int num_images = bottom[1]->num();
    this->num_crops_ = total / num_images;
    CHECK_EQ(num_crops_ * num_images, total);
    (*top)[0]->Reshape(num_images, bottom[0]->channels(),
            bottom[0]->height(), bottom[0]->width());
    this->max_idx_.Reshape(num_images, bottom[0]->channels(),
            bottom[0]->height(), bottom[0]->width());
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void CropPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.crop_pooling_param().pool()) {
  case CropPoolingParameter_PoolMethod_MAX:
    {
    // Initialize
    int* mask = max_idx_.mutable_cpu_data();
    // The main loop
    int each_crop_count = bottom[0]->channels() * bottom[0]->height()
        * bottom[0]->width();
    for (int n = 0; n < bottom[1]->num(); n++) {
        const Dtype* bottom_current_data = bottom_data + 
            bottom[0]->offset(n * num_crops_);
        Dtype* top_current_data = top_data + 
            (*top)[0]->offset(n);
        int* mask_current = mask + 
            max_idx_.offset(n);
        for (int i = 0; i < each_crop_count; i++) {
            Dtype max_value = *(bottom_current_data + i);
            int max_idx = 0;
            for (int c = 1; c < num_crops_; c++) {
                Dtype v = *(bottom_current_data + c * each_crop_count + i);
                if (v > max_value) {
                    max_value = v;
                    max_idx = c;
                }
            }
            *(top_current_data + i) = max_value;
            *(mask_current + i) = max_idx;
        }
    }
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void CropPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  CHECK(!propagate_down[1]);

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  int each_crop_count = (*bottom)[0]->channels() * (*bottom)[0]->height()
      * (*bottom)[0]->width();
  switch (this->layer_param_.crop_pooling_param().pool()) {
  case CropPoolingParameter_PoolMethod_MAX:
      {
          const int* mask = max_idx_.cpu_data();
          for (int n = 0; n < (*bottom)[1]->num(); n++) {
              Dtype* bottom_current_diff = bottom_diff + 
                  (*bottom)[0]->offset(n * num_crops_);
              const Dtype* top_current_diff = top_diff + 
                  top[0]->offset(n);
              const int* mask_current = mask + 
                  max_idx_.offset(n);
              for (int i = 0; i < each_crop_count; i++) {
                  bottom_current_diff[i + 
                      mask_current[i] * each_crop_count] = 
                      top_current_diff[i];
              }
          }
      }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(CropPoolingLayer);
#endif

INSTANTIATE_CLASS(CropPoolingLayer);


}  // namespace caffe
