#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu_single(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
    bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
  }
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_diff[i * dim + j] = std::max(
        Dtype(0), 1 + bottom_diff[i * dim + j]);
    }
  }
  Dtype* loss = (*top)[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu_multi(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  CHECK_EQ(dim, bottom[1]->channels());

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < num; ++i) {
     for (int j = 0; j < dim; ++j) {
         int curr_label = static_cast<int>(label[i * dim + j]);
         if (curr_label == -1) {
             bottom_diff[i * dim + j] = std::max(
                     Dtype(0), 1 + bottom_diff[i * dim + j]);
         } else if (curr_label == 1){
             bottom_diff[i * dim + j] = std::max(
                     Dtype(0), 1 - bottom_diff[i * dim + j]);
         } else {
             bottom_diff[i * dim + j] = std::max(
                     Dtype(0), - bottom_diff[i * dim + j]);
         }
     }
  }
  Dtype* loss = (*top)[0]->mutable_cpu_data();
  switch (this->layer_param_.hinge_loss_param().norm()) {
  case HingeLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case HingeLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}
template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    if (bottom[1]->channels() == 1) {
        Forward_cpu_single(bottom, top);
    }
    else {
        Forward_cpu_multi(bottom, top);
    }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu_single(const vector<Blob<Dtype>*>& top,
        vector<Blob<Dtype>*>* bottom) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
      bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu_multi(const vector<Blob<Dtype>*>& top,
        vector<Blob<Dtype>*>* bottom) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();
    int dim = count / num;

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
            int curr_label = static_cast<int>(label[i * dim + j]);
            if (curr_label == 1 || curr_label == 0) {
                bottom_diff[i * dim + j] *= -1;
            }
        }
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.hinge_loss_param().norm()) {
    case HingeLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case HingeLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
      if ((*bottom)[1]->channels() == 1) {
          Backward_cpu_single(top, bottom);
      } else {
          Backward_cpu_multi(top, bottom);
      }

  }
}

INSTANTIATE_CLASS(HingeLossLayer);

}  // namespace caffe
