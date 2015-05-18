#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  target_prob_.ReshapeLike(prob_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  CHECK_EQ(spatial_dim, 1);
  Dtype loss = 0;
  Dtype gamma = this->layer_param_.multi_label_softmax_loss_param().gamma();
  Dtype* target_prob = target_prob_.mutable_cpu_data(); 
  caffe_set(target_prob_.count(), (Dtype)0.0, target_prob);

  // get the target probability
  for (int i = 0; i < num; ++i) {
      const Dtype* curr_label = label + i * dim;
      const Dtype* curr_prob = prob_data + i * dim;
      vector<pair<Dtype, int> > label_idx;
      label_idx.clear();
      for (int j = 0; j < dim; ++j) {
          if (curr_label[j] > -0.5) {
              label_idx.push_back(make_pair(curr_prob[j], j));
          }
      }
      int curr_label_count = label_idx.size();
      CHECK_GE(curr_label_count, 1);

      sort(label_idx.begin(), label_idx.end(), std::greater<std::pair<Dtype, int> >());
      
      // only used for checking
      for (size_t j = 0; j < label_idx.size() - 1; j++) {
          CHECK_GE(label_idx[j].first, label_idx[j + 1].first);
      }
      //LOG(INFO) << "checked ok: " << label_idx.size(); 

      Dtype threshold = (Dtype)1.0 / curr_label_count / gamma;

      int anchor_idx = -1;
      Dtype s = 0;
      Dtype alpha;
      for (size_t j = 0; j < label_idx.size(); j++) {
          s += label_idx[j].first;
          alpha = (1.0 - threshold * (label_idx.size() - 1 - j)) / s;
          if (j == label_idx.size() - 1) {
              anchor_idx = j;
              break;
          } else if (alpha * label_idx[j + 1].first < threshold) {
              //CHECK_GE(alpha * label_idx[j].first, threshold - FLT_MIN);
              anchor_idx = j;
              break;
          }
      }
      Dtype* curr_target_prob = target_prob + i * dim;
      Dtype log_alpha = log(std::max(Dtype(FLT_MIN), alpha));

      for (size_t j = 0; j <= anchor_idx; j++) {
          int idx = label_idx[j].second;
          Dtype orig_prob = label_idx[j].first;

          curr_target_prob[idx] = orig_prob * alpha;
          //LOG(INFO) << "t: " << orig_prob * alpha;
          loss += orig_prob * alpha *  log_alpha;
      }

      for (size_t j = anchor_idx + 1; j < label_idx.size(); j++) {
          int idx = label_idx[j].second;
          Dtype orig_prob = label_idx[j].first;
          curr_target_prob[idx] = threshold;
          //LOG(INFO) << "t: " << threshold;
          loss += threshold * log(std::max(Dtype(FLT_MIN), 
                      threshold / orig_prob));
      }
      Dtype s1 = 0;
      for (size_t j = 0; j < dim; ++j) {
          CHECK_GE(curr_target_prob[j], 0);
          CHECK_LE(curr_target_prob[j], 1);
          s1 += curr_target_prob[j];
      }
      CHECK_GE(s1, 0.999);
      CHECK_LE(s1, 1.0001); 
  }

  (*top)[0]->mutable_cpu_data()[0] = loss / num;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MultiLabelSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* target_prob_data = target_prob_.cpu_data();
    caffe_sub(target_prob_.count(), prob_data, target_prob_data, bottom_diff);

    int num = prob_.num();
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MultiLabelSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelSoftmaxWithLossLayer);


}  // namespace caffe
