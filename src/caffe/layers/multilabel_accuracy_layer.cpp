#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  (*top)[0]->Reshape(1, 1, 1, 1);

  int batch_size = bottom[0]->num();
  int num_batch = this->layer_param_.multilabel_accuracy_param().test_iter();
  int label_size = bottom[1]->count() / bottom[1]->num();
  CHECK_EQ(label_size, bottom[0]->count() / bottom[0]->num());

  vec_blobs_.resize(2);
  for (int i = 0; i < 2; i++) {
    vec_blobs_[i].reset(new Blob<Dtype>(batch_size * num_batch, label_size, 1, 1));
  }
  iter_ = 0; 
}

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    CHECK_EQ(this->layer_param_.multilabel_accuracy_param().rank_type(),
            MultiLabelAccuracyParameter_RankType_IMAGE);
    int batch_size = bottom[0]->num();
    int label_size = bottom[0]->count() / bottom[0]->num();
    int test_iter = this->layer_param_.multilabel_accuracy_param().test_iter();

    for (int i = 0; i < 2; i++) {
        caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), 
                vec_blobs_[i]->mutable_cpu_data() + iter_ * bottom[i]->count());
    }

    iter_++;
    if (iter_ == this->layer_param_.multilabel_accuracy_param().test_iter()) {
        int total_image = test_iter * batch_size; 
        std::vector<std::pair<Dtype, int> > vec_pair(total_image);
        const Dtype* bottom_data = vec_blobs_[0]->cpu_data();
        const Dtype* label_data = vec_blobs_[1]->cpu_data(); 
        std::vector<Dtype> vec_ap(label_size);

        for (int i = 0; i < label_size; i++) {
            for (int j = 0; j < total_image; j++) {
                vec_pair[i] = std::pair<Dtype, int>(*(bottom_data + j * label_size + i), j);
            }
            std::sort(vec_pair.begin(), vec_pair.end(), std::greater<std::pair<Dtype, int> >());
            int num_correct = 0;
            int num_retrieved = 0;
            Dtype ap = 0;
            for (int j = 0; j < total_image; j++) {
                const pair<Dtype, int> &p = vec_pair[j];
                int idx_image = p.second;
                int gnd_label = static_cast<int>(*(label_data + idx_image * label_size + i));
                if (gnd_label == 1) {
                    num_correct++;
                    num_retrieved++;
                    ap += (Dtype)num_correct / (Dtype)num_retrieved;
                } else if (gnd_label == -1) {
                    num_retrieved++;
                }
            }
            ap /= (Dtype)num_correct;
            vec_ap[i] = ap;
        }
        Dtype map = 0;
        for (int i = 0; i < label_size; i++) {
            map += vec_ap[i];
        }
        map /= (Dtype)label_size;
        string str = "";
        for (int i = 0; i < label_size; i++) {
            stringstream ss;
            ss << vec_ap[i];
            str = str + ss.str() + "\t";
        }
        LOG(INFO) << "ap(s): " << str;
        LOG(INFO) << "map: " << map;
        iter_ = 0;
    }
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
