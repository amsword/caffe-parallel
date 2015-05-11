#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ZeroMeanTargetLoss<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    int num = bottom[0]->num();
    aux_blob_.Reshape(num, 1, 1, 1);
}

template <typename Dtype>
void ZeroMeanTargetLoss<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  if (top->size() == 1) {
      (*top)[0]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void ZeroMeanTargetLoss<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  Dtype* sum_data = aux_blob_.mutable_cpu_data();

  //{
      //Dtype s = 0; 
      //const Dtype* bottom_data = bottom[0]->cpu_data();
      //for (int n = 0; n < num; n++) {
          //for (int i = 0; i < dim; i++) {
              //Dtype v = *bottom_data++;
              //s += v > 0 ? v : -v;
          //}
      //}
      //Dtype bottom_loss = s / num / dim;
      //s = 0; 
      //bottom_data = bottom[0]->cpu_data();
      //for (int n = 0; n < num; n++) {
          //for (int i = 0; i < dim; i++) {
              //Dtype v = *bottom_data++;
              //s += v > 0 ? 1 : 0;
          //}
      //}
      //LOG(INFO) << s << "\t" << (Dtype)s / (Dtype)num / (Dtype)dim;
  //}

  Dtype loss = 0;
  for (int n = 0; n < num; n++) {
      Dtype s = 0;
      for (int i = 0; i < dim; i++) {
        s += *bottom_data++;
      }
      s /= dim;

      sum_data[n] = s; // used in back propagation;
      loss += s * s;
  }
  loss /= num;
  if (top->size() == 1) {
    *((*top)[0]->mutable_cpu_data()) = loss;
  }
}

template <typename Dtype>
void ZeroMeanTargetLoss<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* aux_data = aux_blob_.cpu_data();
    int num = (*bottom)[0]->num();
    int dim = (*bottom)[0]->count() / num;
    for (int i = 0; i < num; ++i) {
        Dtype v = aux_data[i];
        caffe_set(dim, v, bottom_diff + i * dim); 
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal((*bottom)[0]->count(), 2 * loss_weight / num / dim, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ZeroMeanTargetLoss);
#endif

INSTANTIATE_CLASS(ZeroMeanTargetLoss);


}  // namespace caffe
