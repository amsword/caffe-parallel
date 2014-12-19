#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithPlus1LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  this->max_idx_.Reshape(bottom[0]->num(), 1, 1, 1);

  test_bottom_diff();
}

template <typename Dtype>
void SoftmaxWithPlus1LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithPlus1LossLayer<Dtype>::test_bottom_diff()
{
	int num = 1;
	int dim = 11;
	vector<Blob<Dtype>*> bottom(2);
	shared_ptr<Blob<Dtype> > input_data(new Blob<Dtype>(num, dim, 1, 1));
	bottom[0] = input_data.get();
	shared_ptr<Blob<Dtype>* > label_data(new Blob<Dtype>(num, 1, 1, 1));
	bottom[1] = label_data.get();
	FillerParameter filler_param;
	shared_ptr<Filler<Dtype> > filler(GetFiller(filler_param));
	filler->Fill(input_data.get());
	for (int n = 0; n < num; n++)
	{
		*(label_data->mutable_cpu_data() + n) = 2;
	}
	
	shared_ptr<Blob<Dtype> > out_data(new Blob<Dtype>(1, 1, 1, 1));
	*out_data->mutable_cpu_diff() = 1;
	vector<Blob<Dtype>* > top(1);
	top[0] = out_data.get();

	Forward_cpu(bottom, &top);
	Dtype obj = *out_data->cpu_data();
	vector<bool> propagate_down(1);
	propagate_down[0] = true;
	Backward_cpu(top, propagate_down, bottom);
	Dtype eps = 0.000001;
	for (int n = 0; n < num; n++)
	{
		for (int i = 0; i < dim; i++)
		{
			*(input_data->mutable_cpu_data() + input_data->offset(n, i, 0, 0)) += eps;
			Forward_cpu(bottom, &top);
			Dtype after_obj = *out_data->cpu_data();
			Dtype ref_grad = (after_obj - obj) / eps;
			Dtype true_grad = *(input_data->cpu_diff() + input_data->offset(n, i, 0, 0));
			LOG(INFO) << true_grad << "\t" << ref_grad << "\t"
				<< (true_grad - ref_grad) / ref_grad;
			*(input_data->mutable_cpu_data() + input_data->offset(n, i, 0, 0)) -= eps;
		}
	}
	LOG(FATAL);
}

template <typename Dtype>
void SoftmaxWithPlus1LossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  Dtype th = this->layer_param_.sp1_param().threshold();
  Dtype lambda = this->layer_param_.sp1_param().lambda();
  Dtype gamma = this->layer_param_.sp1_param().gamma();
  int* pmax_idx = this->max_idx_.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
	  Dtype max_prob = -1;
	  int max_idx = -1;
	  int label_idx = static_cast<int>(label[i]);

	  for (int j = 0; j < dim - 1; j++)
	  {
		  if (max_prob < prob_data[i * dim + j] && label_idx != j)
		  {
			  max_prob = prob_data[i * dim + j];
			  max_idx = j;
		  }
	  }
	  *(pmax_idx + i) = max_idx;
	  Dtype penalty = max_prob + th - prob_data[i * dim + label_idx];
	  penalty = penalty > 0 ? lambda * penalty : 0;
      loss -= log(std::max(prob_data[i * dim + label_idx] + prob_data[i * dim + dim - 1] + gamma, 
				  Dtype(FLT_MIN)));
	  loss += penalty;
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithPlus1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
	Dtype th = this->layer_param_.sp1_param().threshold();
	Dtype lambda = this->layer_param_.sp1_param().lambda();
	Dtype gamma = this->layer_param_.sp1_param().gamma();
	const int* pmax_idx = this->max_idx_.cpu_data();
    for (int i = 0; i < num; ++i) 
	{
		int max_idx = pmax_idx[i]; 
		int label_idx = static_cast<int>(label[i]);
		Dtype max_prob = prob_data[i * dim + max_idx];
		Dtype gnd_prob = prob_data[i * dim + label_idx];
		Dtype penalty = max_prob + th - gnd_prob;
		Dtype plus1_prob = prob_data[i * dim + dim - 1];
		bool is_penalty = penalty > 0 ? true : false;
		for (int j = 0; j < dim; j++)
		{
			Dtype &curr_diff = bottom_diff[i * dim + j];
			if (j == label_idx)
			{
				curr_diff -= gnd_prob / (gnd_prob + plus1_prob + gamma);
				if (is_penalty)
				{
					curr_diff += lambda * (gnd_prob * (gnd_prob - max_prob) - gnd_prob);
				}
			}
			else if (j == dim - 1)
			{
				curr_diff -= plus1_prob / (plus1_prob + gnd_prob + gamma);
				if (is_penalty)
				{
					curr_diff += lambda * (plus1_prob * (gnd_prob - max_prob));
				}
			}
			else if (j == max_idx)
			{
				if (is_penalty)
				{
					curr_diff += lambda * (max_prob * (gnd_prob - max_prob) + max_prob);
				}
			}
			else
			{
				if (is_penalty)
				{
					curr_diff += lambda * (prob_data[i * dim + j] * (gnd_prob - max_prob));
				}
			}
		}
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num,  bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithPlus1LossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithPlus1LossLayer);


}  // namespace caffe
