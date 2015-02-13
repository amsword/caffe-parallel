#include <algorithm>
#include <cfloat>
#include <vector>
#include <iomanip>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxPlus1LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  opt_target_.Reshape(bottom[0]->num(), 1, 1, 1);
  //test_bottom_diff();
}

template <typename Dtype>
void SoftmaxPlus1LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxPlus1LossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	// The forward pass computes the softmax prob values.
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
	const Dtype* prob_data = prob_.cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	Dtype lower_bound = this->layer_param_.sp1_param().lower_bound();
	Dtype upper_bound = this->layer_param_.sp1_param().upper_bound();
	Dtype* target_data = opt_target_.mutable_cpu_data(); 

	Dtype loss = 0;
	// best target
	for (int n = 0; n < num; n++) 
	{
		int label_idx =static_cast<int>(label[n]);
		Dtype prob_at_label = prob_data[n * dim + label_idx]; 
		Dtype prob_last = prob_data[n * dim + dim - 1];
		Dtype &prob_target = target_data[n]; 
		if (prob_at_label < (prob_at_label + prob_last) * lower_bound)
		{
			prob_target = lower_bound;
		}
		else if (prob_at_label > (prob_at_label + prob_last) * upper_bound)
		{
			prob_target = upper_bound;
		}
		else
		{
			prob_target = prob_at_label / (prob_at_label + prob_last);
		}
		loss -= prob_target * log(std::max(prob_at_label, Dtype(FLT_MIN)))
			+ (1 - prob_target) * log(std::max(prob_last, Dtype(FLT_MIN)));
		loss += prob_target * log(std::max(prob_target, Dtype(FLT_MIN))) 
			+ (1 - prob_target) * log(std::max(1 - prob_target, Dtype(FLT_MIN)));
	}
	(*top)[0]->mutable_cpu_data()[0] = loss / num; 
	if (top->size() == 2) 
	{
		(*top)[1]->ShareData(prob_);
	}
}

template <typename Dtype>
void SoftmaxPlus1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
	  const Dtype* target_data = opt_target_.cpu_data(); 
	  int num = prob_.num();
	  int dim = prob_.count() / num;
	  for (int n = 0; n < num; ++n) {
		  int label_idx =static_cast<int>(label[n]);
		  Dtype prob_target = target_data[n]; 
		  bottom_diff[n * dim + label_idx] -= prob_target;
		  bottom_diff[n * dim + dim - 1] -= 1 - prob_target;
	  }
	  // Scale gradient
	  const Dtype loss_weight = top[0]->cpu_diff()[0];
	  caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
  }
}

template <typename Dtype>
void SoftmaxPlus1LossLayer<Dtype>::test_bottom_diff()
{
	int num = 10;
	int dim = 11;
	LOG(INFO) << num << "\t" << dim; 
	vector<Blob<Dtype>*> bottom(2);
	bottom[0] = softmax_bottom_vec_[0]; 
	Blob<Dtype>* input_data = softmax_bottom_vec_[0];
	shared_ptr<Blob<Dtype> > label_data(new Blob<Dtype>(num, 1, 1, 1));
	bottom[1] = label_data.get();
	FillerParameter filler_param;
	filler_param.set_type("gaussian");
	shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
	filler->Fill(bottom[0]);
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
	LOG(INFO) << obj;
	vector<bool> propagate_down(1);
	propagate_down[0] = true;
	Backward_cpu(top, propagate_down, &bottom);
	Dtype eps = 0.000001;
	Dtype s = 0;
	for (int n = 0; n < num; n++)
	{
		for (int i = 0; i < dim; i++)
		{
			*(input_data->mutable_cpu_data() + input_data->offset(n, i, 0, 0)) += eps;
			Forward_cpu(bottom, &top);
			Dtype after_obj = *out_data->cpu_data();
			Dtype ref_grad = (after_obj - obj) / eps;
			Dtype true_grad = *(input_data->cpu_diff() + input_data->offset(n, i, 0, 0));
			static int s_iter = 0;
			if (s_iter < 50)
			{
			LOG(INFO) << true_grad << "\t" << ref_grad << "\t"
				<< (true_grad - ref_grad) / ref_grad;
			}
			s_iter++;
			s += std::abs((true_grad - ref_grad) / ref_grad);
			*(input_data->mutable_cpu_data() + input_data->offset(n, i, 0, 0)) -= eps;
		}
	}
	LOG(INFO) << s;
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxPlus1LossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxPlus1LossLayer);


}  // namespace caffe
