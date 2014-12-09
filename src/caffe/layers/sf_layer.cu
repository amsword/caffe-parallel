#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  switch (this->layer_param_.sf_param().method())
  {
	  case SFParameter_AdditionMethod_CUBIC:
		  break;
	  case SFParameter_AdditionMethod_PLAIN:
		  {
			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, width_ * height_, 1, 
					  (Dtype)1., multiplier_.gpu_data(),
					  weight, (Dtype)1., top_data);
		  }
		  break;
	  case SFParameter_AdditionMethod_GMM:
		  break;
	  default:
		  NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void SFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	// Gradient with respect to weight
	switch (this->layer_param_.sf_param().method())
	{
		case SFParameter_AdditionMethod_CUBIC:
			break;
		case SFParameter_AdditionMethod_PLAIN:
			if (this->param_propagate_down_[0]) 
			{
				caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 1, 
						width_ * height_ * channels_, num_, (Dtype)1, 
						multiplier_.gpu_data(), top_diff, 
						(Dtype)0, weight_diff);
			}
			break;
		case SFParameter_AdditionMethod_GMM:
			break;
		default:
			NOT_IMPLEMENTED;
	}

	if (propagate_down[0]) {
		// Gradient with respect to bottom data
		switch (this->layer_param_.sf_param().method())
		{
			case SFParameter_AdditionMethod_CUBIC:
				NOT_IMPLEMENTED;
				break;
			case SFParameter_AdditionMethod_PLAIN:
				{
				int count = top[0]->count(); 
				caffe_copy(count, top_diff, bottom_diff);
				}
				break;
			case SFParameter_AdditionMethod_GMM:
				NOT_IMPLEMENTED;
				break;
			default:
				NOT_IMPLEMENTED;
		}
	}
}

INSTANTIATE_CLASS(SFLayer);

}  // namespace caffe
