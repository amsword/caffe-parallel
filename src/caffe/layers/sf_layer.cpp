#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  num_ = bottom[0]->num();	
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
	  switch (this->layer_param_.sf_param().method())
	  {
		  case SFParameter_AdditionMethod_CUBIC:
			  {
				  this->blobs_.resize(4);
				  int num_gaussian = this->layer_param_.sf_param().num_gaussian();
				  this->blobs_[0].reset(new Blob<Dtype>(1, num_gaussian, 1, 1)); // coefcient
				  this->blobs_[1].reset(new Blob<Dtype>(1, 2 * num_gaussian, 1, 1)); // mean
				  this->blobs_[2].reset(new Blob<Dtype>(1, 2 * num_gaussian, 1, 1)); // diagonal
				  this->blobs_[3].reset(new Blob<Dtype>(1, num_gaussian, 1, 1)); // off-diagonal
				  CHECK_EQ(this->layer_param_.sf_param().filler_size(), 4);
				  // init weight
				  for (int i = 0; i < 4; i++)
				  {
					  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
								  this->layer_param_.sf_param().filler(i)));
					  filler->Fill(this->blobs_[i].get());
				  }
			  }
			  break;
		  case SFParameter_AdditionMethod_PLAIN:
			  {
				  this->blobs_.resize(1);
				  // Intialize the weight
				  this->blobs_[0].reset(new Blob<Dtype>(1, 1, height_, width_));
				  CHECK_GE(this->layer_param_.sf_param().filler_size(), 1);
				  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
							  this->layer_param_.sf_param().filler(0)));
				  filler->Fill(this->blobs_[0].get());
				  // initialize the multiplier
				  multiplier_.Reshape(num_, channels_, 1, 1);
				  Dtype* multi = multiplier_.mutable_cpu_data();
				  for (int i = 0; i < multiplier_.count(); i++)
				  {
					  multi[i] = 1.0;
				  }
				  break;
			  }
		  case SFParameter_AdditionMethod_GMM:
			  NOT_IMPLEMENTED;
			  break;
		  default:
			  NOT_IMPLEMENTED;
	  }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  switch (this->layer_param_.sf_param().method())
  {
	  case SFParameter_AdditionMethod_CUBIC:
		  NOT_IMPLEMENTED;
		  break;
	  case SFParameter_AdditionMethod_PLAIN:
		  {
			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, height_ * width_, 1, 
					  (Dtype)1.0, multiplier_.cpu_data(), weight, 
					  (Dtype)1., top_data);
		  }
		  break;
	  case SFParameter_AdditionMethod_GMM:
		  NOT_IMPLEMENTED;
		  break;
	  default:
		  NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void SFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  // Gradient with respect to weight
	switch (this->layer_param_.sf_param().method())
	{
		case SFParameter_AdditionMethod_CUBIC:
			NOT_IMPLEMENTED;
			break;
		case SFParameter_AdditionMethod_PLAIN:
			if (this->param_propagate_down_[0]) 
			{
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, 
						width_ * height_ * channels_, num_, (Dtype)1, 
						multiplier_.cpu_data(), top_diff, 
						(Dtype)0, weight_diff);
			}
			break;
		case SFParameter_AdditionMethod_GMM:
			NOT_IMPLEMENTED;
			break;
		default:
			NOT_IMPLEMENTED;
	}

	if (propagate_down[0]) {
		// Gradient with respect to bottom data
		switch (this->layer_param_.sf_param().method())
		{
			case SFParameter_AdditionMethod_CUBIC:
				break;
			case SFParameter_AdditionMethod_PLAIN:
				{
				int count = top[0]->count(); 
				caffe_copy(count, top_diff, bottom_diff);
				}
				break;
			case SFParameter_AdditionMethod_GMM:
				break;
			default:
				NOT_IMPLEMENTED;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SFLayer);
#endif

INSTANTIATE_CLASS(SFLayer);

}  // namespace caffe
