#include <vector>
#include <iomanip>
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
				  this->blobs_.resize(1);
				  // Intialize the weight
				  this->blobs_[0].reset(new Blob<Dtype>(1, channels_, height_, width_));
				  CHECK_GE(this->layer_param_.sf_param().filler_size(), 1);
				  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
							  this->layer_param_.sf_param().filler(0)));
				  filler->Fill(this->blobs_[0].get());
				  // initialize the multiplier
				  multiplier_.Reshape(num_, 1, 1, 1);
				  Dtype* multi = multiplier_.mutable_cpu_data();
				  for (int i = 0; i < multiplier_.count(); i++)
				  {
					  multi[i] = 1.0;
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
			  {
				  int num_gaussian = this->layer_param_.sf_param().num_gaussian();
				  CHECK_GT(num_gaussian, 0);
				  this->blobs_.resize(4);
				  this->blobs_[0].reset(new Blob<Dtype>(1, num_gaussian, 1, 1)); // coefcient
				  this->blobs_[1].reset(new Blob<Dtype>(1, 1, 2, num_gaussian)); // mean
				  this->blobs_[2].reset(new Blob<Dtype>(1, 1, 2, num_gaussian)); // diagonal
				  this->blobs_[3].reset(new Blob<Dtype>(1, num_gaussian, 1, 1)); // off-diagonal
				  CHECK_EQ(this->layer_param_.sf_param().filler_size(), 4);
				  // init weight
				  for (int i = 0; i < 4; i++)
				  {
					  shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(
								  this->layer_param_.sf_param().filler(i)));
					  filler->Fill(this->blobs_[i].get());
				  }
				  // the last one is the weighted summation;
				  gmm_plains_.Reshape(1, num_gaussian + 1, height_, width_);
				  // initialize the multiplier
				  multiplier_.Reshape(num_, channels_, 1, 1);
				  Dtype* multi = multiplier_.mutable_cpu_data();
				  for (int i = 0; i < multiplier_.count(); i++)
				  {
					  multi[i] = 1.0;
				  }
				  top_diff_sum_.Reshape(1, 1, height_, width_);
				  diff_buffer_.Reshape(5, num_gaussian, height_, width_);
			  }
			  break;
		  default:
			  NOT_IMPLEMENTED;
	  }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  //test_gradient(); LOG(FATAL);
}

template <typename Dtype>
void SFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  switch (this->layer_param_.sf_param().method())
  {
	  case SFParameter_AdditionMethod_CUBIC:
		  {
			  const Dtype* weight = this->blobs_[0]->cpu_data();
			  for (int n = 0; n < num_; n++)
			  {
				  const Dtype* curr_bottom_data = bottom_data + bottom[0]->offset(n);
				  Dtype* curr_top_data = top_data + (*top)[0]->offset(n);

				  for (int i = 0; i < this->blobs_[0]->count(); i++)
				  {
					  curr_top_data[i] = curr_bottom_data[i] + weight[i];
				  }
			  }
		  }
		  break;
	  case SFParameter_AdditionMethod_PLAIN:
		  {
			  const Dtype* weight = this->blobs_[0]->cpu_data();
			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, height_ * width_, 1, 
					  (Dtype)1.0, multiplier_.cpu_data(), weight, 
					  (Dtype)1., top_data);
		  }
		  break;
	  case SFParameter_AdditionMethod_GMM:
		  {
			  int num_gaussian = this->layer_param_.sf_param().num_gaussian();
			  bool is_projection = this->layer_param_.sf_param().is_projection();
			  
			  if (is_projection)
			  {
				Dtype* diagonal_data = this->blobs_[2]->mutable_cpu_data();
				Dtype* off_diag_data = this->blobs_[3]->mutable_cpu_data();
				for (int g = 0; g < num_gaussian; g++)
				{
					Dtype &r11 = diagonal_data[g];
					Dtype &r22 = diagonal_data[g + num_gaussian];
					Dtype &r12 = off_diag_data[g];
					if (r11 > 0 || r22 > 0 || r11 * r22 - r12 * r12 < 0)
					{
						if (r12 > -0.0000001 && r12 < 0.0000001)
						{
							if (r11 > 0) r11 = 0;
							if (r22 > 0) r22 = 0;
						}
						else
						{
							Dtype under_sqrt = (r11 - r22) * (r11 - r22) + 4 * r12 * r12;
							Dtype out_sqrt = sqrt(under_sqrt);
							Dtype lamda1 = (r11 + r22 + out_sqrt) / 2;
							Dtype lamda2 = (r11 + r22 - out_sqrt) / 2;
							Dtype b11 = 1;
							Dtype b12 = (lamda1 - r11) / r12;
							Dtype b22 = 1;
							Dtype b21 = (lamda2 - r22) / r12;
							Dtype ampli = b11 * b11 + b12 * b12; ampli = sqrt(ampli);
							b11 /= ampli; b12 /= ampli;
							ampli = b21 * b21 + b22 * b22; ampli = sqrt(ampli);
							b21 /= ampli; b22 /= ampli;
							if (lamda1 > 0) lamda1 = 0;
							if (lamda2 > 0) lamda2 = 0;
							r11 = lamda1 * b11 * b11 + lamda2 * b21 * b21;
							r22 = lamda1 * b12 * b12 + lamda2 * b22 * b22;
							r12 = lamda1 * b11 * b12 + lamda2 * b21 * b22;
						}
					}
				}
				//Dtype* mean_data = this->blobs_[1]->mutable_cpu_data();
				//for (int g = 0; g < num_gaussian; g++)
				//{
					//if (mean_data[g] < 0) mean_data[g] = 0;
					//if (mean_data[g] > height_ - 1) mean_data[g] = height_ - 1;
					//if (mean_data[g + num_gaussian] < 0) mean_data[g + num_gaussian] = 0;
					//if (mean_data[g + num_gaussian] > width_ - 1) mean_data[g + num_gaussian] = width_ - 1;
				//}
			  }

			  const Dtype* weight_data = this->blobs_[0]->cpu_data();
			  const Dtype* mean_data = this->blobs_[1]->cpu_data();
			  const Dtype* diagonal_data = this->blobs_[2]->cpu_data();
			  const Dtype* off_diag_data = this->blobs_[3]->cpu_data();
			  Dtype* gmm_plain_data = gmm_plains_.mutable_cpu_data();

			  // compute num_gaussian plains
			  for (int g = 0; g < num_gaussian; g++)
			  {
				  Dtype* gmm_data = gmm_plain_data + gmm_plains_.offset(0, g); 
				  Dtype h_bar = *(mean_data + g);
				  Dtype w_bar = *(mean_data + g + num_gaussian);
				  Dtype r11 = *(diagonal_data + g);
				  Dtype r22 = *(diagonal_data + g + num_gaussian);
				  Dtype r12 = *(off_diag_data + g);
				  for (int h = 0; h < height_; h++)
				  {
					  Dtype h_diff = (Dtype)h - h_bar;
					  for (int w = 0; w < width_; w++)
					  {
						  Dtype w_diff = (Dtype)w - w_bar;
						  Dtype v = h_diff * h_diff * r11 + w_diff * w_diff * r22 + 
							  h_diff * w_diff * 2 * r12;
						  CHECK_LT(v, 100);
						  *(gmm_data + h * width_ + w) = exp(v);
					  }
				  }
			  }
			  
			  // compute the weighted summation;
			  Dtype* summation_data = gmm_plain_data + gmm_plains_.offset(0, num_gaussian);
			  caffe_cpu_gemv(CblasTrans, 
					num_gaussian, height_ * width_,  
					(Dtype)1.0, gmm_plain_data, weight_data,
					(Dtype)0.0, summation_data);
			  
			  //gmm_plains_.save_to_file("/home/wangjianfeng/gmm_plain");
			  //LOG(FATAL);

			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, height_ * width_, 1, 
					  (Dtype)1.0, multiplier_.cpu_data(), summation_data, 
					  (Dtype)1., top_data);
		  }
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
  // Gradient with respect to weight
	switch (this->layer_param_.sf_param().method())
	{
		case SFParameter_AdditionMethod_CUBIC:
			NOT_IMPLEMENTED;
			break;
		case SFParameter_AdditionMethod_PLAIN:
			if (this->param_propagate_down_[0]) 
			{
				Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, 
						width_ * height_, channels_ * num_, (Dtype)1, 
						multiplier_.cpu_data(), top_diff, 
						(Dtype)0, weight_diff);
			}
			break;
		case SFParameter_AdditionMethod_GMM:
			{
				const Dtype* weight_data = this->blobs_[0]->cpu_data();
				const Dtype* mean_data = this->blobs_[1]->cpu_data();
				const Dtype* diagonal_data = this->blobs_[2]->cpu_data();
				const Dtype* off_diag_data = this->blobs_[3]->cpu_data();

				Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
				Dtype* mean_diff = this->blobs_[1]->mutable_cpu_diff();
				Dtype* diagonal_diff = this->blobs_[2]->mutable_cpu_diff();
				Dtype* off_diag_diff = this->blobs_[3]->mutable_cpu_diff();

				const Dtype* gmm_plain_data = gmm_plains_.cpu_data();
				Dtype* top_diff_sum = top_diff_sum_.mutable_cpu_data();
				int num_gaussian = this->layer_param_.sf_param().num_gaussian();
				// the sum of top diff
				caffe_cpu_gemv(CblasTrans,  
						channels_ * num_, width_ * height_, 
						(Dtype)1.0, top_diff, 
						multiplier_.cpu_data(), 
						(Dtype)0., top_diff_sum);
				// diff of the weights
				if (this->param_propagate_down_[0])
				{
					caffe_cpu_gemv(CblasNoTrans, num_gaussian, 
							width_ * height_, 
							(Dtype)1.0, gmm_plain_data, 
							top_diff_sum, (Dtype)0., weight_diff);
				}

				const size_t block_offset = num_gaussian * height_ * width_;
				Dtype* buffer_data = diff_buffer_.mutable_cpu_data(); 
				// diff of the mean value;
				for (int g = 0; g < num_gaussian; g++)
				{
					Dtype h_bar = *(mean_data + g);
					Dtype w_bar = *(mean_data + g + num_gaussian);
					Dtype r11 = *(diagonal_data + g);
					Dtype r22 = *(diagonal_data + g + num_gaussian);
					Dtype r12 = *(off_diag_data + g);
					for (int h = 0; h < height_; h++)
					{
						Dtype h_diff = h_bar - (Dtype)h;
						Dtype h_diff2 = h_diff * h_diff;
						for (int w = 0; w < width_; w++)
						{
							Dtype w_diff = w_bar - (Dtype)w;
							Dtype h_proj = r11 * h_diff + r12 * w_diff;
							h_proj *= 2;
							Dtype w_proj = r12 * h_diff + r22 * w_diff;
							w_proj *= 2;
							size_t offset = g * height_ * width_ + h * width_ + w;
							Dtype exp_value = *(gmm_plain_data + offset);
							Dtype w_exp = weight_data[g] * exp_value;
							buffer_data[offset] = h_proj * w_exp;
							buffer_data[offset + block_offset] = w_proj * w_exp;
							Dtype w_diff2 = w_diff * w_diff;
							buffer_data[offset + block_offset * 2] = w_exp * h_diff2; 
							buffer_data[offset + block_offset * 3] = w_exp * w_diff2;
							buffer_data[offset + block_offset * 4] = w_exp * w_diff * h_diff * 2;
						}
					}
				}
				if (this->param_propagate_down_[1])
				{
					caffe_cpu_gemv(CblasNoTrans, 
							2 * num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data, 
							top_diff_sum, (Dtype)0., mean_diff);
				}

				Dtype mu1 = this->layer_param_.sf_param().mu1();
				Dtype mu2 = this->layer_param_.sf_param().mu2();
				if (this->param_propagate_down_[2])
				{
					caffe_cpu_gemv(CblasNoTrans, 
							2 * num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data + 2 * num_gaussian * width_ * height_, 
							top_diff_sum, (Dtype)0., diagonal_diff);
					if (mu1 > 0.00000001 ||  mu2 > 0.00000001)
					{
						for (int g = 0; g < num_gaussian; g++) 
						{
							Dtype r11 = diagonal_data[g];
							Dtype r22 = diagonal_data[g + num_gaussian];
							Dtype r12 = off_diag_data[g];
							if (r11 > 0)
							{
								diagonal_diff[g] += mu1;
							}
							if (r22 > 0)
							{
								diagonal_diff[g + num_gaussian] += mu1;
							}
							if (r12 * r12 - r11 * r22 > 0)
							{
								diagonal_diff[g] -= r22 * mu1;
								diagonal_diff[g + num_gaussian] -= r11 * mu2;
							}
						}
					}
				}
				if (this->param_propagate_down_[3])
				{
					caffe_cpu_gemv(CblasNoTrans, 
							num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data + 4 * num_gaussian * width_ * height_, 
							top_diff_sum, (Dtype)0., off_diag_diff);
					if (mu1 > 0.00000001 ||  mu2 > 0.00000001)
					{
						for (int g = 0; g < num_gaussian; g++) 
						{
							Dtype r11 = diagonal_data[g];
							Dtype r22 = diagonal_data[g + num_gaussian];
							Dtype r12 = off_diag_data[g];
							if (r12 * r12 - r11 * r22 > 0)
							{
								off_diag_diff[g] += 2 * r12 * mu2;
							}
						}
					}
				}
			}
			break;
		default:
			NOT_IMPLEMENTED;
	}

	if (propagate_down[0]) {
		// Gradient with respect to bottom data
		int count = top[0]->count(); 
		caffe_copy(count, top_diff, bottom_diff);
	}
}

template <typename Dtype>
void SFLayer<Dtype>::test_gradient()
{
	// set bottom
	FillerParameter filler_param;
	vector<Blob<Dtype>*> bottom(1);
	shared_ptr<Blob<Dtype> > input_blob(new Blob<Dtype>());
	input_blob->Reshape(num_, channels_, height_, width_);
	bottom[0] = input_blob.get();
	filler_param.set_value(0);
	filler_param.set_type("gaussian");
	filler_param.set_std(1);
	shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
	filler->Fill(input_blob.get());

	// init top
	vector<Blob<Dtype>*> top(1);
	shared_ptr<Blob<Dtype> > output_blob(new Blob<Dtype>());
	output_blob->ReshapeLike(*input_blob);
	top[0] = output_blob.get();

	// forward
	Forward_gpu(bottom,&top);
	const Dtype* pout = output_blob->cpu_data();
	Dtype obj = 0;
	for (int i = 0; i < output_blob->count(); i++)
	{
		obj += pout[i] * pout[i];
	}
	obj *= 0.5;

	LOG(INFO) << input_blob->asum_data() << "\t"
		<< output_blob->asum_data();
	// calc grad
	vector<bool> is_down(1);
	is_down[0] = true;
	caffe_copy(output_blob->count(), 
			output_blob->gpu_data(), 
			output_blob->mutable_gpu_diff());
	Backward_gpu(top, is_down, &bottom);

	LOG(INFO) << "obj: " << obj;

	LOG(INFO) << num_ << "\t" << channels_ << "\t" << height_ << "\t" 
		<< width_;
	Dtype eps = 0.000001;
	Dtype total_error = 0;
	int num_error = 0;
	int num_printed = 0;
	for (int i = 0; i < 1; i++, eps *= 0.1)
	{
		LOG(INFO) << eps;
		for (int idx_blob = 0; idx_blob < this->blobs_.size(); idx_blob++)
		{
			LOG(INFO) << idx_blob;
			shared_ptr<Blob<Dtype> > param = this->blobs_[idx_blob];
			for (int idx_item = 0; idx_item < param->count(); idx_item++)
			{
				Dtype *x = param->mutable_cpu_data() + idx_item;	
				(*x) += eps;
				Forward_gpu(bottom, &top);
				x = param->mutable_cpu_data() + idx_item;
				(*x) -= eps;
				const Dtype* pout = output_blob->cpu_data();
				Dtype obj_eps = 0;
				for (int i = 0; i < output_blob->count(); i++)
				{
					obj_eps += pout[i] * pout[i];
				}
				obj_eps *= 0.5;
				Dtype ref_grad = (obj_eps - obj) / eps;
				Dtype calc_grad = *(param->cpu_diff() + idx_item);
				Dtype rela_error = abs((ref_grad - calc_grad) / ref_grad);
				total_error += rela_error;
				num_error++;
				if (num_printed++ < 100)
				{
					LOG(INFO) << std::setw(10) << ref_grad << "\t"
						<< std::setw(10) << calc_grad << "\t"
						<< std::setw(10) << rela_error;
					
				}
			}
		}
	}
	LOG(FATAL) << total_error / num_error;
}

#ifdef CPU_ONLY
STUB_GPU(SFLayer);
#endif

INSTANTIATE_CLASS(SFLayer);

}  // namespace caffe
