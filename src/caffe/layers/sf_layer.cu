#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void compute_gaussian_plain(const int nthreads, 
		int num_gaussian, int height, int width, 
		const Dtype* mean_data, 
		const Dtype* diagonal_data, const Dtype* off_diag_data, 
		Dtype* gmm_plain_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	  int plain_size = width * height;
	  int g = index / plain_size;
	  int offset_in_one = index % plain_size;
	  int h = offset_in_one / width;
	  int w = offset_in_one % width; 
	  
	  Dtype h_bar = *(mean_data + g);
	  Dtype w_bar = *(mean_data + g + num_gaussian);
	  Dtype r11 = *(diagonal_data + g);
	  Dtype r22 = *(diagonal_data + g + num_gaussian);
	  Dtype r12 = *(off_diag_data + g);
	  Dtype h_diff = ((Dtype)h - h_bar);
	  Dtype w_diff = (Dtype)w - w_bar;
	  Dtype v = h_diff * h_diff * r11 + w_diff * w_diff * r22 + 
		  h_diff * w_diff * 2 * r12;
	  gmm_plain_data[index] = exp(v);
  }
}

template <typename Dtype>
__global__ void project_negative(const int num_gaussian, Dtype* diagonal_data, 
		Dtype* off_diag_data)
{
  CUDA_KERNEL_LOOP(g, num_gaussian)
  {
	  Dtype &r11 = diagonal_data[g];
	  Dtype &r22 = diagonal_data[g + num_gaussian];
	  Dtype &r12 = off_diag_data[g];
	  if (r11 > 0 || r22 > 0 || r11 * r22 - r12 < 0)
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
}

template <typename Dtype>
void SFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  switch (this->layer_param_.sf_param().method())
  {
	  case SFParameter_AdditionMethod_CUBIC:
		  break;
	  case SFParameter_AdditionMethod_PLAIN:
		  {
			  const Dtype* weight = this->blobs_[0]->gpu_data();
			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, width_ * height_, 1, 
					  (Dtype)1., multiplier_.gpu_data(),
					  weight, (Dtype)1., top_data);
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
				  project_negative<Dtype><<<CAFFE_GET_BLOCKS(num_gaussian), CAFFE_CUDA_NUM_THREADS>>>
					  (num_gaussian, diagonal_data, off_diag_data);
			  }

			  Dtype* gmm_plain_data = gmm_plains_.mutable_gpu_data();
			  const Dtype* weight_data = this->blobs_[0]->gpu_data();
			  const Dtype* mean_data = this->blobs_[1]->gpu_data();
			  const Dtype* diagonal_data = this->blobs_[2]->gpu_data();
			  const Dtype* off_diag_data = this->blobs_[3]->gpu_data();
			  // compute num_gaussian plains
			  int nthreads = num_gaussian * height_ * width_;
			  compute_gaussian_plain<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
				  (nthreads, 
					num_gaussian, height_, width_, 
					mean_data, 
					diagonal_data, off_diag_data, 
					gmm_plain_data);
			  
			  // compute the weighted summation;
			  Dtype* summation_data = gmm_plain_data + gmm_plains_.offset(0, num_gaussian);
			  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
					1, height_ * width_, num_gaussian, 
					(Dtype)1.0, weight_data, gmm_plain_data, 
					(Dtype)0.0, summation_data);
			  
			  caffe_copy(bottom[0]->count(), bottom_data, top_data);
			  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, 
					  num_ * channels_, height_ * width_, 1, 
					  (Dtype)1.0, multiplier_.gpu_data(), summation_data, 
					  (Dtype)1., top_data);
		  }
		  break;
	  default:
		  NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
__global__ void compute_diff_buffer(const int nthreads, 
		int num_gaussian, int height, int width, 
		const Dtype* weight_data,
		const Dtype* mean_data, 
		const Dtype* diagonal_data, 
		const Dtype* off_diag_data, 
		const Dtype* gmm_plain_data,
		Dtype* buffer_data)
{
	CUDA_KERNEL_LOOP(index, nthreads) 
	{
		int plain_size = height * width;
		int block_size = plain_size * num_gaussian;
		int type = index / block_size;

		int idx_in_block = index % block_size;
		int g = idx_in_block / plain_size;
		int idx_in_plain = idx_in_block % plain_size;
		int h = idx_in_plain / width;
		int w = idx_in_plain % width;

		Dtype h_bar = *(mean_data + g);
		Dtype w_bar = *(mean_data + g + num_gaussian);
		Dtype r11 = *(diagonal_data + g);
		Dtype r22 = *(diagonal_data + g + num_gaussian);
		Dtype r12 = *(off_diag_data + g);
		Dtype h_diff = ((Dtype)h - h_bar);
		Dtype h_diff2 = h_diff * h_diff;
		Dtype w_diff = (Dtype)w - w_bar;
		Dtype w_diff2 = w_diff * w_diff;
		Dtype hw_diff = 2 * h_diff * w_diff;
		Dtype h_proj = r11 * h_diff + r12 * w_diff;
		Dtype w_proj = r12 * h_diff + r22 * w_diff;
		int offset = idx_in_block; 
		Dtype exp_value = *(gmm_plain_data + offset);
		Dtype w_exp = weight_data[g] * exp_value;
		switch (type)
		{
			case 0:
				buffer_data[index] = h_proj * w_exp;
				break;
			case 1:
				buffer_data[index] = w_proj * w_exp;
				break;
			case 2:
				buffer_data[index] = w_exp * h_diff2; 
				break;
			case 3:
				buffer_data[index] = w_exp * w_diff2;
				break;
			case 4:
				buffer_data[index] = w_exp * hw_diff;
				break;
		}
	}
}

template <typename Dtype>
__global__ void update_diagonal_diff(int nthreads, 
		int num_gaussian, 
		Dtype mu1, Dtype mu2, 
		const Dtype* diagonal_data, 
		const Dtype* off_diag_data, 
		Dtype* diagonal_diff)
{
	CUDA_KERNEL_LOOP(index, 2 * num_gaussian) 
	{
		int g = index % 2;
		int type = index / 2;
		Dtype r11 = diagonal_data[g];
		Dtype r22 = diagonal_data[g + num_gaussian];
		Dtype r12 = off_diag_data[g];
		if (type == 0)
		{
			if (r11 > 0)
			{
				diagonal_diff[g] += mu1;
			}
			if (r12 * r12 - r11 * r22 > 0)
			{
				diagonal_diff[g] -= mu1 * r22; 
			}
		}
		else
		{
			if (r22 > 0)
			{
				diagonal_diff[g + num_gaussian] += mu1;
			}
			if (r12 * r12 - r11 * r22 > 0)
			{
				diagonal_diff[g + num_gaussian] -= r11 * mu2;
			}
		}
	}
}

template <typename Dtype>
__global__ void update_off_diagonal_diff(int num_gaussian, 
		Dtype mu1, Dtype mu2, 
		const Dtype* diagonal_data, 
		const Dtype* off_diag_data, 
		Dtype* off_diag_diff)
{
	CUDA_KERNEL_LOOP(g, num_gaussian) 
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
			{
				const Dtype* weight_data = this->blobs_[0]->gpu_data();
				const Dtype* mean_data = this->blobs_[1]->gpu_data();
				const Dtype* diagonal_data = this->blobs_[2]->gpu_data();
				const Dtype* off_diag_data = this->blobs_[3]->gpu_data();
				
				Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
				Dtype* mean_diff = this->blobs_[1]->mutable_gpu_diff();
				Dtype* diagonal_diff = this->blobs_[2]->mutable_gpu_diff();
				Dtype* off_diag_diff = this->blobs_[3]->mutable_gpu_diff();
				
				Dtype* gmm_plain_data = gmm_plains_.mutable_gpu_data();
				Dtype* buffer_data = diff_buffer_.mutable_gpu_data(); 
				Dtype* top_diff_sum = top_diff_sum_.mutable_gpu_data();
				int num_gaussian = this->layer_param_.sf_param().num_gaussian();
				// the sum of top diff
			    caffe_gpu_gemv(CblasTrans,  
						channels_ * num_, width_ * height_, 
						(Dtype)1.0, top_diff, 
						multiplier_.gpu_data(), 
						(Dtype)0., top_diff_sum);
				// diff of the weights
				if (this->param_propagate_down_[0])
				{
					caffe_gpu_gemv(CblasNoTrans, num_gaussian, 
							width_ * height_, 
							(Dtype)1.0, gmm_plain_data, 
							top_diff_sum, (Dtype)0., weight_diff);
				}
			
				// diff of the mean value;
				int nthreads = 5 * num_gaussian * height_ * width_;
				compute_diff_buffer<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
					(nthreads, 
					 num_gaussian, height_, width_, 
					 weight_data, 
					 mean_data, 
					 diagonal_data, 
					 off_diag_data, 
					 gmm_plain_data, 
					 buffer_data);
				CUDA_POST_KERNEL_CHECK;
				if (this->param_propagate_down_[1])
				{
					caffe_gpu_gemv(CblasNoTrans, 
							2 * num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data, 
							top_diff_sum, (Dtype)0., mean_diff);
				}
				Dtype mu1 = this->layer_param_.sf_param().mu1();
				Dtype mu2 = this->layer_param_.sf_param().mu2();
				if (this->param_propagate_down_[2])
				{
					caffe_gpu_gemv(CblasNoTrans, 
							2 * num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data + 2 * num_gaussian * width_ * height_, 
							top_diff_sum, (Dtype)0., diagonal_diff);
					if (mu1 > 0.00000001 ||  mu2 > 0.00000001)
					{
						int nthreads = 2 * num_gaussian;
						update_diagonal_diff<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
								nthreads, num_gaussian, mu1, mu2, diagonal_data, off_diag_data, diagonal_diff);
					}
				}
				if (this->param_propagate_down_[3])
				{
					caffe_gpu_gemv(CblasNoTrans, 
							num_gaussian, width_ * height_, 
							(Dtype)1., buffer_data + 4 * num_gaussian * width_ * height_, 
							top_diff_sum, (Dtype)0., off_diag_diff);
					if (mu1 > 0.00000001 ||  mu2 > 0.00000001)
					{
						update_off_diagonal_diff<Dtype><<<CAFFE_GET_BLOCKS(num_gaussian), CAFFE_CUDA_NUM_THREADS>>>(
								num_gaussian, mu1, mu2, diagonal_data, off_diag_data, off_diag_diff);
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

INSTANTIATE_CLASS(SFLayer);

}  // namespace caffe
