#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
using namespace caffe;



void test_time(int N)
{
	LOG(INFO) << "begin";
	Caffe::Get().SetDevice(1);
	LOG(INFO) << N;
	Blob<float> raw_data;
	raw_data.Reshape(N, 1, 1, 1);
	for (int i = 0; i < N; i++)
	{
		raw_data.mutable_cpu_data()[i] = i;
	}
	int nth = 2;
	vector<shared_ptr<Blob<float> > > vec_out(nth);
	for (int i = 0; i < nth; i++)
	{
		vec_out[i].reset(new Blob<float>());
		vec_out[i]->Reshape(N, 1, 1, 1);
	}
	const float* gpu_data = raw_data.gpu_data();
	vector<float*> vec_out_ptr(nth);
	for (int i = 0; i < nth; i++)
	{
		vec_out_ptr[i] = vec_out[i]->mutable_gpu_data();
	}
	
	Timer timer;
	timer.Start();
	for (int i = 0; i < nth; i++)
	{
		test_kernel(gpu_data, N, vec_out_ptr[i]);
	}
	cudaDeviceSynchronize();
	timer.Stop();
	LOG(INFO) << timer.MilliSeconds();

	vector<cudaStream_t> vec_streams(nth);
	for (int i = 0; i < nth; i++)
	{
		cudaStream_t stream1;
		CUDA_CHECK(cudaStreamCreate(&stream1));
		vec_streams[i] = stream1;
	}

	LOG(INFO) << "created";
	timer.Start();
	for (int i = 0; i < nth; i++)
	{
		test_kernel(gpu_data, N, vec_out_ptr[i], 
				vec_streams[i]);
	}
	cudaDeviceSynchronize();
	timer.Stop();
	LOG(INFO) << timer.MilliSeconds();
	for (int i =  0; i < nth; i++)
	{
		cudaStreamDestroy(vec_streams[i]);
	}
}

int main(int argc, char** argv)
{
	//google::ParseCommandLineFlags(&argc, &argv, true);

	test_time(100);



	return 0;
}

