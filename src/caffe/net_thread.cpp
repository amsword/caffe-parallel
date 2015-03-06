#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net_thread.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
NetThread<Dtype>::NetThread(const NetParameter& param, 
        int device_id, 
        int thread_id,
        int random_seed)
    : device_id_(device_id), thread_id_(thread_id),
      random_seed_(random_seed) {

  CHECK(InfiniteThread::th_);

  this->process_type_ = INIT;
  this->process_param_const_ = &param;
  this->send_cmd(RUN);
  this->wait_ready();
}

template <typename Dtype>
Dtype NetThread<Dtype>::ForwardBackward(const vector<Blob<Dtype>* > & bottom, int num_iter) {
    CHECK_EQ(bottom.size(), 0);

    process_param_int_ = num_iter;
    process_type_ = FORWARDBACKWARD;
    InfiniteThread::send_cmd(RUN);
    return 0;
}

template <typename Dtype>
void NetThread<Dtype>::process() {
    switch (process_type_) {
        case INIT:
            {
                if (device_id_ == -1) {
                    Caffe::set_mode(Caffe::CPU);
                } else {
                    LOG(INFO) << "settting device_id = " << device_id_;
                    Caffe::SetDevice(device_id_);
                    Caffe::set_mode(Caffe::GPU);
                }
                Caffe::set_phase(Caffe::TRAIN);
                Caffe::SetThreadID(this->thread_id_);
                if (random_seed_ >= 0) {
                    Caffe::set_random_seed(random_seed_);
                }

                const NetParameter& param = *((const NetParameter*)(this->process_param_const_));
                Net<Dtype>::Init(param);
            }
            break;
        case FORWARDBACKWARD:
            {
                int num_iter = process_param_int_;
                vector<Blob<Dtype>* > bottom;
                process_output0_ = Net<Dtype>::ForwardBackward(bottom, num_iter);
            }
            break;
        default:
            NOT_IMPLEMENTED;
    }
}

INSTANTIATE_CLASS(NetThread);

//--------------------------------------------------------------------------
template <typename Dtype>
NetParallel<Dtype>::NetParallel(const NetParameter &param, 
   const vector<int>& vec_device_id, const vector<int>& vec_random_seeds) 
    :NetThread<Dtype>::NetThread(param, vec_device_id[0], 0, vec_random_seeds[0]),
    is_enabled_peer_access_(false) {

    { // only check
        if (vec_device_id[0] == -1) {
            CHECK_EQ(Caffe::mode(), Caffe::CPU);
        } else {
            CHECK_EQ(Caffe::GetDeviceID(), vec_device_id[0]);
        }
    }

    vec_nets_.resize(vec_device_id.size() - 1);
    for (size_t i = 0; i < vec_nets_.size(); i++) {
        vec_nets_[i].reset(new NetThread<Dtype>(param, 
                    vec_device_id[i + 1], i + 1,
                    vec_random_seeds[i + 1]));
    }
}

template <typename Dtype>
void NetParallel<Dtype>::CheckPeerAccess() {
    if (!is_enabled_peer_access_) {
        Caffe::SetDevice(this->device_id_);
        for (size_t i = 0; i < vec_nets_.size(); i++) {
            caffe_enable_peer_access(vec_nets_[i]->device_id());
        }
        is_enabled_peer_access_ = true;
    }
}

template <typename Dtype>
void GetDataPointer(shared_ptr<Blob<Dtype> >& blob, 
        const Dtype* &ptr, int &device) {
    SyncedMemory::SyncedHead h = blob->data()->head();
    if (h == SyncedMemory::HEAD_AT_CPU || h == SyncedMemory::UNINITIALIZED) {
        device = -1;
        ptr = blob->cpu_data();
    } else if (h == SyncedMemory::HEAD_AT_GPU || h == SyncedMemory::SYNCED) {
        device = 0;
        ptr = blob->gpu_data();
    } else {
        LOG(FATAL);
    }
}

template <typename Dtype>
void GetMutableDataPointer(shared_ptr<Blob<Dtype> >& blob, 
        Dtype* &ptr, int &device) {
    SyncedMemory::SyncedHead h = blob->data()->head();
    if (h == SyncedMemory::HEAD_AT_CPU || h == SyncedMemory::UNINITIALIZED) {
        device = -1;
        ptr = blob->mutable_cpu_data();
    } else if (h == SyncedMemory::HEAD_AT_GPU || h == SyncedMemory::SYNCED) {
        device = 0;
        ptr = blob->mutable_gpu_data();
    } else {
        LOG(FATAL);
    }
}

template <typename Dtype>
void GetDiffPoniter(shared_ptr<Blob<Dtype> >& blob, 
        const Dtype* &ptr, int &device) {
    SyncedMemory::SyncedHead h = blob->diff()->head();
    if (h == SyncedMemory::HEAD_AT_CPU || h == SyncedMemory::UNINITIALIZED) {
        device = -1;
        ptr = blob->cpu_diff();
    } else if (h == SyncedMemory::HEAD_AT_GPU || h == SyncedMemory::SYNCED) {
        device = 0;
        ptr = blob->gpu_diff();
    } else {
        LOG(FATAL);
    }
}


template <typename Dtype>
void NetParallel<Dtype>::SendParameter() {
    for (size_t i = 0; i < vec_nets_.size(); i++) {
        const int to_device_id = vec_nets_[i]->device_id();
        std::vector<boost::shared_ptr<Blob<Dtype> > > &to_vec = vec_nets_[i]->params();

        for (size_t j = 0; j < to_vec.size(); j++) {
            if (this->device_id_ >= 0 && to_device_id >= 0) {
                const Dtype* from_data;
                int from_position;
                GetDataPointer(this->params()[j], from_data, from_position);
                from_position = from_position == 0 ? this->device_id_: from_position;

                Dtype* to_data;
                int to_position;
                GetMutableDataPointer(to_vec[j], to_data, to_position);
                to_position = to_position == 0 ? to_device_id : to_position;
                
                int count = this->params()[j]->count();
                CHECK_EQ(count, to_vec[j]->count());

                caffe_device_copy_asyc(count, from_data, from_position, 
                        to_data, to_position);
            } else {
                NOT_IMPLEMENTED;
            }
        }
    }

    if (vec_nets_.size() > 0) {
        caffe_device_asyc_copy_sync();
    }
}

template <typename Dtype>
void NetParallel<Dtype>::CollectParameter() {
    size_t num_params = this->params().size();

    if (vec_buffer_.size() == 0) {
        size_t total_storage = 0;
        vec_buffer_.resize(num_params);
        size_t num_threads = vec_nets_.size();
        if (num_threads) {
            for (size_t i = 0; i < num_params; i++) {
                vec_buffer_[i].reset(new Blob<Dtype>(this->params()[i]->count() * num_threads, 1, 1, 1));
                total_storage += this->params()[i]->count() * num_threads;
                if (this->device_id_ == -1) {
                    vec_buffer_[i]->mutable_cpu_data();
                } else {
                    vec_buffer_[i]->mutable_gpu_data();
                }
            }
        }
        LOG(INFO) << "auxiliary storage: " << total_storage * sizeof(Dtype) / 1024.0 / 1024.0 / 1024.0 
            << "GB";
    }

    for (size_t i = 0; i < vec_nets_.size(); i++) {
        std::vector<boost::shared_ptr<Blob<Dtype> > >& from_vec = 
            vec_nets_[i]->params();
        const int from_device = vec_nets_[i]->device_id();
        for (size_t j = 0; j < num_params; j++) {
            int count = this->params()[j]->count();
            int from_position;

            const Dtype* from_diff;

            GetDiffPoniter(from_vec[j], from_diff, from_position);
            from_position = from_position == 0 ? from_device : from_position;
            
            Dtype* to_ptr;
            int to_position;
            if (this->device_id_ == -1) {
                to_ptr = vec_buffer_[j]->mutable_cpu_data();
                to_position = -1;
            } else {
                to_ptr = vec_buffer_[j]->mutable_gpu_data();
                to_position = this->device_id_;
            }
            caffe_device_copy_asyc(count, 
                    from_diff, from_position, 
                    to_ptr + count * i, to_position);
        }
    }

    if (vec_nets_.size() > 0) {
        caffe_device_asyc_copy_sync();
        if (multiplier_ == NULL) {
            multiplier_.reset(new Blob<Dtype>(vec_nets_.size(), 1, 1, 1) );
            Dtype* ptr = multiplier_->mutable_cpu_data();
            for (int i = 0; i < multiplier_->count(); i++) {
                ptr[i] = 1.0 / (1.0 + vec_nets_.size());
            }
        }
        const Dtype* multiplier_data = multiplier_->gpu_data();
        // add and divide. 
        for (size_t i = 0; i < num_params; i++) {
            caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
                    1, this->params()[i]->count(), vec_nets_.size(), 
                    (Dtype)1.0, multiplier_data, vec_buffer_[i]->gpu_data(), 
                    (Dtype)(1.0 / (vec_nets_.size() + 1.0)), this->params()[i]->mutable_gpu_diff());
        }
    }
}

template <typename Dtype>
Dtype NetParallel<Dtype>::ForwardBackward(const vector<Blob<Dtype>* > & bottom, int num_iter) {
   // send the parameters to the others;
   CheckPeerAccess();
   SendParameter();

   NetThread<Dtype>::ForwardBackward(bottom, num_iter);
   for (size_t i = 0; i < vec_nets_.size(); i++) {
       vec_nets_[i]->ForwardBackward(bottom, num_iter);
   }

   this->wait_ready();
   for (size_t i = 0; i < vec_nets_.size(); i++) {
       vec_nets_[i]->wait_ready();
   }

   CollectParameter();
   Dtype r;
   this->get_output(r);
   for (size_t i = 0; i < vec_nets_.size(); i++) {
       Dtype r2;
       vec_nets_[i]->get_output(r2);
       r += r2;
   }
   return r / (1.0 + vec_nets_.size());
}

INSTANTIATE_CLASS(NetParallel);
}  // namespace caffe
