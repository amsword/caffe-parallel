#ifndef CAFFE_NET_THREAD_HPP_
#define CAFFE_NET_THREAD_HPP_


#include "caffe/infinite_thread.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
class NetThread : public Net<Dtype>, public InfiniteThread {
public:
    explicit NetThread(const NetParameter& param, 
            int device_id, int thread_id = 0,
            int random_seed = -1);
    virtual ~NetThread() {}

public:
    virtual Dtype ForwardBackward(const vector<Blob<Dtype>* > & bottom, int num_iter);

    int device_id() { return device_id_; }

    void get_output(Dtype &output) { output = process_output0_; }

protected:
    virtual void process();

    enum CommandType {
        INIT,
        FORWARDBACKWARD,
    };

    CommandType process_type_;
    const void* process_param_const_;
    void* process_param0_;
    void* process_param1_;
    int process_param_int_;
    Dtype process_output0_;

    int device_id_;
    int thread_id_;
    int random_seed_; // if -1, no setting
    DISABLE_COPY_AND_ASSIGN(NetThread);
};

template <typename Dtype>
class NetParallel: public NetThread<Dtype> {
public:
    explicit NetParallel(const NetParameter &param, 
            const vector<int>& vec_device_id, 
            const vector<int>& vec_random_seeds);

    virtual ~NetParallel() {}

    virtual Dtype ForwardBackward(const vector<Blob<Dtype>* > & bottom, int num_iter);

protected:
    void CheckPeerAccess();
    void SendParameter();
    void CollectParameter();

protected:
    std::vector<boost::shared_ptr<NetThread<Dtype> > > vec_nets_;
    std::vector<boost::shared_ptr<Blob<Dtype> > > vec_buffer_;
    bool is_enabled_peer_access_;
    shared_ptr<Blob<Dtype> > multiplier_;
    
    DISABLE_COPY_AND_ASSIGN(NetParallel);
};

}  // namespace caffe

#endif  // CAFFE_NET_HPP_
