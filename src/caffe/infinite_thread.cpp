#include <glog/logging.h>
#include "caffe/infinite_thread.hpp"


InfiniteThread::InfiniteThread() : 
    worker_type_(WAIT), is_finished_(true) {
        th_.reset(new boost::thread(&InfiniteThread::entry, this));
    }

InfiniteThread::~InfiniteThread() {
    send_cmd(QUIT);
    th_->join();
}

void InfiniteThread::send_cmd(Command type) {
    CHECK_EQ(worker_type_, WAIT);
    is_finished_ = false; // the thread enter busy state.

    boost::mutex::scoped_lock lck(mu_worker_type_);
    worker_type_ = type;
    con_worker_type_.notify_one();
}

void InfiniteThread::wait_ready() {
    boost::mutex::scoped_lock lck(mu_status_);
    while (is_finished_ == false) {
        //boost::system_time const timeout = boost::get_system_time() + 
            //boost::posix_time::milliseconds(35000);
        con_result_.wait(lck);
    }
}

void InfiniteThread::entry() {
    while (1) {
        Command type = WAIT;
        {
            boost::mutex::scoped_lock lock(mu_worker_type_);
            while (worker_type_ == WAIT) {
                con_worker_type_.wait(lock); 
            }
            type = worker_type_;
            worker_type_ = WAIT;
        }
        // begin to work
        is_finished_ = false;
        // working 
        if (type == QUIT) {
            break;
        } else {
            process();
        }
    
        {
            boost::mutex::scoped_lock lck(mu_status_);
            is_finished_ = true;
            con_result_.notify_one();
        }
    }
}

void InfiniteThread::process() {
}
