#pragma once


#include <iostream>
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <string>
#include <fstream>

enum Command {
    WAIT,
    QUIT,
    RUN,
};

class InfiniteThread
{
public:
    InfiniteThread();

    virtual ~InfiniteThread();

public:
    void send_cmd(Command type);

    void wait_ready();

protected:
    void entry();

    virtual void process();

protected:
    boost::shared_ptr<boost::thread> th_;

    // received the worker type;
    boost::condition_variable con_worker_type_;
    boost::mutex mu_worker_type_;
    Command worker_type_;

    // finished working
    boost::condition_variable con_result_;
    boost::mutex mu_status_;
    bool is_finished_;
};

