#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"
#include "leveldb/db.h"
#include "leveldb/write_batch.h"

#include "caffe/caffe.hpp"
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/net.hpp"
//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/io.hpp"
//#include "caffe/vision_layers.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = 1;
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features1  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_leveldb_name1[,name2,...]  num_mini_batches phase [CPU/GPU]"
    "  [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and leveldb names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and leveldbs must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  string str_phase = argv[num_required_args - 1];
  if (str_phase == "train")
  {
	  Caffe::set_phase(Caffe::TRAIN);
  }
  else if (str_phase == "test")
  {
	  Caffe::set_phase(Caffe::TEST);
  }
  else
  {
	  LOG(FATAL);
  }

  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_names(argv[++arg_pos]);
  vector<string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  string save_feature_leveldb_names(argv[++arg_pos]);
  vector<string> leveldb_names;
  boost::split(leveldb_names, save_feature_leveldb_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), leveldb_names.size()) <<
      " the number of blob names and leveldb names must be equal";
  size_t num_features = blob_names.size();

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  //vector<shared_ptr<leveldb::DB> > feature_dbs;
  vector<FILE* > feature_dbs;
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening save file: " << leveldb_names[i];
	FILE* fp = fopen(leveldb_names[i].c_str(), "wb");
	CHECK(fp) << leveldb_names[i];
	feature_dbs.push_back(fp);
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  caffe::Timer timer;
  timer.Start();
  vector<Blob<float>*> input_vec;
  vector<int> image_indices(num_features, 0);
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
	  if (timer.MilliSeconds() >= 2000) {
		  LOG(INFO) << batch_index;
		  timer.Start();
	  }
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) 
	{
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
	  if (batch_index == 0)
	  {
		 int total_num = num_mini_batches * batch_size;
		 CHECK_EQ(fwrite(&total_num, sizeof(int), 1, feature_dbs[i]), 1);
		 CHECK_EQ(fwrite(&dim_features, sizeof(int), 1, feature_dbs[i]), 1);
	  }
	  const Dtype* p_data = feature_blob->cpu_data();
	  CHECK_EQ(fwrite(p_data, sizeof(Dtype), feature_blob->count(), feature_dbs[i]), feature_blob->count());
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)

  for (int i = 0; i < num_features; i++)
  {
	  fclose(feature_dbs[i]);
  }
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

