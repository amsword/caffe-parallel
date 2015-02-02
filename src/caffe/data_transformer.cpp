#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param) 
    :param_(param) {
    phase_ = Caffe::phase();

    scale_ = param_.scale();
    if (param_.mean_values_size() != 0) {
        mean_values_.resize(param_.mean_values_size());
        for (int i = 0; i < param_.mean_values_size(); i++) {
            mean_values_[i] = param_.mean_values(i);
        }
    }
}

template <typename Dtype>
Dtype ImageElement(const Dtype* data, int c, int h, int w, 
        const int height, const int width) {
    if (h < 0 || w < 0 || h >= height || w >= width) {
        return 0.0;
    } else {
        int data_index = (c * height + h) * width + w;
        return data[data_index];
    }
}
template <typename DataType, typename PositionType>
DataType BilinearInter(const DataType* data, int c, PositionType h, PositionType w,
        const int height, const int width) {
    if (h < 0 || w < 0 || h >= height - 1 || w >= width - 1) {
        return 0.0;
    } else {
        int lower_h = (int)h;
        int lower_w = (int)w;
        DataType off_h = h - lower_h;
        DataType off_w = w - lower_w;
        return ImageElement(data, c, lower_h, lower_w, height, width) * (1 - off_h) * (1 - off_w) + 
            ImageElement(data, c, lower_h + 1, lower_w, height, width) * (off_h) * (1 - off_w) + 
            ImageElement(data, c, lower_h, lower_w + 1, height, width) * (1 - off_h) * off_w + 
            ImageElement(data, c, lower_h + 1, lower_w + 1, height, width) * off_h * off_w;
    }
}
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id, const Datum& datum, 
          const Dtype* mean, Dtype* transformed_data, 
          const int crop_size, const int cover_size, 
          int h_off, int w_off, bool is_mirror, bool is_random_rotate) {
  const int height = datum.height();
  const int width = datum.width();
  Dtype angle = 0;
  Dtype cos_angle;
  Dtype sin_angle = std::sin(angle);
  if (is_random_rotate) {
     angle = RandAngle();
     LOG(INFO) << angle / 3.1415926 * 180.0;
     cos_angle = std::cos(angle);
     sin_angle = std::sin(angle);
  }
  Dtype ratio = (Dtype)cover_size / (Dtype)crop_size;
  const unsigned char* udata = (const unsigned char*)datum.data().c_str();
  const float* fdata = datum.float_data().data();
  const int channels = datum.channels();

  for (int c = 0; c < channels; ++c) {
      Dtype channel_mean = mean_values_[c];
      for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
              Dtype datum_element;
              Dtype mean_value;
              if (cover_size == crop_size && !is_random_rotate) {
                  int h_in_real = h + h_off;
                  int w_in_real = w + w_off;
                  if (udata) {
                      datum_element = ImageElement(udata, c, h_in_real, w_in_real, height, width);
                  } else {
                      datum_element = ImageElement(fdata, c, h_in_real, w_in_real, height, width);
                  }
                  mean_value = ImageElement(mean, c, h_in_real, w_in_real, height, width) + channel_mean; 
              } else {
                  Dtype fine_h;
                  Dtype fine_w;
                  if (is_random_rotate){
                      Dtype anchor_h = ((Dtype)h - (crop_size - 1) / 2.0) * ratio;
                      Dtype anchor_w = ((Dtype)w - (crop_size - 1) / 2.0) * ratio;
                      fine_h = cos_angle * anchor_h + sin_angle * anchor_w;
                      fine_w = -sin_angle * anchor_h + cos_angle * anchor_w;
                  } else {
                      fine_h = ((Dtype)h - (crop_size - 1) / 2.0) * ratio;
                      fine_w = ((Dtype)w - (crop_size - 1) / 2.0) * ratio;
                  }
                  fine_h += (height - 1) / 2.0 + h_off;
                  fine_w += (width - 1) / 2.0 + w_off;
                  if (udata) {
                      datum_element = BilinearInter(udata, c, fine_h, fine_w, height, width);
                  } else {
                      datum_element = BilinearInter(fdata, c, fine_h, fine_w, height, width);
                  }
                  mean_value = BilinearInter(mean, c, fine_h, fine_w, height, width) + channel_mean;
              }

              int top_index;
              if (is_mirror) {
                  top_index = ((batch_item_id * channels + c) * crop_size + h)
                      * crop_size + (crop_size - 1 - w);
              } else {
                  top_index = ((batch_item_id * channels + c) * crop_size + h)
                      * crop_size + w;
              }
              transformed_data[top_index] =
                  (datum_element - mean_value) * scale_;
          }
      }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const bool random_rotate = param_.random_rotate();
  int cover_size = param_.cover_size();
  int crop_size = param_.crop_size();
  const int height = datum.height();
  const int width = datum.width();
  const int padding = param_.padding();
  const int padded_width = width + padding * 2; // left and right
  const int padded_height = height + padding * 2;
  const bool mirror = param_.mirror();

  if (crop_size == 0) {
	CHECK_EQ(height, width);
	crop_size = height;
  }
  if (cover_size == 0) {
      cover_size = crop_size;
  }
  int h_off, w_off;
  bool is_mirror = false;
  bool is_random_rotate = false;
  if (phase_ == Caffe::TRAIN) {
      h_off = padded_height == cover_size? 0 : (Rand() % (padded_height - cover_size));
      w_off = padded_width == cover_size? 0 : (Rand() % (padded_width - cover_size));
      is_mirror = mirror && Rand() % 2;
      is_random_rotate = random_rotate;
  } else {
      CHECK_EQ(mirror, false);
      h_off = (padded_height - cover_size) / 2;
      w_off = (padded_width - cover_size) / 2;
  }

  if (mean_values_.size() == 0) {
      mean_values_.resize(datum.channels(), 0);
  }

  Transform(batch_item_id, datum, mean, transformed_data, 
          crop_size, cover_size, h_off - padding, w_off - padding, 
          is_mirror, is_random_rotate);

  //{
      //CHECK_EQ(batch_item_id, 0);
      //LOG(INFO) << cover_size << "\t" << crop_size << "\t"
          //<< h_off - padding << "\t" << w_off - padding;
      //FILE* fp = fopen("a.uchar.bin", "wb");
      //int channels = datum.channels();
      //int height = datum.height();
      //int width = datum.width();
      //fwrite(&channels, sizeof(int), 1, fp);
      //fwrite(&height, sizeof(int), 1, fp);
      //fwrite(&width, sizeof(int), 1, fp);
      //fwrite(datum.data().data(), sizeof(unsigned char), channels * height * width, fp); 
      //fclose(fp);

      //fp = fopen("a.float.bin", "wb");
      //fwrite(&channels, sizeof(int), 1, fp);
      //fwrite(&crop_size, sizeof(int), 1, fp);
      //fwrite(&crop_size, sizeof(int), 1, fp);
      //fwrite(transformed_data, sizeof(float), channels * crop_size * crop_size, fp); 
      //fclose(fp);
      //LOG(FATAL);
  //}
  //} 
  //else 
  //{
	  //CHECK_EQ(padding, 0);
      //CHECK_EQ(param_.mean_values_size(), 0);
    //// we will prefer to use data() first, and then try float_data()
    //if (data.size()) {
      //for (int j = 0; j < size; ++j) {
        //Dtype datum_element =
            //static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        //transformed_data[j + batch_item_id * size] =
            //(datum_element - mean[j]) * scale;
      //}
    //} else {
      //for (int j = 0; j < size; ++j) {
        //transformed_data[j + batch_item_id * size] =
            //(datum.float_data(j) - mean[j]) * scale;
      //}
    //}
  //}
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

template <typename Dtype>
Dtype DataTransformer<Dtype>::RandAngle() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  Dtype r = ((*rng)() - (Dtype)rng->min()) / 
      ((Dtype)rng->max() - (Dtype)rng->min());
  return r * 2 * 3.1415926;
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
