#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int padding = param_.padding();
  const int padded_width = width + padding * 2; // left and right
  const int padded_height = height + padding * 2;

  int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) 
  {
    //LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               //<< "set at the same time.";
	CHECK_EQ(height, width);
	crop_size = height;
  }

  if (crop_size) 
  {
	  //CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
	  h_off = padded_height == crop_size? 0 : (Rand() % (padded_height - crop_size));
	  w_off = padded_width == crop_size ? 0 : (Rand() % (padded_width - crop_size));
    } else {
      h_off = (padded_height - crop_size) / 2;
      w_off = (padded_width - crop_size) / 2;
    }
	const bool is_mirror = mirror && Rand() % 2;
	for (int c = 0; c < channels; ++c) 
	{
		for (int h = 0; h < crop_size; ++h) 
		{
			for (int w = 0; w < crop_size; ++w) 
			{
				Dtype datum_element;
				Dtype mean_value;
				int h_in_real = h + h_off - padding;
				int w_in_real = w + w_off - padding;
				if (h_in_real < 0 || w_in_real < 0 || h_in_real >= height || w_in_real >= width)
				{
					datum_element = 0;
					mean_value = 0;
				}
				else
				{
					int data_index = (c * height + h_in_real) * width + w_in_real;
					if (data.size())
					{
						datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
					}
					else
					{
						datum_element = datum.float_data(data_index);
					}
					mean_value = mean[data_index];
				}
				int top_index;
				if (is_mirror)
				{
					top_index = ((batch_item_id * channels + c) * crop_size + h)
						* crop_size + (crop_size - 1 - w);
				}
				else
				{
					top_index = ((batch_item_id * channels + c) * crop_size + h)
						* crop_size + w;
				}
				transformed_data[top_index] =
					(datum_element - mean_value) * scale;
			}
		}
	}
  } 
  else 
  {
	  CHECK_EQ(padding, 0);
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
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

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
