

library(tensorflow)
library(tfdatasets)
library(keras)
library(cloudml)

# tfe_enable_eager_execution()

# get reference to data directory
data_dir <- gs_data_dir("gs://speech-commands-data/v0.01")

# initialize class_id_table
class_names <- tf$constant(
  c("bed", "bird", "cat", "dog", "down",  "eight", "five",
    "four", "go", "happy", "house", "left", "marvin", "nine", 
    "no", "off", "on", "one", "right", "seven", "sheila", "six",
    "stop", "three", "tree", "two", "up", "wow", "yes", "zero" )
)
class_id_table <- tf$contrib$lookup$index_table_from_tensor(class_names)
tf$tables_initializer()


audio_ops <- tf$contrib$framework$python$ops$audio_ops

audio_files_dataset <- function(split, batch_size, shuffle = TRUE, 
                                window_size_ms = 30, window_stride_ms = 10) {
  
  window_size <- as.integer(16000*window_size_ms/1000)
  stride <- as.integer(16000*window_stride_ms/1000)
  fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
  n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))
  
  ds <- text_line_dataset(file.path(data_dir, split))
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  ds <- ds %>%
    dataset_map(num_parallel_calls = 4, function(file) {
      
      # form full path to file
      path <- tf$string_join(list(data_dir, file), separator = "/")
      
      # decoding wav files
      audio_binary <- tf$read_file(path)
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1L)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio, 
        window_size = window_size, 
        stride = stride,
        magnitude_squared = TRUE
      )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # extract class name and lookup id
      class_name <- tf$string_split(list(file), delimiter = "/")$values[[1]]
      class_id <- class_id_table$lookup(class_name)
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(class_id, 30L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(n_chunks, fft_size, NULL), shape(NULL)),
                         drop_remainder = TRUE)
  
  ds
}

ds_train <- audio_files_dataset("split_training.txt", 32L)
ds_test <- audio_files_dataset("split_testing.txt", 32, shuffle = FALSE)


model <- keras_model_sequential()
model %>%  
  layer_conv_2d(input_shape = c(98, 257, 1), 
                filters = 32, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 30, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train model
model %>% fit(
  ds_train,
  steps_per_epoch = 1000,
  epochs = 10, 
  validation_data = ds_test, 
  validation_steps = 1000
)

save_model_hdf5(model, "model.hdf5")




