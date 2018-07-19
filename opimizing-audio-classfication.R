

library(keras)
library(tensorflow)
library(tfdatasets)
library(cloudml)

# flags which control local vs. cloudml training behavior
FLAGS <- flags(
  flag_boolean("data_gs", FALSE),
  flag_string("data_gs_bucket", "gs://speech-commands-data/v0.01")
)

# determine data_dir based on whether we are using local or gs bucket data
# (if local then rsync from google storage)
if (FLAGS$data_gs) {
  data_dir <- FLAGS$data_gs_bucket
} else {
  data_dir <- "data/speech_commands_v0.01"
  if (!dir.exists(data_dir))
    gs_rsync(FLAGS$data_gs_bucket, data_dir, recursive = TRUE)
}

tfe_enable_eager_execution()

library(stringr)
library(dplyr)
library(fs)

# initialize classes and factors
classes <- tf$constant(
  c("bed", "bird", "cat", "dog", "down",  "eight", "five",
    "four", "go", "happy", "house", "left", "marvin", "nine", 
    "no", "off", "on", "one", "right", "seven", "sheila", "six",
    "stop", "three", "tree", "two", "up", "wow", "yes", "zero" )
)
class_id_table <- tf$contrib$lookup$index_table_from_tensor(classes)
tf$tables_initializer()


# start with all directories (one for each class)
classes_dataset <- tensor_slices_dataset(classes)

# function which maps a directory to the shuffled records in those files
per_class_dataset <- function(class) {
  
  # list files in directory
  glob <- tf$string_join(list(data_dir, class, "*.wav"), separator = "/")
  files <- file_list_dataset(glob, shuffle = TRUE) 
  
  # determine class id
  class_id <- class_id_table$lookup(class)
  
  # create dataset with filename, class, and class id
  zip_datasets(
    files,
    tensors_dataset(class) %>% dataset_repeat(),
    tensors_dataset(class_id) %>% dataset_repeat()
  )
}

# interleave all of the per-class datasets
audio_files_dataset <- classes_dataset %>% 
  dataset_interleave(per_class_dataset, cycle_length = length(classes))




## TODO: use updated dataset (see README)
## https://stackoverflow.com/questions/50356677/how-to-create-tf-data-dataset-from-directories-of-tfrecords




files <- file_list_dataset(file.path(data_dir, "*.wav"))


files <- dir_ls(
  path = data_dir, 
  recursive = TRUE, 
  glob = "*.wav"
)

files <- files[!str_detect(files, "background_noise")]

df <- data_frame(
  fname = files, 
  class = fname %>% str_extract("1/.*/") %>% 
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

saveRDS(df, "data/df.rds")

audio_ops <- tf$contrib$framework$python$ops$audio_ops

data_generator <- function(df, batch_size, shuffle = TRUE, 
                           window_size_ms = 30, window_stride_ms = 10) {
  
  window_size <- as.integer(16000*window_size_ms/1000)
  stride <- as.integer(16000*window_stride_ms/1000)
  fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
  n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))
  
  ds <- tensor_slices_dataset(df)
  
  if (shuffle) 
    ds <- ds %>% dataset_shuffle(buffer_size = 100)  
  
  ds <- ds %>%
    dataset_map(function(fname, class, class_id) {
      
      # decoding wav files
      audio_binary <- tf$read_file(tf$reshape(fname, shape = list()))
      wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
      
      # create the spectrogram
      spectrogram <- audio_ops$audio_spectrogram(
        wav$audio, 
        window_size = window_size, 
        stride = stride,
        magnitude_squared = TRUE
      )
      
      spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
      spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
      
      # transform the class_id into a one-hot encoded vector
      response <- tf$one_hot(class_id, 30L)
      
      list(spectrogram, response)
    }) %>%
    dataset_repeat()
  
  ds <- ds %>% 
    dataset_padded_batch(batch_size, list(shape(n_chunks, fft_size, NULL), shape(NULL)))
  
  ds
}


df <- readRDS("data/df.rds") %>% sample_frac(1)
id_train <- sample(nrow(df), size = 0.7*nrow(df))

ds_train <- data_generator(df[id_train,], 32L)
ds_test <- data_generator(df[-id_train,], 32, shuffle = FALSE)


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
model %>% fit_generator(
  generator = ds_train,
  steps_per_epoch = 0.7*nrow(df)/32,
  epochs = 10, 
  validation_data = ds_test, 
  validation_steps = 0.3*nrow(df)/32
)

save_model_hdf5(model, "model.hdf5")




