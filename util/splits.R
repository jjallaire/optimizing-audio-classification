
# directory containing wav files
data_dir <- "data/speech_commands_v0.01"


# enumerate all wav files
wav_files <- list.files(data_dir, pattern = glob2rx("*.wav"), recursive = TRUE)
wav_files <- wav_files[!startsWith(wav_files, "_background_noise_")]

# determine lists of testing/validation/training files based on provided lists
testing_wav_files <- readLines(file.path(data_dir, "testing_list.txt"))
validation_wav_files <- readLines(file.path(data_dir, "validation_list.txt"))
training_wav_files <- setdiff(wav_files, c(testing_wav_files, validation_wav_files))

# shuffle the lists
testing_wav_files <- sample(testing_wav_files)
validation_wav_files <- sample(validation_wav_files)
training_wav_files <- sample(training_wav_files)

# write the splits
writeLines(testing_wav_files, file.path(data_dir, "split_testing.txt"))
writeLines(validation_wav_files, file.path(data_dir, "split_validation.txt"))
writeLines(training_wav_files, file.path(data_dir, "split_training.txt"))



