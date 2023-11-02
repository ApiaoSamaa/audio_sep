#!/bin/bash

# Directory containing the dataset
DATASET_DIR="musdb18"
SAVE_DIR="musdb18_wav"

channels=("0" "1" "2" "3" "4")
labels=("mixture" "drums" "bass" "accompaniment" "vocals")

# Loop over directories: train and test
for dir in "train" "test"; do
    if [ ! -d "$HOME/proj/audioSep/$SAVE_DIR/$dir/Mixtures" ]; then
        mkdir -p "./$SAVE_DIR/$dir/Mixtures"
    fi
    if [ ! -d "$HOME/proj/audioSep/$SAVE_DIR/$dir/Sources" ]; then
        mkdir -p "./$SAVE_DIR/$dir/Sources"
    fi

    cd "$HOME/proj/$DATASET_DIR/$dir"
    # Navigate to the directory
    for file in *.mp4; do
        for idx in "${!channels[@]}"; do
        if [ "$idx" -eq "0" ]; then # remember between the 0 and ] need a space.
            ffmpeg -i "$file" -map 0:"${channels[idx]}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "$HOME/proj/audioSep/$SAVE_DIR/$dir/Mixtures/${file%.*}_${labels[idx]}.wav"
        else
            ffmpeg -i "$file" -map 0:"${channels[idx]}" -vn -acodec pcm_s16le -ar 44100 -ac 1 "$HOME/proj/audioSep/$SAVE_DIR/$dir/Sources/${file%.*}_${labels[idx]}.wav"
        fi
        done
    done
done
