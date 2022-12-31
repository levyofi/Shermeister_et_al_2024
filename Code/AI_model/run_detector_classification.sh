#!/bin/bash
## make sure you have the gnu parallel package : sudo apt-get install parallel

GPU=$2
N=$1

# shellcheck disable=SC2051
for ((i=1;i<=$N;i++)); do
    gnome-terminal --tab --title="lizard_detection$i" -- /bin/bash -c "export CUDA_VISIBLE_DEVICES=$GPU; python detect_and_classify.py; exec bash;"
done
