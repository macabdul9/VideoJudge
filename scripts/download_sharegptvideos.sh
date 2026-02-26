#!/bin/bash

# Set directory name from input argument or default to 'VATEX-EVAL'
# DIR=${1:-VATEX-EVAL}

# Download the dataset to the specified directory
# huggingface-cli download ShareGPTVideo/train_video_and_instruction --repo-type dataset --local-dir "$DIR"

# # Change to the specified directory
# cd "$DIR" || exit

# # Merge all the parts into a single zip file and then unzip it
# cat VATEX_EVAL_VIDEOS_part_*.part > VATEX_EVAL_VIDEOS.zip

# mkdir -p VATEX_VIDEOS 
# unzip VATEX_EVAL_VIDEOS.zip -d VATEX_VIDEOS/

# # ── 1.  jump to the folder that holds the tarballs
# cd /storage1/abdulw/VideoAutoEval/sharegpt_train_raw_video/activitynet

# # ── 2.  make the destination directory (call it “videos” one level up)
# mkdir -p ../videos          #  …/sharegpt_train_raw_video/videos

# # ── 3.  extract every chunk into that videos/ dir
# for f in chunk_*.tar.gz; do
#     echo "▶︎ extracting $f"
#     tar -xzf "$f" -C ../videos          # drop “--strip-components=1” in front of -xzf if you need to shave off the top-level folder stored inside each tar
# done


# # 1) go into the folder that holds the 600 k chunks
# cd /storage1/abdulw/VideoAutoEval/sharegpt_videos/train_600k

# # 2) create a common destination ­– here: …/sharegpt_videos/videos
# mkdir -p ../videos            # adjust if you prefer another name

# # 3) extract every tarball into that videos/ directory
# for f in chunk_*.tar.gz; do
#     echo "▶︎ $f"
#     tar -xzf "$f" -C ../videos          # add --strip-components=1 right after tar if you need to drop the top-level path stored inside each archive
# done




# 1) go to the folder that holds the .tar files
cd /storage1/abdulw/VideoAutoEval/video-chatgpt-videos

# 2) (optional) one destination for everything
mkdir -p ../VideoGPTVideos          # change this if you like

# 3) extract every archive, skipping anything that was already unpacked
for f in activitynet_train_videos_video_chatgpt_*.tar; do
    echo "▶︎ extracting $f"
    tar -xvf "$f" -C ../VideoGPTVideos
done
