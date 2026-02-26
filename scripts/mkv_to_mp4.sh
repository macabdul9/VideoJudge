# Count files in a directory

# find data/VideoChatGPT/Test_Videos -type f | wc -l

# with a particular extension
## find data/VideoChatGPT/Test_Videos -type f -name "*.mkv" | wc -l

# find data/VideoGPTVideos -type f | wc -l


# # Convert all .mkv files in the specified directory to .mp4 format
# for f in data/VideoChatGPT/Test_Videos/*.mkv; do
#   ffmpeg -i "$f" -c copy "${f%.mkv}.mp4"
# done


find data/VideoGPTVideos/*.mkv -type f | wc -l

# Convert all .mkv files in the specified directory to .mp4 format
for f in data/VideoGPTVideos/*.mkv; do
  ffmpeg -i "$f" -c copy "${f%.mkv}.mp4"
done


find data/sharegpt_train_raw_video/videos/ -type f | wc -l
