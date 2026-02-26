# 1. On the login node, forward a local port to the compute node’s SSH:
# ssh -L 2222:babel-2-29:22 abdulw@babel.lti.cs.cmu.edu

# 2. In the same terminal (now with the tunnel open), run:
# rsync -avz \
#   -e 'ssh -p 2222' \
#   VideoChatGPT.zip \
#   abdulw@localhost:/data/hf_cache/VideoAutoEval/videos/

# rsync -avz -e 'ssh -p 2222' VideoChatGPT.zip abdulw@localhost:/data/hf_cache/VideoAutoEval/videos/

# From your HPC (aiva):
# rsync -avz --progress -e 'ssh -J abdulw@babel.lti.cs.cmu.edu' \
#   VideoChatGPT.zip \
#   abdulw@babel-2-29:/data/hf_cache/VideoAutoEval/videos/

# From your HPC (aiva):
# tar czf - VideoChatGPT.zip | \
#   ssh abdulw@babel.lti.cs.cmu.edu "ssh babel-2-29 'tar xzf - -C /data/hf_cache/VideoAutoEval/videos/'"

# tar -C /storage1/abdulw/VideoAutoEval -czf - . \
#   | ssh -A -J abdulw@babel.lti.cs.cmu.edu abdulw@babel-2-29 \
#       "mkdir -p /data/hf_cache/VideoAutoEval/videos && \
#        tar -C /data/hf_cache/VideoAutoEval/videos -xzvf -"




scp -avz --progress -e 'ssh -J abdulw@babel.lti.cs.cmu.edu' data/TrainingVideos.zip awaheed@narval.computecanada.ca:/home/awaheed/scratch/

rsync -avz --progress -e ssh \
      data/TrainingVideos.zip \
      awaheed@narval.computecanada.ca:/home/awaheed/scratch/
