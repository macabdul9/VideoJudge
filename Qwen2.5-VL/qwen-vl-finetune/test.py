import datasets
import os
import json


# root_dir = "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback"

# prefix = "videojudgepreferencewithfeedback"

root_dir = "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference"
prefix = "videojudgepreference"

count = 0
for i in range(10):

    with open(f'{root_dir}/{prefix}{i}.json', "r") as file:
        data = json.load(file)

    # import pdb;pdb.set_trace()
    count += len(data)

print(count)
