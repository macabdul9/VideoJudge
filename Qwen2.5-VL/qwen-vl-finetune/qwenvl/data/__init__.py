import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "/home/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/demo/video.json",
    "data_path": "/home/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/demo/videos",
}

VIDEOJUDGE0 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_0.json",
    "data_path": "",
}
VIDEOJUDGE1 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_1.json",
    "data_path": "",
}
VIDEOJUDGE2 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_2.json",
    "data_path": "",
}
VIDEOJUDGE3 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_3.json",
    "data_path": "",
}
VIDEOJUDGE4 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_4.json",
    "data_path": "",
}
VIDEOJUDGE5 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_5.json",
    "data_path": "",
}
VIDEOJUDGE6 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_6.json",
    "data_path": "",
}
VIDEOJUDGE7 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_7.json",
    "data_path": "",
}
VIDEOJUDGE8 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_8.json",
    "data_path": "",
}
VIDEOJUDGE9 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_9.json",
    "data_path": "",
}
VIDEOJUDGE10 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudge/VideoJudge-RS-20K_10.json",
    "data_path": "",
}

# ---------------- VideoJudgePreference ----------------
VIDEOJUDGEPREFERENCE0 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference0.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE1 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference1.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE2 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference2.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE3 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference3.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE4 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference4.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE5 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference5.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE6 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference6.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE7 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference7.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE8 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference8.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE9 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference9.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE10 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference10.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE11 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference11.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE12 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference12.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE13 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference13.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE14 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference14.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE15 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference15.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE16 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference16.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE17 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference17.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE18 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference18.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE19 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference19.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCE20 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreference/videojudgepreference20.json",
    "data_path": "",
}

# ---------------- VideoJudgePreferenceWithFeedback ----------------
VIDEOJUDGEPREFERENCEWITHFEEDBACK0 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback0.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK1 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback1.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK2 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback2.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK3 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback3.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK4 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback4.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK5 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback5.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK6 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback6.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK7 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback7.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK8 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback8.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK9 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback9.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK10 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback10.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK11 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback11.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK12 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback12.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK13 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback13.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK14 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback14.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK15 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback15.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK16 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback16.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK17 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback17.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK18 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback18.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK19 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback19.json",
    "data_path": "",
}
VIDEOJUDGEPREFERENCEWITHFEEDBACK20 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgePreferenceWithFeedback/videojudgepreferencewithfeedback20.json",
    "data_path": "",
}

# VIDEO Judge with Rubric
VIDEOJUDGEWITHRUBRIC0 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric0.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC1 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric1.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC2 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric2.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC3 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric3.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC4 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric4.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC5 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric5.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC6 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric6.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC7 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric7.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC8 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric8.json",
    "data_path": "",
}
VIDEOJUDGEWITHRUBRIC9 = {
    "annotation_path": "/opt/app-root/src/users/abdulw/VideoAutoEval/Qwen2.5-VL/qwen-vl-finetune/data/VideoJudgeWithRubric/videojudgewithrubric9.json",
    "data_path": "",
}

# ---------------- Data dict update ----------------
data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "videojudge0": VIDEOJUDGE0,
    "videojudge1": VIDEOJUDGE1,
    "videojudge2": VIDEOJUDGE2,
    "videojudge3": VIDEOJUDGE3,
    "videojudge4": VIDEOJUDGE4,
    "videojudge5": VIDEOJUDGE5,
    "videojudge6": VIDEOJUDGE6,
    "videojudge7": VIDEOJUDGE7,
    "videojudge8": VIDEOJUDGE8,
    "videojudge9": VIDEOJUDGE9,
    "videojudge10": VIDEOJUDGE10,
    # add preference datasets
    **{f"videojudgepreference{i}": globals()[f"VIDEOJUDGEPREFERENCE{i}"] for i in range(0, 21)},
    **{f"videojudgepreferencewithfeedback{i}": globals()[f"VIDEOJUDGEPREFERENCEWITHFEEDBACK{i}"] for i in range(0, 21)},
    # add videojudgewithrubric datasets
    **{f"videojudgewithrubric{i}": globals()[f"VIDEOJUDGEWITHRUBRIC{i}"] for i in range(0, 10)},
}




def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
