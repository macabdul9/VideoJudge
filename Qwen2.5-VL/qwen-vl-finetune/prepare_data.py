import os
import json
import pandas as pd
import datasets
from tqdm import tqdm


def main(args):
    
   
    # Load the datsets
    dataset = datasets.load_dataset(args.dataset_name_or_path, split="train")
    
    print(f"Loaded dataset: {args.dataset_name_or_path} with {len(dataset)} examples.")

    # load the corrupted videos list
    corrupted_video_files = [video.split("/")[-1] for video in open(args.corrupted_list_of_videos, "r").read().splitlines()]
    
    # import pdb; pdb.set_trace()
    
    # filter the dataset to remove corrupted videos
    dataset = dataset.filter(lambda x: x[args.video_file] not in corrupted_video_files)
    
    # also remove the example if video_file (without extension) video_file.mkv exist
    dataset = dataset.filter(lambda x: not os.path.exists(f"{args.video_root_dir}/{x[args.video_file].split('.')[0]}.mkv"))
    
    print(f"Filtered dataset: {args.dataset_name_or_path} with {len(dataset)} examples after removing corrupted videos.")

    dataset_name = args.dataset_name_or_path.split("/")[-1]
    
    # import pdb; pdb.set_trace()
    
    # create a directory for the output if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    # Load the prompt file
    with open(args.prompt_fp, "r") as f:
        prompt_template = f.read()
    
    
    # Prepare the data for LLaMA-Factory
    prepared_data = []
    
    # iterate through the dataset and prepare the data
    
    chunk_id = 0
    
    for example in tqdm(dataset, desc="Preparing VideoJudge data"):
        
        # Skip the example if the video file is in the corrupted list
        if example[args.video_file] in corrupted_video_files:
            continue
        
        id = example[args.unique_id]

    
        data = {}

        instruction = example[args.instruction_column]
        response = example[args.response_column]
        prompt = prompt_template.format(instruction=instruction, response=response)

        reasoning = example[args.reasoning_column]
        rating = example[args.rating_column]

        output = f"{reasoning}\n<score>{rating}</score>"

    
        video_file = f'{args.video_root_dir}/{example[args.video_file]}'


        data = {
            "video": video_file,
            "conversations": [
                {
                    "from": "human",
                    "value": f'<video>\n{prompt}'
                },
                {
                    "from": "gpt",
                    "value": output
                }
            ]
        }
        
        prepared_data.append(data)
        
        # if the dataset is large, you can save the data in chunks
        if len(prepared_data) % args.chunk_size == 0:
            output_file = os.path.join(args.output_dir, f"{dataset_name}_{chunk_id}.json")
            with open(output_file, "w") as f:
                json.dump(prepared_data, f, indent=4)
            print(f"Saved {len(prepared_data)} examples to {output_file}")
            prepared_data = []
            chunk_id += 1

    # Save the prepared data to a JSON file
    output_file = os.path.join(args.output_dir, f"{dataset_name}_{chunk_id}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(prepared_data, f, indent=4)
    
    print(f"Prepared data saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare VideoJudge data for LLaMA-Factory.")
        
    parser.add_argument("--dataset_name_or_path", type=str, default="uEval/VideoJudge-RS-20K", help="Name or path of the dataset to be used in LLaMA-Factory.")
    
    parser.add_argument("--unique_id", type=str, default="unique_id", help="Column name for unique IDs in the CSV file.")
    parser.add_argument("--video_file", type=str, default="video_file", help="Column name for video paths in the CSV file.")
    parser.add_argument("--instruction_column", type=str, default="instruction", help="Column name for instructions in the CSV file.")
    parser.add_argument("--response_column", type=str, default="response", help="Column name for responses in the CSV file.")
    parser.add_argument("--reasoning_column", type=str, default="eval_reasoning", help="Column name for reasoning in the CSV file.")
    parser.add_argument("--rating_column", type=str, default="gen_rating", help="Column name for ratings in the CSV file.")
    parser.add_argument("--description_column", type=str, default="description", help="Column name for descriptions in the CSV file.")
    parser.add_argument("--video_root_dir", type=str, default="/opt/app-root/src/users/abdulw/VideoAutoEval/data/TrainingVideos", help="Root directory for video files.")
    
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of examples to process in each chunk.")
    
    parser.add_argument("--prompt_fp", type=str, default="/opt/app-root/src/users/abdulw/VideoAutoEval/prompts/evaluation_model.txt", help="Path to the prompt file for VideoJudge data.")
    
    parser.add_argument("--corrupted_list_of_videos", type=str, default="small_videos.txt", help="Path to the list of corrupted videos.")

    parser.add_argument("--output_dir", type=str, default="data/VideoJudge", help="Directory to save the processed VideoJudge data.")
    
    args = parser.parse_args()
    main(args)