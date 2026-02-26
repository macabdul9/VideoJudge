import re
import os
import json
import datasets
from videojudge.utils import (
    load_generator, 
    video_to_text_generation as default_video_to_text_generation,
    load_vllm_generator,
    video_to_text_generation_vllm
)
from tqdm import tqdm


def format_prompt(prompt_template, instruction, response, video_file):
    """
    Format the prompt with the given instruction, response, and video file.
    """
    return prompt_template.format(instruction=instruction, response=response, video_file=video_file)
    


def main(args):
    
    # Load the dataset
    dataset = datasets.load_dataset(args.dataset_name_or_path, split=args.split)
    
    if args.debug:
        # For debugging, limit the dataset to 100 examples
        dataset = dataset.select(range(10))
    
    # Load the evaluator
    if args.use_vllm:
        evaluator = load_vllm_generator(
            model_type=args.model_type, 
            model_name_or_path=args.model_name_or_path, 
            tensor_parallel_size=args.tensor_parallel_size,
        )
        video_to_text_generation = video_to_text_generation_vllm
    else:
        evaluator = load_generator(
            model_type=args.model_type, 
            model_name_or_path=args.model_name_or_path, 
            device=args.device
        )
        video_to_text_generation = default_video_to_text_generation
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompt template
    with open(args.prompt_fp, 'r') as f:
        prompt_template = f.read().strip()
    
    # Output file - model_name_dataset_name_prompt_name_temp_
    output_file = os.path.join(args.output_dir, f"{args.model_name_or_path.split('/')[-1]}_{args.dataset_name_or_path.split('/')[-1]}_{args.split}_prompt_{args.prompt_fp.split('/')[-1].split('.')[0]}_do_sample_{args.do_sample}_temp_{args.temperature}.json")

    output = {}
    
    # load the output file if it exists
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            output = json.load(f)
    
    
    params = {
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature if hasattr(args, 'temperature') else 0.01,
    }
    
    
    # Iterate through the dataset
    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        
        # try:
        if example[args.id_column] in output:
            continue

        # Prepare prompt
        prompt = prompt_template.format(instruction=example[args.instruction_column], response=example[args.response_column])

        # Prepare the messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video",
                        "video": f"{args.video_dir}/{example[args.video_column]}",
                        "fps": args.fps,  # Frames per second for video processing
                        "max_frames": args.max_frames,  # Maximum number of frames to process
                        "total_pixels": args.max_pixels,  # Maximum pixel count for the video frames
                        "min_pixels": args.min_pixels,  # Minimum pixel count for the video frames
                    },
                    
                    
                ],
            }
        ]
        
        response = video_to_text_generation(
            model_type=args.model_type,
            generator=evaluator,
            messages=messages,
            params=params,
        )

        
        # We need to extract the number from the response using regex
        match = re.search(r'\d+(\.\d+)?', response)
        predicted_rating = None
        if match:
            predicted_rating = match.group(0)
            

        # Store the response
        output[example[args.id_column]] = {
            "instruction": example[args.instruction_column],
            "response": example[args.response_column],
            "video_file": example[args.video_column],
            "true_rating": example[args.rating_column],
            "predicted_rating": predicted_rating,
            "eval_output": response,
        }
        
        if idx % args.checkpoint_step == 0:
            # Save the output to file every checkpoint_step
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=4)
                
        # except Exception as e:
        #     print(f"Error processing example {idx}: {e}")
        #     continue
    
    # Save the final output to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"Evaluation completed. Results saved to {output_file}")
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video generation models.")
    
    # Model parameters
    parser.add_argument("--use_vllm", action="store_true", help="Use vllm for evaluation.")
    parser.add_argument("--model_type", type=str, default="qwenvlm", choices=["qwenvlm", "smolvlm"], help="Type of the model to evaluate.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Path to the model or model name.")
    
    # Dataset parameters
    parser.add_argument("--dataset_name_or_path", type=str, default="uEval/VideoJudgeVCG-MetaEval", help="Path to the dataset or dataset name.")
    parser.add_argument("--split", type=str, default="testmini", help="Split of the dataset to be used in evaluation.")
    parser.add_argument("--video_dir", type=str, default="data/VideoChatGPT/Test_Videos", help="Directory containing the video files.")
    
    parser.add_argument("--instruction_column", type=str, default="instruction", help="Column name for instructions in the dataset.")
    parser.add_argument("--response_column", type=str, default="response", help="Column name for responses in the dataset.")
    parser.add_argument("--rating_column", type=str, default="gen_rating", help="Column name for ratings in the dataset.")
    parser.add_argument("--video_column", type=str, default="video_file", help="Column name for video paths in the dataset.")
    parser.add_argument("--id_column", type=str, default="unique_id", help="Column name for unique IDs in the dataset.")     
    
    
    
    # evaluation model - [pointwise, pairwise]
    parser.add_argument("--evaluation_model", type=str, default="pointwise", choices=["pointwise", "pairwise"], help="Type of evaluation model to use.")
    
    # Decoding parameters LLM -- other parameters should use default values
    parser.add_argument("--temperature", type=float, default=0.01, help="Temperature for sampling.")
    parser.add_argument("--do_sample", action="store_true", help="Whether to do sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    
    

    # Parameters for Videos -- default values should be used. 
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for video processing. Default value for Qwen2.5-VL is 2. https://github.com/QwenLM/Qwen2.5-VL")
    parser.add_argument("--max_frames", type=int, default=180, help="Maximum number of frames to process from the video.")
    parser.add_argument("--max_pixels", type=int, default=20480 * 28 * 28, help="Maximum number of pixels for the video frames.")
    parser.add_argument("--min_pixels", type=int, default=16 * 28 * 28, help="Minimum number of pixels for the video frames.")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
 
    
    # Which prompt to use
    parser.add_argument("--prompt_fp", type=str, default="prompts/evaluation_model.txt", help="Path to the prompt file.")  
    
    
    # Device parameters
    parser.add_argument("--device", type=str, default="auto", help="Device to run the evaluation on.")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="Tensor parallel size for vllm. Use all available GPUs if None.")
    
    
    # Checkpoint and output parameters
    parser.add_argument("--checkpoint_step", type=int, default=10, help="Checkpoint step for saving intermediate results.")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the evaluation results.")


    args = parser.parse_args()
    
    main(args)