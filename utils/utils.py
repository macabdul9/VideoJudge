
import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv
import av
import numpy as np
import base64
from PIL import Image
from io import BytesIO


from transformers.image_utils import load_image
from qwen_vl_utils import process_vision_info

from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForCausalLM,
    AutoTokenizer

)
from vllm import LLM, SamplingParams


import torch

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")

# Set up headers for the request
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

def format_message(messages, role, prompt):
    messages.append({
        "role": role,
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    })
    return messages

def create_llm_payload(messages, parameters=None):
    if parameters is None:
        parameters = {}
    payload = {
        "messages": messages,
        "temperature": parameters.get("temperature", 0.6),
        "max_tokens": parameters.get("max_tokens", 1000),
    }
    return payload

async def async_generator(payload, max_retries=3, backoff=3):
    """
    Asynchronous helper function to send a POST request to the LLM endpoint 
    using aiohttp, with simple retry logic.

    Args:
        payload (dict): JSON payload for the POST request.
        max_retries (int): Number of times to retry on failure.
        backoff (int): Seconds to wait between retries.

    Returns:
        str: The 'content' from the LLM response.
    """
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(ENDPOINT, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    # Return the message's content from the first choice
                    return response_data["choices"][0]["message"]["content"]

        except Exception as e:
            # If we haven't exhausted retries, wait and try again
            if attempt < max_retries - 1:
                print(f"[async_generator] Request failed (attempt {attempt+1}). "
                      f"Retrying in {backoff} seconds...")
                await asyncio.sleep(backoff)
            else:
                # If we've exhausted our retries, re-raise the error
                raise e

def generator(payload):
    """
    Synchronous wrapper that calls the async_generator using asyncio.run().
    This lets the rest of the code remain synchronous, but under the hood,
    the call is performed asynchronously with retries.
    """
    return asyncio.run(async_generator(payload=payload))

def read_video_pyav_base64(video_path, fps=3, max_frames=250):
    container = av.open(video_path)
    stream = container.streams.video[0]

    # Fallback if duration is None
    if stream.duration is not None:
        duration = float(stream.duration * stream.time_base)
    else:
        # Estimate duration from total frames and frame rate
        frame_rate = float(stream.average_rate) if stream.average_rate is not None else 25.0
        duration = stream.frames / frame_rate if stream.frames else 10.0  # fallback default

    # Decide timestamps to sample
    est_total = int(duration * fps)
    if est_total <= max_frames:
        timestamps = np.arange(0, duration, 1 / fps)
    else:
        timestamps = np.linspace(0, duration, max_frames, endpoint=False)

    base64_frames = []
    ts_index = 0

    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        current_time = float(frame.pts * stream.time_base)
        if ts_index >= len(timestamps):
            break
        if current_time >= timestamps[ts_index]:
            img_array = frame.to_ndarray(format="rgb24")
            image = Image.fromarray(img_array)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_frames.append(img_str)
            ts_index += 1

    return base64_frames


def load_smolvlm(model_name_or_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct", device="auto"):
    
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map=device,
    )

    return {"processor": processor, "model": model}


def smolvlm_video_to_text_generation(generator, messages, params):
    inputs = generator["processor"].apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(generator["model"].device, dtype=torch.bfloat16)

    generated_ids = generator["model"].generate(**inputs, **params)
    generated_texts = generator["processor"].batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    # return generated_texts
    return generated_texts[0].split("Assistant:")[1].strip()
    
def smolvlm_image_to_text_generation(generator, messages, params):
    
    inputs = generator["processor"].apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(generator["model"].device, dtype=torch.bfloat16)

    generated_ids = generator["model"].generate(**inputs, **params)
    generated_texts = generator["processor"].batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts
    

def load_qwenvlm(model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", device="auto"):
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    return {"processor": processor, "model": model}

def qwenvlm_image_to_text_generation(generator, messages, params):
    
    model, processor = generator["model"], generator["processor"]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, **params)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text
    
def qwenvlm_video_to_text_generation(generator, messages, params):
    
    # inference is same as image to text
    return qwenvlm_image_to_text_generation(generator, messages, params)[0]


def load_generator(model_type, model_name_or_path, device):
    
    loader_fn = eval(f'load_{model_type}')
    
    evaluator = loader_fn(model_name_or_path=model_name_or_path, device=device)
    
    return evaluator

def video_to_text_generation(model_type, generator, messages, params):
    
    generator_fn = eval(f'{model_type}_video_to_text_generation')
    
    generated_texts = generator_fn(generator=generator, messages=messages, params=params)
    
    return generated_texts
    

def format_message(messages, role, prompt):
    messages.append({
        "role": role,
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    })
    return messages


def load_qwenvlm_vllm(model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", tensor_parallel_size=1):
    """
    Load the Qwen2.5-VL model using vLLM for efficient inference.
    
    Args:
        model_name_or_path (str): Path or name of the model to load.
        tensor_parallel_size (int): Number of GPUs to use for tensor parallelism.
        device (str): Device to load the model on, e.g., "cuda" or "cpu".
    
    Returns:
        LLM: An instance of the vLLM LLM class with the loaded model.
    """
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,  # Adjust based on your GPU setup
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    return {
        "llm": llm,
        "processor": processor,
    }
    
def qwenvlm_video_to_text_generation_vllm(generator, messages, params):
    """
    Generate text from video input using the Qwen2.5-VL model with vLLM.
    
    Args:
        generator (dict): Contains the vLLM instance and processor.
        messages (list): List of messages containing video input.
        params (dict): Parameters for text generation, e.g., temperature, max_tokens.
    
    Returns:
        str: Generated text response from the model.
    """
    llm, processor = generator["llm"], generator["processor"]
    
    sampling_params = SamplingParams(
        temperature=params.get("temperature", 0.0),
        max_tokens=params.get("max_new_tokens", 1024),
        n=params.get("n", 1),
        stop_token_ids=[],
    )
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)

    texts = [cand.text.strip() for cand in outputs[0].outputs]

    if sampling_params.n > 1 and len(texts) == 1:
        print(f"[WARN] requested n={sampling_params.n} but got {len(texts)} candidate")

    return texts if sampling_params.n > 1 else (texts[0] if texts else "")

def load_vllm_generator(model_type, model_name_or_path, tensor_parallel_size=None):
    """
    Load a vLLM generator for the specified model type.
    
    Args:
        model_type (str): Type of the model, e.g., "qwenvlm".
        model_name_or_path (str): Path or name of the model to load.
        tensor_parallel_size (int, optional): Number of GPUs to use for tensor parallelism.
    
    Returns:
        dict: A dictionary containing the vLLM instance and processor.
    """
    loader_fn = eval(f'load_{model_type}_vllm')
    
    generator = loader_fn(model_name_or_path=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
    
    return generator

def video_to_text_generation_vllm(model_type, generator, messages, params):
    """
    Generate text from video input using a vLLM generator.
    
    Args:
        model_type (str): Type of the model, e.g., "qwenvlm".
        generator (dict): Contains the vLLM instance and processor.
        messages (list): List of messages containing video input.
        params (dict): Parameters for text generation, e.g., temperature, max_tokens.
    
    Returns:
        str: Generated text response from the model.
    """
    generator_fn = eval(f'{model_type}_video_to_text_generation_vllm')
    
    generated_texts = generator_fn(generator=generator, messages=messages, params=params)
    
    return generated_texts


def load_qwenllm_vllm(model_name_or_path, tensor_parallel_size=4, gpu_memory_utilization=0.5):
    llm = LLM(model=model_name_or_path, generation_config="auto", tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
    return llm

def qwenllm_vllm_text_to_text_generation(generator, prompts, sampling_params, enable_thinking):
    outputs = generator.generate(prompts, sampling_params)
    return

def load_qwenllm(model_name_or_path):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    return {"tokenizer":tokenizer, "model":model}

def qwenllm_text_to_text_generation(generator, prompts, sampling_params, enable_thinking=False):

    tokenizer = generator['tokenizer']
    model = generator['model']
    # prepare the model input
    prompt = prompts[0] # do not use more than 1 prompt for hf based generator
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        do_sample=sampling_params['do_sample'],
        temperature=sampling_params['temperature']
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # # parsing thinking content
    # try:
    #     # rindex finding 151668 (</think>)
    #     index = len(output_ids) - output_ids[::-1].index(151668)
    # except ValueError:
    #     index = 0
    
    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return outputs


def main():
    
    model_name_or_path = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    
    generator = load_smolvlm(model_name_or_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device="cuda")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "path": "data/VideoChatGPT/Test_Videos/v__a527xEzhkw.mp4",
                    "fps": 2, "max_frames": 180,
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28

                },
                {"type": "text", "text": "Rate this video from 1 to 10."},
                
            ]
        },
    ]
    params = {"max_new_tokens": 256, "temperature": 0.0}
    
    generated_texts = smolvlm_video_to_text_generation(generator, messages, params)
    # import pdb; pdb.set_trace()
    print(generated_texts[0].split("Assistant:")[1].strip())
    
    # # image to text
    
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "path": "MLLM-Judge/image/0.jpg"},
    #             {"type": "text", "text": "Rate this image from 1 to 10."},
    #         ]
    #     },
    # ]
    
    # generated_texts = smolvlm_image_to_text_generation(generator, messages, params)
    # print(generated_texts[0].split("Assistant:")[1].strip())
    
    
    # # Test openai generator
    # # openai generator takes images and videos in base64 format
    
    # base64Frames = read_video_pyav_base64("LongVideoBench/videos/__Bchxr3ejw.mp4")
    
    # # import pdb; pdb.set_trace()
    # messages = [
    #     {
    #         "role": "system",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": "You are an assistant specialized in evaluating video content."
    #             }
    #         ]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": f"Rate the quality of this video from 1 to 10."
    #             },
    #             *map(lambda x: {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpg;base64,{x}", "detail": "low"}
    #             }, base64Frames)
    #         ]
    #     }
    # ]

    # payload = create_llm_payload(messages=messages, parameters={"temperature": 0.0})
    
    # response = openai_generator(payload=payload)
    
    # print(response)
    
    # # # now make it work for single image
    # image = Image.open("MLLM-Judge/image/0.jpg")
    
    # image_base64 = image_to_base64(image)
    
    # messages = [
    #     {
    #         "role": "system",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": "You are an assistant specialized in evaluating image content."
    #             }
    #         ]
    #     },
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": "Describe this image."
    #             },
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpg;base64,{image_base64}", "detail": "low"}
    #             }
    #         ]
    #     }
    # ]
    # payload = create_llm_payload(messages=messages, parameters={"temperature": 0.0})
    
    # response = openai_generator(payload=payload)
    
    # print(response)
    
    # Messages containing a local video path and a text query

    
    
    # Testing Qwen2.5-VL-3B-Instruct model 
    
    # model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    # # load qwen model
    # generator = load_qwenvlm(model_name_or_path=model_name_or_path, device="cuda")
        
    # params = {"do_sample": False, "max_new_tokens": 256}
    

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "data/VideoChatGPT/Test_Videos/v__a527xEzhkw.mp4",
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]
    
    # # generated_texts = qwenvlm_video_to_text_generation(generator, messages, params)
    # # print(generated_texts)
    
    # # Test Qwen2.5-VL-3B-Instruct model with vllm
    # model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    # generator = load_qwenvlm_vllm(model_name_or_path=model_name_or_path, tensor_parallel_size=2)
    
    # output = qwenvlm_video_to_text_generation_vllm(
    #     generator=generator,
    #     messages=messages,
    #     params={"temperature": 0.0, "max_new_tokens": 256}
    # )
    # print(output)
    
 

# Example usage (if you want to test just this file):
if __name__ == "__main__":
    
    ## To test the openai generator, uncomment the following lines:
    # messages = []
    # messages = format_message(messages, "user", "Describe machine learning in simple terms.")
    # # messages = format_message(messages, "assistant", "The capital of France is Paris.")
    # # messages = format_message(messages, "user", "Why is Paris called the city of love?")
    
    # # The rest of your code remains the same; generator() is still called.
    # response = generator(payload=create_llm_payload(messages=messages))
    # print("Response:", response)
    
    
    ## To test the huggingface smolvlm and qwen models, uncomment the following lines:
    main()
    

