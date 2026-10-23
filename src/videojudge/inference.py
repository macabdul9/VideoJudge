from __future__ import annotations

from typing import Any


class QwenVideoJudge:
    def __init__(
        self,
        model: str,
        device: str = "auto",
        backend: str = "hf",
        tensor_parallel_size: int = 1,
        attention: str = "flash_attention_2",
    ) -> None:
        from transformers import AutoProcessor

        self.backend = backend
        self.processor = AutoProcessor.from_pretrained(model)
        if backend == "vllm":
            from vllm import LLM

            self.model = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
        else:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map=device,
                attn_implementation=attention,
            )
            self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if self.backend == "vllm":
            from vllm import SamplingParams

            media = {}
            if images is not None:
                media["image"] = images
            if videos is not None:
                media["video"] = videos
            output = self.model.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": media,
                        "mm_processor_kwargs": video_kwargs,
                    }
                ],
                sampling_params=SamplingParams(
                    temperature=temperature if do_sample else 0.0, max_tokens=max_new_tokens
                ),
            )
            return output[0].outputs[0].text.strip()
        import torch

        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)
        parameters = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
        if do_sample:
            parameters["temperature"] = temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **parameters)
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated, strict=True)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
