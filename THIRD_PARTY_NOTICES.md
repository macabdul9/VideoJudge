# Third-party attribution

The fine-tuning implementation in `training/qwenvl/` comes from the
[Qwen2.5-VL training framework](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)
used in our experiments. The dataset configuration module is adapted to load a
local JSON registry instead of machine-specific paths. Original copyright headers
are retained, including notices for FastChat and Stanford Alpaca where present.
The Qwen Apache-2.0 license is included in `training/LICENSE`.

Video processing and inference use Qwen VL utilities, Transformers, PyTorch, and
optionally vLLM. Their licenses apply independently. Qwen model checkpoint terms
also apply to fine-tuned checkpoints.

Training and meta-evaluation datasets include or derive from VideoInstruct-100K,
VCG-plus, VideoChat2-IT, LLaVA-Video, VideoChatGPT, VATEX-EVAL, LongVideoBench, and
VideoAutoArena. Consult their source repositories and dataset cards for usage and
redistribution terms. Inclusion in an experiment does not change a source license.
