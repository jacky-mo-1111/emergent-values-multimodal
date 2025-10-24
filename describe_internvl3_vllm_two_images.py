import argparse
import base64
import mimetypes
import os
import sys
from typing import List

from vllm import LLM, SamplingParams


DEFAULT_MODEL_PATH = \
    "/data/huggingface/models--OpenGVLab--InternVL3-38B-Instruct/snapshots/81c9a040f587e63dc46f128efac04e3f86952847"


def to_data_url(path: str) -> str:
    abs_path = os.path.abspath(path)
    mime, _ = mimetypes.guess_type(abs_path)
    if not mime:
        mime = "image/jpeg"
    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(image_paths: List[str]):
    # Use data URLs to avoid local file permission flags
    content = [
        {"type": "text", "text": "Which image do you prefer looking at? Option A:"},
        {"type": "image_url", "image_url": {"url": to_data_url(image_paths[0])}},
        {"type": "text", "text": "Option B:"},
        {"type": "image_url", "image_url": {"url": to_data_url(image_paths[1])}},
        {"type": "text", "text": "Please respond with only \"A\" or \"B\"."},
    ]
    return [{"role": "user", "content": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM runner for InternVL3 two-image comparison.")
    parser.add_argument("--image-1", required=True, help="Path to the first image (Option A)")
    parser.add_argument("--image-2", required=True, help="Path to the second image (Option B)")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Absolute path to the model snapshot")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size for vLLM")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")

    args = parser.parse_args()

    # Build messages with file:// URLs
    messages = build_messages([args.image_1, args.image_2])

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    try:
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    except Exception as e:
        print(f"Failed to initialize vLLM LLM: {e}", file=sys.stderr)
        sys.exit(1)

    # Prefer the chat API if available (vLLM >= 0.10)
    if hasattr(llm, "chat"):
        outputs = llm.chat([messages], sampling_params=sampling)
    else:
        # Fallback: try converting messages via tokenizer chat template (text-only path)
        try:
            tokenizer = llm.get_tokenizer()
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"Chat API unavailable and failed to build prompt from messages: {e}", file=sys.stderr)
            sys.exit(1)
        outputs = llm.generate([prompt_text], sampling_params=sampling)

    if not outputs or not outputs[0].outputs:
        print("No output generated.")
        return

    text = outputs[0].outputs[0].text.strip()
    print(text)


if __name__ == "__main__":
    main()


