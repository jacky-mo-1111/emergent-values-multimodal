import argparse
import sys
import inspect
from typing import Optional

from PIL import Image
import torch

from transformers import AutoTokenizer, AutoModel, AutoProcessor


DEFAULT_MODEL_PATH = \
    "/data/huggingface/models--OpenGVLab--InternVL3-38B-Instruct/snapshots/81c9a040f587e63dc46f128efac04e3f86952847"


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def select_dtype(preferred: str) -> torch.dtype:
    if preferred == "bfloat16":
        return torch.bfloat16
    if preferred == "float16":
        return torch.float16
    return torch.float32


def _normalize_chat_output(result) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    # Common variants: (response, history), {"response": text}, or object with .text
    if isinstance(result, (list, tuple)) and len(result) > 0:
        first = result[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict) and "response" in first:
            return str(first["response"])
    if isinstance(result, dict):
        if "response" in result:
            return str(result["response"])
        if "text" in result:
            return str(result["text"])
    # Last resort: attribute access
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return None


def describe_with_chat(model, tokenizer, prompt: str, image: Image.Image) -> Optional[str]:
    """Use model.chat if available, trying common argument orders used by InternVL."""
    if not hasattr(model, "chat"):
        return None

    # Try (tokenizer, image, prompt)
    try:
        return _normalize_chat_output(model.chat(tokenizer, image, prompt))
    except Exception:
        pass

    # Try (tokenizer, prompt, image)
    try:
        return _normalize_chat_output(model.chat(tokenizer, prompt, image))
    except Exception:
        pass

    # Try with image list
    try:
        return _normalize_chat_output(model.chat(tokenizer, [image], prompt))
    except Exception:
        pass

    try:
        return _normalize_chat_output(model.chat(tokenizer, prompt, [image]))
    except Exception:
        pass

    # Try without tokenizer
    try:
        return _normalize_chat_output(model.chat(image, prompt))
    except Exception:
        pass
    try:
        return _normalize_chat_output(model.chat(prompt, image))
    except Exception:
        pass

    # Try keyword argument variants
    kw_variants = [
        {"tokenizer": tokenizer, "image": image, "question": prompt},
        {"tokenizer": tokenizer, "images": [image], "question": prompt},
        {"images": [image], "question": prompt},
        {"image": image, "question": prompt},
        {"images": [image], "query": prompt},
        {"image": image, "query": prompt},
    ]
    for kwargs in kw_variants:
        try:
            return _normalize_chat_output(model.chat(**kwargs))
        except Exception:
            pass

    return None


@torch.inference_mode()
def describe_with_generate(model, processor, prompt: str, image: Image.Image, max_new_tokens: int,
                           temperature: float, top_p: float) -> str:
    device = list(model.hf_device_map.values())[0] if hasattr(model, "hf_device_map") else ("cuda" if torch.cuda.is_available() else "cpu")

    inputs = processor(images=[image], text=[prompt], return_tensors="pt").to(device)

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "use_cache": True,
    }
    output_ids = model.generate(**inputs, **generate_kwargs)

    # Prefer decode via processor if available, otherwise tokenizer inside processor
    if hasattr(processor, "batch_decode"):
        text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
        text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:
        # Fallback: try model's tokenizer attribute if any
        tok = getattr(model, "tokenizer", None)
        if tok is None or not hasattr(tok, "batch_decode"):
            raise RuntimeError("Unable to decode generated ids; no suitable tokenizer found.")
        text = tok.batch_decode(output_ids, skip_special_tokens=True)[0]

    return text


def describe_with_chat_interleaved(model, tokenizer, prompt: str, images: list[Image.Image]) -> Optional[str]:
    if not hasattr(model, "chat"):
        return None

    # Common InternVL patterns
    try:
        return _normalize_chat_output(model.chat(tokenizer, images, prompt))
    except Exception:
        pass
    try:
        return _normalize_chat_output(model.chat(tokenizer, prompt, images))
    except Exception:
        pass
    try:
        return _normalize_chat_output(model.chat(tokenizer, [images[0], images[1]], prompt))
    except Exception:
        pass
    try:
        return _normalize_chat_output(model.chat(tokenizer, prompt, [images[0], images[1]]))
    except Exception:
        pass
    # Without tokenizer
    try:
        return _normalize_chat_output(model.chat(images, prompt))
    except Exception:
        pass
    try:
        return _normalize_chat_output(model.chat(prompt, images))
    except Exception:
        pass
    # Keyword variants
    kw_variants = [
        {"tokenizer": tokenizer, "images": images, "question": prompt},
        {"images": images, "question": prompt},
        {"tokenizer": tokenizer, "images": images, "query": prompt},
        {"images": images, "query": prompt},
    ]
    for kwargs in kw_variants:
        try:
            return _normalize_chat_output(model.chat(**kwargs))
        except Exception:
            pass
    # Some repos accept a mixed content list
    try:
        content = [
            {"type": "image", "image": images[0]},
            {"type": "text", "text": prompt},
            {"type": "image", "image": images[1]},
        ]
        return _normalize_chat_output(model.chat(tokenizer, content))
    except Exception:
        pass
    return None


@torch.inference_mode()
def describe_with_generate_interleaved(model, processor, prompt: str, images: list[Image.Image],
                                       max_new_tokens: int, temperature: float, top_p: float) -> str:
    device = list(model.hf_device_map.values())[0] if hasattr(model, "hf_device_map") else ("cuda" if torch.cuda.is_available() else "cpu")

    # Try common shapes for multi-image inputs
    last_err: Optional[Exception] = None
    for image_payload in ([images], images):  # [[img1, img2]] then [img1, img2]
        try:
            inputs = processor(images=image_payload, text=[prompt], return_tensors="pt").to(device)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "temperature": temperature,
                "top_p": top_p,
                "use_cache": True,
            }
            output_ids = model.generate(**inputs, **generate_kwargs)
            if hasattr(processor, "batch_decode"):
                return processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
                return processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            tok = getattr(model, "tokenizer", None)
            if tok is None or not hasattr(tok, "batch_decode"):
                raise RuntimeError("Unable to decode generated ids; no suitable tokenizer found.")
            return tok.batch_decode(output_ids, skip_special_tokens=True)[0]
        except Exception as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err
    raise RuntimeError("Failed to prepare multi-image inputs for generation.")


def describe_with_chat_two_turns(model, tokenizer, interleave_prompt: str, images: list[Image.Image]) -> Optional[str]:
    if not hasattr(model, "chat"):
        return None
    # Split prompt into two clauses for img1 then img2
    # Heuristic: split on 'Then ' if present; else use fixed defaults
    p1 = interleave_prompt
    p2 = ""
    marker = "Then "
    if marker in interleave_prompt:
        idx = interleave_prompt.index(marker)
        p1 = interleave_prompt[:idx].strip()
        p2 = interleave_prompt[idx:].strip()
    if not p1:
        p1 = "Please describe this image."
    if not p2:
        p2 = "Then summarize what is going on in another image."

    img1, img2 = images[0], images[1]

    # Try common chat signatures with history
    history = None
    ans1 = None
    # Variants for first turn
    first_variants = [
        ((), {"tokenizer": tokenizer, "image": img1, "question": p1, "history": None}),
        ((), {"tokenizer": tokenizer, "images": [img1], "question": p1, "history": None}),
        ((tokenizer, img1, p1, None), {}),
        ((tokenizer, p1, img1, None), {}),
        ((img1, p1, None), {}),
    ]
    for args, kwargs in first_variants:
        try:
            result = model.chat(*args, **kwargs)
            # Expected to be (response, history) or similar
            if isinstance(result, tuple) and len(result) >= 2:
                ans1 = _normalize_chat_output(result[0])
                history = result[1]
            else:
                ans1 = _normalize_chat_output(result)
                history = None
            if ans1 is not None:
                break
        except Exception:
            continue
    if ans1 is None:
        return None

    # Second turn
    ans2 = None
    second_variants = [
        ((), {"tokenizer": tokenizer, "image": img2, "question": p2, "history": history}),
        ((), {"tokenizer": tokenizer, "images": [img2], "question": p2, "history": history}),
        ((tokenizer, img2, p2, history), {}),
        ((tokenizer, p2, img2, history), {}),
        ((img2, p2, history), {}),
    ]
    for args, kwargs in second_variants:
        try:
            result = model.chat(*args, **kwargs)
            if isinstance(result, tuple) and len(result) >= 2:
                ans2 = _normalize_chat_output(result[0])
            else:
                ans2 = _normalize_chat_output(result)
            if ans2 is not None:
                break
        except Exception:
            continue

    if ans2 is None:
        # Return at least the first answer
        return ans1
    return f"{ans1}\n{ans2}"


def _processor_supports_images(processor) -> bool:
    if processor is None:
        return False
    # Heuristics for multimodal processors
    if hasattr(processor, "image_processor") and processor.image_processor is not None:
        return True
    if hasattr(processor, "vision_processor") and processor.vision_processor is not None:
        return True
    if hasattr(processor, "feature_extractor") and processor.feature_extractor is not None:
        return True
    try:
        sig = inspect.signature(processor.__call__)
        return "images" in sig.parameters
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe two images using OpenGVLab InternVL3-38B-Instruct.")
    parser.add_argument("--image-1", required=True, help="Path to the first image")
    parser.add_argument("--image-2", required=True, help="Path to the second image")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Absolute path to the model snapshot")
    parser.add_argument("--prompt", default="请用简洁中文描述这张图片。", help="Prompt to describe each image")
    parser.add_argument("--interleave", action="store_true", help="Use a single prompt with both images interleaved")
    parser.add_argument(
        "--interleave-prompt",
        default="Please describe this image <img1>. Then summarize what is going on in another image <img2>.",
        help="Prompt used when --interleave is on; refers to both images",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16",
                        help="Model dtype")

    args = parser.parse_args()

    # Device and dtype
    using_cuda = torch.cuda.is_available()
    dtype = select_dtype(args.dtype if using_cuda else "float32")

    # Optional speed tweaks
    if using_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass

    # Load tokenizer/model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        print(f"Failed to load tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if using_cuda else None,
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load processor for generate() fallback path
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = None

    img1 = load_image(args.image_1)
    img2 = load_image(args.image_2)

    if args.interleave:
        # Interleaved single generation
        images = [img1, img2]
        text = args.interleave_prompt
        out = describe_with_chat_interleaved(model, tokenizer, text, images)
        if out is None:
            # Fallback: two-turn conversation within same session
            out = describe_with_chat_two_turns(model, tokenizer, text, images)
        if out is None:
            if not _processor_supports_images(processor):
                print("Model chat() attempts failed and no multimodal processor available; cannot proceed with generate().", file=sys.stderr)
                sys.exit(1)
            out = describe_with_generate_interleaved(model, processor, text, images, args.max_new_tokens, args.temperature, args.top_p)
        print(out)
    else:
        # Separate generations per image
        output1 = describe_with_chat(model, tokenizer, args.prompt, img1)
        output2 = describe_with_chat(model, tokenizer, args.prompt, img2)

        if output1 is None or output2 is None:
            if not _processor_supports_images(processor):
                print("Model chat() attempts failed and no multimodal processor available; cannot proceed with generate().", file=sys.stderr)
                sys.exit(1)
            if output1 is None:
                output1 = describe_with_generate(model, processor, args.prompt, img1, args.max_new_tokens, args.temperature, args.top_p)
            if output2 is None:
                output2 = describe_with_generate(model, processor, args.prompt, img2, args.max_new_tokens, args.temperature, args.top_p)

        print(f"Image 1: {output1}")
        print(f"Image 2: {output2}")


if __name__ == "__main__":
    main()


