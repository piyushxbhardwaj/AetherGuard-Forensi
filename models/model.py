from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import os

def load_hf_model(model_name="dima806/deepfake_vs_real_image_detection", device='cpu'):
    cache_dir = os.path.abspath("./models/cache")
    os.makedirs(cache_dir, exist_ok=True)

    kwargs = {"cache_dir": cache_dir}

    print("Loading model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained(model_name, **kwargs)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        **kwargs
    )

    model.to(device)
    model.eval()
    return model, processor

def load_model(weights_path=None, device='cpu'):
    return load_hf_model(device=device)
