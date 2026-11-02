#!/usr/bin/env python3
"""Generate a test input image with the local Qwen-Image 2.1 install: tools/gen_test_image.py name "prompt"."""
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import Graph, run, COMFY_OUT

name, prompt = sys.argv[1], sys.argv[2]
g = Graph()
unet = g.add("UNETLoader", unet_name="qwen_image_2.1_int8_convrot.safetensors", weight_dtype="default")
model = g.add("QwenImage21Cache", model=unet[0], device="auto", dtype="default")
clip = g.add("CLIPLoader", clip_name="qwen3vl_8b_int8_convrot.safetensors", type="qwen_image", device="default")
vae = g.add("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors")
enc = g.add("TextEncodeQwenImage21", clip=clip[0], prompt=prompt, negative_prompt="", resolution=1024, images={}, vae=vae[0])
lat = g.add("EmptyLatentImage", width=1024, height=1024, batch_size=1)
ks = g.add("KSampler", model=model[0], seed=int(sys.argv[3]) if len(sys.argv) > 3 else 7, steps=25, cfg=1.0,
           sampler_name="euler", scheduler="simple", positive=enc[0], negative=enc[1], latent_image=lat[0], denoise=1.0)
dec = g.add("VAEDecode", samples=ks[0], vae=vae[0])
g.add("SaveImage", images=dec[0], filename_prefix=f"trellis_inputs/{name}")
hist, dt = run(g.nodes)
for out in hist["outputs"].values():
    for im in out.get("images", []):
        src = COMFY_OUT / im["subfolder"] / im["filename"]
        dst = Path(__file__).resolve().parent.parent / "test_inputs" / f"{name}.png"
        shutil.copy2(src, dst)
        print(dst, f"{dt:.0f}s")
