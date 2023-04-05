import shutil
import torch, sys, os, argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    #python scripts/inpaint.py -i=0
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-m', '--model_type', default='sd')
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
        
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to('cuda')
    
    pipe.enable_xformers_memory_efficient_attention()
    prompt = "a laptop on a beach, palm trees on a sandy beach, beautiful beach, on the coast, studio ghibli, anime, by hayao miyazaki, anime wallpaper, masterful digital art, trending on artstation, extremely detailed anime wallpaper"
    base_file = {
        0: "hires.png",
        1: "hires.png",
    }
    neg_prompt = "text, watermark, people, signature, poorly drawn, bad art, dithering, texture, photorealistic"
    p_ix = int(args.index)
    folder = f'out{p_ix}'
    os.makedirs(folder, exist_ok=True)
    w,h = 1280*2, 768*2
    init_image = Image.open(base_file[p_ix]).convert('RGB').resize((w,h), Image.Resampling.BICUBIC)
    mask = Image.open('mask.png').resize((w,h), Image.Resampling.BICUBIC)
    for ix in range(20):
        images = pipe(prompt=prompt, image=init_image,
            mask_image=mask, width=w, height=h,
            guidance_scale=5, num_inference_steps=60,
            negative_prompt=neg_prompt,
        ).images
        images[0].save(f"{folder}/{ix:02d}.png")

