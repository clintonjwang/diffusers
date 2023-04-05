import pdb
import torch, sys, os, argparse
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from ckconv.ckc_sd import CKCStableDiffusionPipeline

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index')
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    
    pipe = CKCStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        safety_checker=None,
        torch_dtype=torch.float16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    
    neg_prompt = "low quality, ugly, blurry, lopsided"
    prompt = {
        0: "Escher masterpiece, optical illusion, abstract art, surrealism, impossible",
        1: "beautiful line drawing, abstract art, elegant, simple, minimalist masterpiece",
        2: "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation",
    }
    p_ix = int(args.index)
    folder = f'out{p_ix}'
    os.makedirs(folder, exist_ok=True)
    open(folder+'/prompt.txt', 'w').write(str(prompt[p_ix]))
    for ix in range(1):
        img1, img2, img3 = pipe(prompt[p_ix],
            num_inference_steps=50,
            negative_prompt=neg_prompt,
            guidance_scale=8)
    
        img1[0].save(f"{folder}/{ix:02d}.png")
        img2[0].save(f"{folder}/{ix:02d}_.png")
        img3[0].save(f"{folder}/{ix:02d}_cc.png")