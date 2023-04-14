import pdb
import shutil
import torch, sys, os, argparse
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from sds import SDSStableDiffusionPipeline
from diffusers import HeunDiscreteScheduler, EulerDiscreteScheduler

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default='0')
    parser.add_argument('-s', '--seed', default=0)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    
    pipe = SDSStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        safety_checker=None, torch_dtype=torch.float16).to("cuda")
    # pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    neg_prompt = "low quality, ugly, blurry"
    prompt = {
        0: "photo of a cute dog playing on a lawn",
        # 0: "Escher masterpiece, optical illusion, abstract art, surrealism, impossible",
        1: "beautiful line drawing, abstract art, elegant, simple, minimalist masterpiece",
        2: "minimal flat 2d vector icon. lineal color. on a white background. trending on artstation",
    }
    # torch.manual_seed(args.seed)
    p_ix = int(args.index)
    folder = f'out{p_ix}'
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)
    open(folder+'/prompt.txt', 'w').write(str(prompt[p_ix]))
    for ix in range(1):
        kwargs = dict(prompt=prompt[p_ix], num_inference_steps=50, num_sds_steps=30,
                      guidance_scale=7.5, negative_prompt=neg_prompt, lr=1e-3)
        img = pipe.sds(**kwargs).images
        img[0].save(f"{folder}/0_sds.png")
        # img = pipe(**kwargs).images
        # img[0].save(f"{folder}/0_standard.png")
