import torch, sys, os, argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler

def ptp():
    from ptp.null_text import null_inversion, make_controller, run_and_display

    img_path='final.png'
    old_prompt='a 3D poorly made render of bugs bunny holding dual light sabers'
    _, x_t, uncond_embeddings = null_inversion.invert(
        img_path, old_prompt, offsets=(0,0,0,0), verbose=True)
    new_prompt = 'a 2D Marvel highres cartoon of bugs bunny holding dual light sabers'#, highres 8k wallpaper, highquality, disney'
    strength = 0.9 # higher preserves more

    prompts = [old_prompt, new_prompt]
    cross_replace_steps = {'default_': .5,} #.8 # seems to change more when higher
    # blend_word = None # global edit
    blend_word = None#((('render',), ("cartoon",))) # for local edit
    eq_params = {"words": ("cartoon","2D","highres", "Marvel"), "values": (2,2,2,4)} # amplify attention
    # blend_word = ((('cat',), ("tiger",))) # for local edit
    # eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
    # eq_params = {"words": ("cartoon",  ), "values": (5, 1,)}  # amplify attention to the word "watercolor" by 5

    controller = make_controller(prompts, True, cross_replace_steps, strength, blend_word, eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)
    Image.fromarray(images[1]).save('fi2.png')
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index')
    parser.add_argument('-m', '--model_type', default='sd')
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
        
    if args.model_type == 'pgen':
        pipe = StableDiffusionPipeline.from_pretrained(
            'darkstorm2150/Protogen_x5.3_Official_Release', #Photorealism
            torch_dtype=torch.float16, safety_checker=None,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = StableDiffusionImg2ImgPipeline(**pipe.components).to("cuda")
    
    elif args.model_type == 'sd':
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            # "runwayml/stable-diffusion-v1-5", 
            "stabilityai/stable-diffusion-2-1",
            safety_checker=None,
            torch_dtype=torch.float16).to('cuda')

    elif args.model_type == 'db':
        pipe = StableDiffusionPipeline.from_pretrained(
            "./dreambooth", torch_dtype=torch.float16,
            safety_checker=None).to("cuda")
        pipe = StableDiffusionImg2ImgPipeline(**pipe.components).to("cuda")

    elif args.model_type == 'ptp':
        ptp()
        exit()

    pipe.enable_xformers_memory_efficient_attention()
    # prompt = "palm trees on a sandy beach, beautiful beach, on the coast, studio ghibli, anime, by hayao miyazaki, anime wallpaper"
    # prompt = "beautiful anime wallpaper, magical night, starry sky, northern lights above a forest at night, studio ghibli, anime, by hayao miyazaki, romantic anime scene, anime wallpaper, highly detailed, dramatic colors, aurora borealis"
    # prompt = "a close-up of grass with water droplets, low angle, bright rays of light, wide angle, gardening, springtime morning, studio ghibli, anime, by hayao miyazaki, anime wallpaper, highly detailed"
    # prompt = "beautiful anime wallpaper, a view of the earth from space, european union, stars shining, studio ghibli, anime, magical night, by hayao miyazaki, anime wallpaper, highly detailed"
    # prompt = "the golden gate bridge, san francisco, beautiful anime wallpaper, studio ghibli, anime, by hayao miyazaki, anime wallpaper, highly detailed"
    prompt = 'people on a rocky landscape, HDR photo, photorealistic, ultra hd, 30mm lens, wide angle'
    base_file = {
        0: "00.png",
        # 0: "aurora.png",
        # 1: "beach.png",
        # 2: "grass.png",
        # 3: "earth.png",
        # 4: "sf.png",
    }
    p_ix = int(args.index)
    folder = f'out{p_ix}'
    os.makedirs(folder, exist_ok=True)
    init_image = Image.open(base_file[p_ix]).convert('RGB').resize((768, 768), Image.Resampling.BICUBIC)
    for ix in range(20):
        images = pipe(prompt=prompt, image=init_image,
            strength=0.4, guidance_scale=10, num_inference_steps=100).images
        # lower strength preserves more of the original image
        images[0].save(f"{folder}/{ix:02d}.png")
