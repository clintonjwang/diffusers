import torch, sys, os, argparse
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from diffusers import StableDiffusionPipeline
from diffusers.evolving_prompt import EvolvingStableDiffusionPipeline

from diffusers import DPMSolverMultistepScheduler
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index')
    parser.add_argument('-m', '--model_type', default='sd')
    # parser.add_argument('-c', '--use_clip', store_true)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    
    if args.model_type == 'pgen':
        pipe = EvolvingStableDiffusionPipeline.from_pretrained(
            # 'darkstorm2150/Protogen_Infinity_Official_Release',
            'darkstorm2150/Protogen_x5.8_Official_Release', #Scifi+Anime
            # 'darkstorm2150/Protogen_x5.3_Official_Release', #Photorealism
            torch_dtype=torch.float16, safety_checker=None,
        ).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )

    elif args.model_type == 'sd':
        model_id = "stabilityai/stable-diffusion-2-1"
        pipe = EvolvingStableDiffusionPipeline.from_pretrained(model_id,
            safety_checker=None,
            torch_dtype=torch.float16).to("cuda")

    elif args.model_type == 'db':
        pipe = EvolvingStableDiffusionPipeline.from_pretrained(
            "./dreambooth", torch_dtype=torch.float16,
            safety_checker=None).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()
    
    #             neg_prompt = """cropped, low quality, 
    # low resolution, out of frame, jpg, off-center"""
    # neg_prompt = "low quality, ugly, blurry, out of focus, off-center, lopsided, logo, signature, text, badly drawn, poorly drawn, deformed, defective, incoherent, twisted, extra limbs, extra fingers"
    #, signature, text, cluttered, busy
    # neg_prompt = "low quality, signature, text, 3d render, b&w, bad anatomy, bad anatomy, bad anatomy, bad art, bad art, bad proportions, blurry, blurry, blurry, body out of frame, cross-eye, deformed, deformed, deformed, disfigured, disfigured, disfigured, duplicate, extra arms, extra fingers, extra legs, extra legs, extra limbs, extra limbs, extra limbs, extra limbs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, morbid, mutated, mutated hands, mutated hands, mutation, mutation, mutilated, out of frame, out of frame, Photoshop, poorly drawn face, poorly drawn face, poorly drawn feet, poorly drawn hands, poorly drawn hands, tiling, too many fingers, weird colors"
    # neg_prompt = "low quality, ugly, blurry, off-center, signature, text, watermark, complex"
    neg_prompt = "low quality, ugly, blurry, off-center"
    # neg_prompt = "low quality, ugly, weird, badly drawn, amateur, complicated"
    # """badly drawn, poorly drawn, messy, wip, unfinished, 
    # bad, mediocre, average, dull, boring, text, letters, writing, 
    # quote, signature, screencap, sepia, faded, colored sclera"""
    #         """black and white, blur, blurry, soft, blush, filter, noise, deformed, defective, 
    # incoherent, twisted, extra limbs, extra fingers, poorly drawn hands, messy drawing"""
#             neg_prompt = """bad cropping, low quality, 
# low resolution, error, bad anatomy, missing bodyparts, 
# extra bodyparts, out of frame, compression"""

# "modelshoot style, extremely detailed 8k wallpaper, A detailed side portrait of a woman using a smartphone illustrator, hyperrealistic, digital art, realistic painting, dnd, character design, trending on artstation"
# modelshoot style, (extremely detailed 8k wallpaper), block paint depicting a character in a cyberpunk street, posed character design study, backlit, light rays, highly detailed, trending on artstation
# "dramatic lighting art by brandon anschultz by yoji shinkawa by richard schmid by greg rutkowski by sandra chevrier by jeremy lipking cinematic dramatic brush strokes background, dramatic"
# "modelshoot style, (extremely detailed CG unity 8k wallpaper), full shot body photo of the most beautiful artwork in the world, emma stone in a dress, professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, photorealistic painting art by midjourney and greg rutkowski"

# "3d, 3d render, b&w, bad anatomy, bad anatomy, bad anatomy, bad art, bad art, bad proportions, blurry, blurry, blurry, body out of frame, canvas frame, cartoon, cloned face, close up, cross-eye, deformed, deformed, deformed, disfigured, disfigured, disfigured, duplicate, extra arms, extra arms, extra fingers, extra legs, extra legs, extra limbs, extra limbs, extra limbs, extra limbs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, morbid, mutated, mutated hands, mutated hands, mutation, mutation, mutilated, out of frame, out of frame, out of frame, Photoshop, poorly drawn face, poorly drawn face, poorly drawn feet, poorly drawn hands, poorly drawn hands, tiling, too many fingers, video game, weird colors"
# canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render


    prompt = {
    # 0: "a laughing toddler playing with a large kindly robot, photography, ultra-detailed, hyperrealistic, unreal engine, highly detailed, wide-angle lens, modelshoot style",
    # 1: {
    #     0: "(╯°□°)╯︵ ┻━┻, a line drawing, cartoon, single line, Disney, simple, elegant",
    # },
    2: {
        0: "Wes anderson",
        10: "Wes anderson",
    },
    1: "a photo of a cat, photography, ultra-detailed, hyperrealistic, highly detailed, modelshoot style",
    # 2: "huge origami paper castle under siege, 3D, intricate, extremely detailed, hyper-realistic, colorful, sharp, unreal engine, photography, wide-angle lens",
    # 3: {
    #     0: "miniature origami paper soldiers, 3D, intricate, extremely detailed, hyper-realistic, colorful, sharp, unreal engine, close-up photography, wide-angle lens",
    # },
    # 3: {
    #     0: "eldritch horror, hyperrealistic monster, dark, masterpiece, Lovecraftian abomination, intricate, depths of hell, ultra-detailed",
    #     # 70: "blue screen of death as the backdrop",
    #     # 100: "eldritch horror, hyperrealistic monster, masterpiece, Lovecraftian abomination, intricate, ultra-detailed",
    # },
    # 4: {
    #     0: "army of securitrons from fallout new vegas, robots, heavily armed,",
    #     # 70: "blue screen of death as the backdrop",
    #     # 100: "eldritch horror, hyperrealistic monster, masterpiece, Lovecraftian abomination, intricate, ultra-detailed",
    # },
    }
    p_ix = int(args.index)
    folder = f'out{p_ix}'
    os.makedirs(folder, exist_ok=True)
    open(folder+'/prompt.txt', 'w').write(str(prompt[p_ix]))
    for ix in range(20):
        image = pipe(prompt[p_ix],
            num_inference_steps=80,
            negative_prompt=neg_prompt,
            guidance_scale=8).images[0]
    
        image.save(f"{folder}/{ix:02d}.png")