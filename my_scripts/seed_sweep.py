import shutil
import torch, sys, os, argparse
from diffusers import DiffusionPipeline
torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-s', '--start_seed', type=int, default=0)
    parser.add_argument('-n', '--num_seeds', type=int, default=30)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        custom_pipeline="interpolate_stable_diffusion",
    ).to("cuda")
    pipe.enable_attention_slicing()

    prompt = {
        0: "a bedroom designed to look like it's inside a giant tree, with branches and vines covering the walls and ceiling",
        1: "a bedroom with the bed in the middle floating on a giant lily pad, with a small pond, stream and small waterfalls",
        2: "a bedroom with a jungle theme, complete with vines, plants, and even a small zip line",
        3: "a bedroom with a futuristic, space-age theme, featuring sleek, angular furniture and high-tech gadgets",
        4: "an elegant magical bedroom for a princess in a fairy tale, with beautiful giant roses, fairy lights 8k photography",
        5: "a bedroom inside a volcano, bed in the middle, lava, eruption, fire, dramatic vivid 8k",
        6: "a bedroom in a crystal cave, bed in the middle, shiny purple gems, massive stalactites and stalagmites, beautiful vivid 8k",
        7: "a bedroom in a field of sunflowers, bed in the middle, beautiful field of yellow, shining sun, vivid 8k",
        8: "a bedroom underwater, bed in the middle, deep blue sea, coral reef, colorful fish, octupus, seaweed forest, vivid 8k",
    }
    shutil.rmtree(f"./out{args.index}", ignore_errors=True)
    pipe.walk(
        prompts=[prompt[args.index]] * args.num_seeds,
        seeds=list(range(args.start_seed, args.start_seed+args.num_seeds)),
        num_interpolation_steps=1,
        output_dir=f"./out{args.index}",
        batch_size=1,
        height=768,
        width=768,
        guidance_scale=8.5,
        num_inference_steps=50,
    )
