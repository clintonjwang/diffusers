import shutil
import torch, sys, os, argparse
from diffusers import DiffusionPipeline
torch.backends.cudnn.benchmark = True

from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-n', '--num_steps', type=int, default=120)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    
    seg_feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-large-ade")
    segmenter = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade")

    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        custom_pipeline="seg_interpolate_sd",
    ).to("cuda")
    pipe.init_segmenter(
        seg_feature_extractor=seg_feature_extractor,
        segmenter=segmenter)
    pipe.enable_attention_slicing()
    
    prompts = {
        'tree': "a bedroom designed to look like it's inside a giant tree, with branches and vines covering the walls and ceiling",
        'pond': "a bedroom with the bed in the middle floating on a giant lily pad, with a small pond, stream and small waterfalls",
        'jungle': "a bedroom with a jungle theme, complete with vines, plants, and even a small zip line",
        'space': "a bedroom with a futuristic, space-age theme, featuring sleek, angular furniture and high-tech gadgets",
        'fairy': "an elegant magical bedroom for a princess in a fairy tale, with beautiful giant roses, fairy lights 8k photography",
        'lava': "a bedroom with a volcano theme, bed in the middle, lava, eruption, fire, dramatic vivid 8k",
        'crystal': "a bedroom in a crystal cave, bed in the middle, shiny purple gems, massive stalactites and stalagmites, beautiful vivid 8k",
        'sunflower': "a bedroom in a field of sunflowers, bed in the middle, beautiful field of yellow, shining sun, vivid 8k",
        'sea': "a bedroom underwater, bed in the middle, deep blue sea, coral reef, colorful fish, octupus, seaweed forest, vivid 8k",
    }
    seeds = {
        'tree': 23,
        'pond': 24,
        'jungle': 25,
        'space': 6,
        'fairy': 10,
        'lava': 24,
        'crystal': 8,
        'sunflower': 0,
        'sea': 11,
    }

    ordering = ['crystal', 'sea', 'lava', 'sunflower',
        'pond', 'tree', 'jungle', 'fairy', 'space', 'crystal']
    shutil.rmtree(f"./out{args.index}", ignore_errors=True)
    keys = ordering[args.index], ordering[args.index+1]
    pipe.walk(
        prompts=[prompts[keys[0]], prompts[keys[1]]],
        seeds=[seeds[keys[0]], seeds[keys[1]]],
        num_interpolation_steps=args.num_steps,
        output_dir=f"./out{args.index}",
        batch_size=4,
        height=768,
        width=768,
        guidance_scale=8.5,
        seg_consistency=1,
        num_inference_steps=50,
    )
