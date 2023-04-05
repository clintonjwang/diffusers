import shutil
import torch, sys, os, argparse
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    #python scripts/inpaint.py -i=0
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int)
    parser.add_argument('-s', '--speed', type=int, default=4)
    parser.add_argument('--pan_direction', default='right')
    parser.add_argument('--overlap_thresh', default=20)
    parser.add_argument('--separation', default=300)
    args = parser.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")

    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
    ).to('cuda')

    prompts = {
        'tree': "an elegant bedroom in a giant tree, with branches and vines covering the walls and ceiling",
        'pond': "an elegant bedroom floating on a giant lily pad, with a small pond, stream and small waterfalls",
        'jungle': "an elegant bedroom with a jungle theme, vines, plants",
        'space': "an elegant bedroom with a futuristic, space-age theme, featuring sleek, angular furniture and high-tech gadgets",
        'fairy': "an elegant magical bedroom for a princess in a fairy tale, with beautiful giant roses, fairy lights",
        'lava': "an elegant bedroom with a volcano theme, bed in the middle, lava, eruption, fire, dramatic",
        'crystal': "an elegant bedroom in a crystal cave, bed in the middle, shiny purple gems, massive stalactites and stalagmites",
        'sunflower': "an elegant bedroom in a field of sunflowers, bed in the middle, beautiful field of yellow, shining sun",
        'sea': "an elegant bedroom underwater, bed in the middle, deep blue sea, coral reef, colorful fish, octupus, seaweed forest",
    }
    ordering = ['crystal', 'sea', 'lava', 'sunflower', 'pond',
            'tree', 'jungle', 'fairy', 'space', 'crystal']

    #p_ix = args.index
    for p_ix in range(9):
        shutil.rmtree(f"./out{p_ix}", ignore_errors=True)
        keys = ordering[p_ix], ordering[p_ix+1]
        root = './keyframes/'

        p_ix = int(p_ix)
        folder = f'out{p_ix}'

        cur_image = pil_to_tensor(Image.open(root+keys[0]+'.png').convert('RGB').resize((512,512), resample=Image.Resampling.LANCZOS))
        keyframe2 = pil_to_tensor(Image.open(root+keys[1]+'.png').convert('RGB').resize((512,512), resample=Image.Resampling.LANCZOS))

        neg_prompt = 'blurry, multiple rooms, artifacts, border'

        joined_image = torch.cat((cur_image,
            cur_image.new_zeros(3, cur_image.size(1), args.separation),
            keyframe2), dim=-1)
        total_width = joined_image.size(-1)

        if args.separation + args.overlap_thresh <= 512:
            dx = (total_width - 512)//2
            crop = joined_image[...,dx:-dx]
            mask = torch.zeros_like(crop)
            mask[...,512-dx:-512+dx] = 255
            prompt = "to the left, " + prompts[keys[0]] + ", smoothly transitioning, to the right, " + prompts[keys[1]] + ", beautiful vivid high resolution"
            with torch.autocast('cuda'):
                output = pipe(prompt=prompt,
                    image=to_pil_image(crop), mask_image=to_pil_image(mask),
                    num_inference_steps=200, 
                    negative_prompt=neg_prompt, guidance_scale=7,
                ).images[0]
            joined_image[...,dx:-dx] = pil_to_tensor(output)
            #alpha = 1
            # alpha blend
        else:
            raise NotImplementedError
            
            cur_left = 0
            cur_right = 512
            k2_left = args.separation + 512
            zeros = torch.zeros_like(cur_image[:,:args.speed])

            while cur_right + args.overlap_thresh <= k2_left:
                # inpaint right of keyframe1
                mask = torch.zeros_like(cur_image)
                mask[:,-args.speed:] = 1
                cur_image = torch.cat((cur_image[:,args.speed:], zeros), dim=1)
                cur_image = pipe(prompt=prompts[keys[0]], image=cur_image, mask_image=mask,
                    num_inference_steps=50, 
                    negative_prompt=neg_prompt, guidance_scale=8,
                ).images[0]

                cur_left += args.speed
                cur_right += args.speed


        # prompt = "to the left, " + left_prompt + ", and to the right, " + right_prompt
        # if cur_right <= k2_left:
        #     # inpaint in between cur_image and keyframe2
        #     cur_image = torch.cat((cur_image[:,args.speed:], zeros), dim=1)
        #     mask = torch.zeros_like(image)
        #     mask[args.speed:] = 1
        #     image = pipe(prompt=prompt, image=cur_image, mask_image=mask,
        #         num_inference_steps=50, 
        #         negative_prompt=neg_prompt, guidance_scale=8,
        #     ).images[0]
        
        cur_left = 0
        cur_right = 512
        k2_left = args.separation + 512
        ix = 0
        os.makedirs(folder, exist_ok=True)
        joined_image = to_pil_image(joined_image)
        for cur_left in range(0, total_width - 512, args.speed):
            joined_image.crop((cur_left,0,cur_left+512,512)).save(f"{folder}/{ix:04d}.png")
            # np.save(f"{folder}/{ix:04d}.png", joined_image[:,cur_left:cur_left+512])
            ix += 1
