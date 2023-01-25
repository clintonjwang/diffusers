"""
# Image Mixer
_Created by [Justin Pinkney](https://www.justinpinkney.com) at [Lambda Labs](https://lambdalabs.com/)_
To skip the queue you can <a href="https://huggingface.co/spaces/lambdalabs/image-mixer-demo?duplicate=true" style="display:inline-block;position: relative;"><img style="margin-top: 0;margin-bottom: 0;margin-left: .25em;" src="https://bit.ly/3gLdBN6"></a>
### __Provide one or more images to be mixed together by a fine-tuned Stable Diffusion model (see tips and advice below👇).__
![banner-large.jpeg](https://s3.amazonaws.com/moonup/production/uploads/1674039767068-62bd5f951e22ec84279820e8.jpeg)
## Tips
- You can provide between 1 and 5 inputs, these can either be an uploaded image a text prompt or a url to an image file.
- The order of the inputs shouldn't matter, any images will be centre cropped before use.
- Each input has an individual strength parameter which controls how big an influence it has on the output.
- The model was not trained using text and can not interpret complex text prompts.
- Using only text prompts doesn't work well, make sure there is at least one image or URL to an image.
- The parameters on the bottom row such as cfg scale do the same as for a normal Stable Diffusion model.
- Balancing the different inputs requires tweaking of the strengths, I suggest getting the right balance for a small number of samples and with few steps until you're
happy with the result then increase the steps for better quality.
- Outputs are 640x640 by default.
## How does this work?
This model is based on the [Stable Diffusion Image Variations model](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)
but it has been fined tuned to take multiple CLIP image embeddings. During training, up to 5 random crops were taken from the training images and
the CLIP image embeddings were computed, these were then concatenated and used as the conditioning for the model. At inference time we can combine the image
embeddings from multiple images to mix their concepts (and we can also use the text encoder to add text concepts too).
The model was trained on a subset of LAION Improved Aesthetics at a resolution of 640x640 and was trained using 8xA100 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
"""

from io import BytesIO
import pdb
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
import requests
import functools

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.extras import load_model_from_config, load_training_dir
import clip

from PIL import Image

from huggingface_hub import hf_hub_download

def mixer(img_paths, prompts,
    strengths=None,
    guidance_scale=2,
    n_samples=1, seed=None, steps=150):

    ckpt = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-pruned.ckpt")
    config = hf_hub_download(repo_id="lambdalabs/image-mixer", filename="image-mixer-config.yaml")

    device = "cuda:0"
    model = load_model_from_config(config, ckpt, device=device, verbose=False)
    model = model.to(device).half()

    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    n_inputs = 5

    torch.cuda.empty_cache()

    @torch.no_grad()
    def get_im_c(im_path, clip_model):
        im = Image.open(im_path).convert("RGB")
        prompts = preprocess(im).to(device).unsqueeze(0)
        return clip_model.encode_image(prompts).float()

    @torch.no_grad()
    def get_txt_c(txt, clip_model):
        text = clip.tokenize([txt,]).to(device)
        return clip_model.encode_text(text)


    def to_im_list(x_samples_ddim):
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        ims = []
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            ims.append(Image.fromarray(x_sample.astype(np.uint8)))
        return ims

    @torch.no_grad()
    def sample(sampler, model, c, uc, scale, start_code, h=512, w=512, precision="autocast",ddim_steps=50):
        ddim_eta=0.0
        precision_scope = autocast if precision=="autocast" else nullcontext
        with precision_scope("cuda"):
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=c.shape[0],
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=start_code)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
        return to_im_list(x_samples_ddim)

    
    h = w = 640

    sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)
    N = len(img_paths)
    if strengths is None:
        strengths = [1/N] * N
    if isinstance(prompts, str):
        prompts = [prompts] * N

    if seed is not None:
        torch.manual_seed(seed)
    start_code = torch.randn(n_samples, 4, h//8, w//8, device=device)
    conds = []

    for i in range(2):
        conds.append(strengths[i] * get_im_c(img_paths[i], clip_model))
        conds.append(strengths[i] * get_txt_c(prompts[i], clip_model))
    # this_cond = torch.zeros((1, 768), device=device)
    conds = torch.cat(conds, dim=0).unsqueeze(0)
    conds = conds.tile(n_samples, 1, 1)

    ims = sample(sampler, model, conds, 0*conds, guidance_scale, start_code, ddim_steps=steps)
    return ims
    
if __name__ == '__main__':
    # img_paths = ('02.png', '03.png')
    # prompt = 'close-up pokemon on white background, pokemon, by Ken Sugimori, venusaur, ivysaur, grass-type, smooth, official splash art'
    # for ix in range(4):
    #     imgs = mixer(img_paths, prompt)
    #     imgs[0].save(f'23_{ix}.png')
    img_paths = ('01.png', '02.png')
    prompt = 'pokemon on white background, pokemon, by Ken Sugimori, bulbasaur, ivysaur, grass-type'
    for ix in range(4):
        imgs = mixer(img_paths, prompt)
        imgs[0].save(f'12_{ix}.png')