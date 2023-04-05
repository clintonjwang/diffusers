from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from empatches import EMPatches
import matplotlib.pyplot as plt
import numpy as np

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipeline.to("cuda")
img = Image.open('07.png')
patches = []
for i in range(768//64-1):
    for j in range(768//64-1):
        patch = img.crop((64*i, 64*j, 64*(i+2), 64*(j+2)))
        prompt='marble sculpture of Billy Mays triumphantly advertising extra long spaghetti, Michelangelo, dramatic lighting shadows, 8k highly detailed Renaissance masterpiece'
        super_img = pipe(prompt=prompt, image=patch, num_inference_steps=100, eta=1, guidance_scale=2)
        patches.append(super_img['images'][0])

emp = EMPatches()
_, indices = emp.extract_patches(np.array(img), patchsize=128, overlap=0.5)
indices = list(map(lambda x: (x[0]*4, x[1]*4, x[2]*4, x[3]*4), indices))
patches_np = list(map(np.array, patches))
merged_img = emp.merge_patches(patches_np, indices, mode='avg')
plt.imsave('merged_img.png', merged_img.astype('uint8'))