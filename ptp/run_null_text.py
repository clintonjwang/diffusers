
import pdb
from null_text import *
from PIL import Image

def visualize_attention(image_path, prompt):
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(
        image_path, prompt, offsets=(0,0,200,0), verbose=True)

    prompts = [prompt]
    controller = AttentionStore()
    image_inv, x_t = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)
    print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
    ptp_utils.view_images([image_gt, image_enc, image_inv[0]])
    show_cross_attention(controller, 16, ["up", "down"])


def modify_img(img_path, old_prompt, new_prompt, strength):
    # self_replace_steps = strength
    _, x_t, uncond_embeddings = null_inversion.invert(
        img_path, old_prompt, offsets=(0,0,0,0), verbose=True)

    prompts = [old_prompt, new_prompt]
    cross_replace_steps = {'default_': .8,}
    blend_word = ((('cat',), ("tiger",))) # for local edit
    # blend_word = ((('cat',), ("cat",))) # for local edit
    # blend_word = None # global edit
    eq_params = {"words": ("tiger",), "values": (2,)} # amplify attention to the word "tiger" by *2 
    # eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
    # eq_params = {"words": ("watercolor",  ), "values": (5, 2,)}  # amplify attention to the word "watercolor" by 5

    controller = make_controller(prompts, True, cross_replace_steps, strength, blend_word, eq_params)
    images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)
    pdb.set_trace()