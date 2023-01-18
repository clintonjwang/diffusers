# bk2_blending with CFG

import inspect
import pdb, os
import shutil
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import PIL
from PIL import Image
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@torch.no_grad()
def interpolate_spherical(p0, p1, fract_mixing: float):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    See https://en.wikipedia.org/wiki/Slerp
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """ 
    
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    
    p0 = p0.double()
    p1 = p1.double()
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
        
    return interp

def interpolate_linear(p0, p1, fract_mixing):
    r"""
    Helper function to mix two variables using standard linear interpolation.
    Args:
        p0: 
            First tensor / np.ndarray for interpolation
        p1: 
            Second tensor / np.ndarray  for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a linear mix between both.
    """ 
    reconvert_uint8 = False
    if type(p0) is np.ndarray and p0.dtype == 'uint8':
        reconvert_uint8 = True
        p0 = p0.astype(np.float64)
        
    if type(p1) is np.ndarray and p1.dtype == 'uint8':
        reconvert_uint8 = True
        p1 = p1.astype(np.float64)
    
    interp = (1-fract_mixing) * p0 + fract_mixing * p1
    
    if reconvert_uint8:
        interp = np.clip(interp, 0, 255).astype(np.uint8)
        
    return interp


def add_frames_linear_interp(
        list_imgs: List[np.ndarray], 
        fps_target: Union[float, int] = None, 
        duration_target: Union[float, int] = None,
        nmb_frames_target: int=None,
    ):
    r"""
    Helper function to cheaply increase the number of frames given a list of images, 
    by virtue of standard linear interpolation.
    The number of inserted frames will be automatically adjusted so that the total of number
    of frames can be fixed precisely, using a random shuffling technique.
    The function allows 1:1 comparisons between transitions as videos.
    
    Args:
        list_imgs: List[np.ndarray)
            List of images, between each image new frames will be inserted via linear interpolation.
        fps_target: 
            OptionA: specify here the desired frames per second.
        duration_target: 
            OptionA: specify here the desired duration of the transition in seconds.
        nmb_frames_target: 
            OptionB: directly fix the total number of frames of the output.
    """ 
    
    # Sanity
    if nmb_frames_target is not None and fps_target is not None:
        raise ValueError("You cannot specify both fps_target and nmb_frames_target")
    if fps_target is None:
        assert nmb_frames_target is not None, "Either specify nmb_frames_target or nmb_frames_target"
    if nmb_frames_target is None:
        assert fps_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        assert duration_target is not None, "Either specify duration_target and fps_target OR nmb_frames_target"
        nmb_frames_target = fps_target*duration_target
    
    # Get number of frames that are missing
    nmb_frames_diff = len(list_imgs)-1
    nmb_frames_missing = nmb_frames_target - nmb_frames_diff - 1
    
    if nmb_frames_missing < 1:
        return list_imgs
    
    list_imgs_float = [img.astype(np.float32) for img in list_imgs]
    
    # Distribute missing frames, append nmb_frames_to_insert(i) frames for each frame
    mean_nmb_frames_insert = nmb_frames_missing/nmb_frames_diff
    constfact = np.floor(mean_nmb_frames_insert)
    remainder_x = 1-(mean_nmb_frames_insert - constfact)
    
    nmb_iter = 0
    while True:
        nmb_frames_to_insert = np.random.rand(nmb_frames_diff)
        nmb_frames_to_insert[nmb_frames_to_insert<=remainder_x] = 0
        nmb_frames_to_insert[nmb_frames_to_insert>remainder_x] = 1
        nmb_frames_to_insert += constfact
        if np.sum(nmb_frames_to_insert) == nmb_frames_missing:
            break
        nmb_iter += 1
        if nmb_iter > 100000:
            print("issue with inserting the right number of frames")
            break
        
    nmb_frames_to_insert = nmb_frames_to_insert.astype(np.int32)
    list_imgs_interp = []
    for i in range(len(list_imgs_float)-1):#, desc="STAGE linear interp"):
        img0 = list_imgs_float[i]
        img1 = list_imgs_float[i+1]
        list_imgs_interp.append(img0.astype(np.uint8))
        list_fracts_linblend = np.linspace(0, 1, nmb_frames_to_insert[i]+2)[1:-1]
        for fract_linblend in list_fracts_linblend:
            img_blend = interpolate_linear(img0, img1, fract_linblend).astype(np.uint8)
            list_imgs_interp.append(img_blend.astype(np.uint8))
        
        if i==len(list_imgs_float)-2:
            list_imgs_interp.append(img1.astype(np.uint8))
    
    return list_imgs_interp



def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class BlendingPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-guided image to image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.__init__
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_latents(self, image1, image2, timesteps, dtype, device, generator=None):
        """Returns a stack of latents at all timesteps"""

        image1 = image1.to(device=device, dtype=dtype)
        image2 = image2.to(device=device, dtype=dtype)
        init_latents1 = self.vae.encode(image1).latent_dist.sample(generator).double()
        init_latents2 = self.vae.encode(image2).latent_dist.sample(generator).double()
        latents1 = [0.18215 * torch.cat([init_latents1], dim=0)]
        latents2 = [0.18215 * torch.cat([init_latents2], dim=0)]
        shape = latents1[-1].shape
        t_prev = None
        sch = self.scheduler
        if not (isinstance(sch, DDIMScheduler) or isinstance(sch, PNDMScheduler)):
            raise NotImplementedError

        # sch.betas *= 1-self.strength
        # sch.alphas = 1.0 - sch.betas
        # sch.alphas_cumprod = torch.cumprod(sch.alphas, dim=0)
        # sch.final_alpha_cumprod = sch.alphas_cumprod[0]
        sch.alphas_cumprod = sch.alphas_cumprod.to(
            device=init_latents1.device, dtype=init_latents1.dtype)
        for t_now in timesteps[::self.steps_per_frame].flip(0):
            noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.double)#dtype)
            latents1.append(self.add_more_noise(latents1[-1], noise, t_now, t_prev))
            latents2.append(self.add_more_noise(latents2[-1], noise, t_now, t_prev))
            t_prev = t_now
        sch.alphas_cumprod = sch.alphas_cumprod.to(dtype=dtype)
        return [l.to(dtype=dtype) for l in latents1], [l.to(dtype=dtype) for l in latents2]

    def add_more_noise(self, latents, noise, t2, t1=None):
        sch = self.scheduler
        if t1 is None:
            return sch.add_noise(latents, noise, t2)

        t1 = t1.to(latents.device)
        t2 = t2.to(latents.device)

        a1 = sch.alphas_cumprod[t1] ** 0.5
        while len(a1.shape) < len(latents.shape):
            a1 = a1.unsqueeze(-1)

        var1 = 1 - sch.alphas_cumprod[t1]
        while len(var1.shape) < len(latents.shape):
            var1 = var1.unsqueeze(-1)

        a2 = sch.alphas_cumprod[t2] ** 0.5
        while len(a2.shape) < len(latents.shape):
            a2 = a2.unsqueeze(-1)

        var2 = 1 - sch.alphas_cumprod[t2]
        while len(var2.shape) < len(latents.shape):
            var2 = var2.unsqueeze(-1)

        scale = a2/a1
        sigma = (var2 - scale**2 * var1).sqrt()
        return scale * latents + sigma * noise

    @torch.no_grad()
    def __call__(
        self,
        image1: Union[torch.FloatTensor, PIL.Image.Image] = None,
        image2: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt: Optional[str] = "",
        strength: Optional[float] = 0.7,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_dir: Optional[str] = None,
        steps_per_frame: Optional[int] = 16,
    ):
        self.strength = strength
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self._encode_prompt(
            prompt, device, do_classifier_free_guidance, negative_prompt
        )
        self.steps_per_frame = steps_per_frame
        if output_dir is None:
            output_dir = '.'
        os.makedirs(output_dir, exist_ok=True)
        dtype = text_embeddings.dtype

        # 4. Preprocess image
        image1 = preprocess(image1)
        image2 = preprocess(image2)

        # 5. set timesteps
        sch = self.scheduler
        sch.set_timesteps(num_inference_steps, device=device)
        sch.timesteps = sch.timesteps[int(len(sch.timesteps)*self.strength):]
        if (t := len(sch.timesteps) % self.steps_per_frame) != 0:
            sch.timesteps = sch.timesteps[t:]
        timesteps = sch.timesteps
        F = len(timesteps)*2 + 1

        # 6. Prepare latent variables
        with torch.autocast('cuda'):
            latents1, latents2 = self.get_latents(
                image1, image2, timesteps, dtype, device, generator
            )
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        T = len(timesteps)
        total_frames = 2**(T // self.steps_per_frame) + 1
        step = total_frames - 1
        latents = [None] * total_frames

        # 8. Denoising loop
        N = self.steps_per_frame * sum([2**x-1 for x in range(1, T//self.steps_per_frame+1)])
        with self.progress_bar(total=N) as progress_bar:
            for i, t in enumerate(timesteps):
                if i % self.steps_per_frame == 0:
                    latents[0] = latents1[-i//self.steps_per_frame-1]
                    latents[-1] = latents2[-i//self.steps_per_frame-1]
                    step //= 2

                    if i // self.steps_per_frame == 2:
                        latents[total_frames // 2] = interpolate_spherical(
                            latents[total_frames // 4], latents[3 * total_frames // 4], .5)
                    # elif i // self.steps_per_frame == 3:
                    #     latents[total_frames // 2] = interpolate_spherical(
                    #         latents[3 * total_frames // 8], latents[5 * total_frames // 8], .5)

                    for frame_ix in range(step, total_frames-1, step*2):
                        assert latents[frame_ix] is None
                        frac = .5
                        if frame_ix-step == 0:
                            frac -= .25
                        if frame_ix+step == total_frames-1:
                            frac += .25
                        latents[frame_ix] = interpolate_spherical(
                            latents[frame_ix-step], latents[frame_ix+step], frac)

                for frame_ix in range(step, total_frames-1, step): # exclude endpoints
                    latent_model_input = torch.cat([latents[frame_ix]] * 2) if do_classifier_free_guidance else latents[frame_ix]
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    with torch.autocast('cuda'):
                        noise_pred = self.unet(latent_model_input, t,
                            encoder_hidden_states=text_embeddings).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents[frame_ix] = self.scheduler.step(noise_pred, t, latents[frame_ix], **extra_step_kwargs).prev_sample
                    progress_bar.update()
                

        latents[0] = latents1[0]
        latents[-1] = latents2[0]

        with torch.autocast('cuda'):
            list_imgs = [self.decode_latents(latents[i]) for i in range(len(latents))]
        # list_imgs = add_frames_linear_interp(list_imgs, nmb_frames_target = len(list_imgs)*2-1)
        for i,image in enumerate(list_imgs):
            self.numpy_to_pil(image)[0].save(os.path.join(output_dir, f'{i:03d}.png'))


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt):
        r"""Encodes the prompt into text encoder hidden states.
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        with torch.autocast('cuda'):
            text_embeddings = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.view(bs_embed, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings


if __name__ == "__main__":
    pipe = BlendingPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        # "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
    ).to("cuda")
    # pipe.enable_attention_slicing()
    from os.path import expanduser

    image1 = Image.open(expanduser('04.png')).convert('RGB')
    image2 = Image.open(expanduser('05.png')).convert('RGB')
    folder = "./charmander"
    shutil.rmtree(folder, ignore_errors=True)
    frame_filepaths = pipe(
        image1, image2,
        prompt="""a close up of a pokemon on a white background, 
fire type, charmander, charmeleon, orange-red, large eyes, flame tail, 
upright, by Ken Sugimori, has fire powers, 
official splash art""",
        num_inference_steps=70,
        strength=0.,
        steps_per_frame=12,
        guidance_scale=7,
        negative_prompt="blurry, photograph, text, lopsided, twisted",
        output_dir=folder,
    )

    # import subprocess, glob
    # folder = "./out"
    # cmd = [
    #     'ffmpeg',
    #     '-y',
    #     '-vcodec', 'png',
    #     '-r', '40',
    #     '-start_number', '0',
    #     '-i', f'{folder}/%03d.png',
    #     '-frames:v', str(len(glob.glob(f'./{folder}/*.png'))),
    #     '-c:v', 'libx264',
    #     '-vf', 'fps=40',
    #     '-pix_fmt', 'yuv420p',
    #     '-crf', '17',
    #     '-preset', 'veryfast',
    #     '-pattern_type', 'sequence',
    #     './gengar.mp4',
    # ]
    # process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = process.communicate()