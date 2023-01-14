# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import pdb, os
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

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2



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

    def get_latents(self, image, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        init_latents = self.vae.encode(image).latent_dist.sample(generator)
        return 0.18215 * torch.cat([init_latents], dim=0)

    def interpolate_latents(self, latents1, latents2, timesteps, generator, device, dtype):
        # out to in
        shape = latents1.shape
        total_frames = len(timesteps)*2-1# // self.steps_per_frame
        latents = [None] * total_frames
        latents[0] = latents1
        latents[-1] = latents2
        t_prev = None

        for t_ix in range(1, total_frames//2):#, self.steps_per_frame):
            t_now = timesteps[t_ix]
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            for ix in (t_ix, total_frames-t_ix-1):
                left = t_ix-1
                right = total_frames-t_ix
                frac = (ix - left) / (right - left)
                latents[ix] = slerp(frac, latents[left], latents[right])
                latents[ix] = self.add_more_noise(latents[ix], noise, t_now, t_prev)
            t_prev = t_now

        # repeat once for midpoint
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        ix = t_ix = total_frames//2
        t_now = timesteps[t_ix]
        left = t_ix-1
        right = total_frames-t_ix
        frac = (ix - left) / (right - left)
        latents[ix] = slerp(frac, latents[left], latents[right])
        latents[ix] = self.add_more_noise(latents[ix], noise, t_now, t_prev)
        
        return [l.float() for l in latents]

    def add_more_noise(self, latents, noise, t2, t1=None):
        if t1 is None:
            return self.scheduler.add_noise(latents, noise, t2)

        sch = self.scheduler
        if not (isinstance(sch, DDIMScheduler) or isinstance(sch, PNDMScheduler)):
            raise NotImplementedError

        sch.alphas_cumprod = sch.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)
        t1 = t1.to(latents.device)
        t2 = t2.to(latents.device)

        a1 = sch.alphas_cumprod[t1] ** 0.5
        a1 = a1.flatten()
        while len(a1.shape) < len(latents.shape):
            a1 = a1.unsqueeze(-1)

        sig1 = (1 - sch.alphas_cumprod[t1]) ** 0.5
        sig1 = sig1.flatten()
        while len(sig1.shape) < len(latents.shape):
            sig1 = sig1.unsqueeze(-1)

        a2 = sch.alphas_cumprod[t2] ** 0.5
        a2 = a2.flatten()
        while len(a2.shape) < len(latents.shape):
            a2 = a2.unsqueeze(-1)

        var2 = 1 - sch.alphas_cumprod[t2]
        var2 = var2.flatten()
        while len(var2.shape) < len(latents.shape):
            var2 = var2.unsqueeze(-1)

        scale = a2/a1
        sigma = (var2 - (scale*sig1)**2).sqrt()
        return scale * latents + sigma * noise

    @torch.no_grad()
    def __call__(
        self,
        image1: Union[torch.FloatTensor, PIL.Image.Image] = None,
        image2: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt: Optional[str] = "",
        num_inference_steps: Optional[int] = 50,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_dir: Optional[str] = None,
        steps_per_frame: Optional[int] = 1,
    ):
        device = self._execution_device
        text_embeddings = self._encode_prompt(
            prompt, device,
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
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        with torch.autocast('cuda'):
            latents1 = self.get_latents(
                image1, dtype, device, generator
            )
            latents2 = self.get_latents(
                image2, dtype, device, generator
            )
        
        latents = self.interpolate_latents(latents1, latents2, timesteps.flip(0), generator, device, dtype)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_frames = len(timesteps)*2 - 1
        with self.progress_bar(total=timesteps.size(0)) as progress_bar:
            for i, t in enumerate(timesteps):
                for j in range(num_frames//2-i, num_frames//2+i+1):
                    latent_model_input = self.scheduler.scale_model_input(latents[j], t)

                    # predict the noise residual
                    with torch.autocast('cuda'):
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                    # compute the previous noisy sample x_t -> x_t-1
                    latents[j] = self.scheduler.step(noise_pred, t, latents[j], **extra_step_kwargs).prev_sample
                progress_bar.update()

        with torch.autocast('cuda'):
            for i in range(len(latents)):
                image = self.decode_latents(latents[i])
                image = self.numpy_to_pil(image)
                image[0].save(os.path.join(output_dir, f'{i}.png'))


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(self, prompt, device):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
        """
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

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
        text_embeddings = text_embeddings.repeat(1, 1, 1)
        text_embeddings = text_embeddings.view(bs_embed, seq_len, -1)

        return text_embeddings


if __name__ == "__main__":
    pipe = BlendingPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        # "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16,
    ).to("cuda")
    # pipe.enable_attention_slicing()
    from os.path import expanduser

    image1 = Image.open(expanduser('01.png'))
    image2 = Image.open(expanduser('02.png'))
    frame_filepaths = pipe(
        image1,
        image2,
        prompt="HDR photography",
        num_inference_steps=50,
        output_dir="./out",
    )