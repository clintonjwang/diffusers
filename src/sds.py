import pdb
from functools import partial
import numpy as np
import torch
F = torch.nn.functional
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from tqdm import trange
from typing import Any, Dict, Callable, List, Optional, Union
from diffusers.utils import (
    randn_tensor,
)
nn = torch.nn

class SDSStableDiffusionPipeline(StableDiffusionPipeline):
    def sds(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        lr = 1,
        num_sds_steps = 40,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )#.half()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.flip(dims=[-1])

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels

        self.network = nn.Sequential(nn.Linear(28,64), nn.ReLU(), nn.Linear(64,num_channels_latents)).to(device=device)
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,96), torch.linspace(-1,1,96)), dim=-1)
        freqs = [2**L * torch.sin(coords*torch.pi) for L in range(7)]
        freqs += [2**L * torch.cos(coords*torch.pi) for L in range(7)]
        freqs = torch.cat(freqs, dim=-1).unsqueeze(0)
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height, width, prompt_embeds.dtype, device, generator, latents,
        # )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        self.min_step = num_inference_steps-1
        self.max_step = num_inference_steps
        self.lower_bound = 0#int(.01*num_inference_steps)

        # 7. Denoising loop
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=prompt_embeds.dtype)
        with self.progress_bar(total=num_sds_steps) as progress_bar:
            for i in range(num_sds_steps):#i, t in enumerate(timesteps):
                latents = self.network(freqs.to(device=device)).to(dtype=prompt_embeds.dtype)
                latents = latents.permute(0,3,1,2)

                t = timesteps[np.random.randint(self.min_step, self.max_step)].unsqueeze(0)
                add_t = (t * 0.1).round().long()
                if self.min_step > self.lower_bound:# and i % 2 == 0:
                    self.min_step -= 1
                    self.max_step -= 1

                if i == 20:
                    pdb.set_trace()
                    self.sweep2d(latents, generator, device, prompt_embeds, guidance_scale, cross_attention_kwargs)
                    
                shape = latents.shape
                noise = 0 * randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
                noisy_latents = self.scheduler.add_noise(latents, noise, add_t)
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                denoised_latents = self.scheduler.step(noise_pred, t, noisy_latents, **extra_step_kwargs).prev_sample
                # if i % 10 == 0:
                #     image = self.decode_latents(noisy_latents)
                #     image = self.numpy_to_pil(image)
                #     image[0].save(f"out0/noisy_{i:03d}.png")
                #     image = self.decode_latents(denoised_latents)
                #     image = self.numpy_to_pil(image)
                #     image[0].save(f"out0/denoised_{i:03d}.png")
                #     image = self.decode_latents(latents)
                #     image = self.numpy_to_pil(image)
                #     image[0].save(f"out0/latents_{i:03d}.png")

                latents = latents + lr * (denoised_latents - latents)
                
                matching_error = (denoised_latents - latents).pow(2).mean()
                matching_error.backward()
                self.network.parameters()

                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

    def sweep2d(self, latents, generator, device, prompt_embeds, guidance_scale, cross_attention_kwargs):
        do_classifier_free_guidance = guidance_scale > 1.0
        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        times = self.scheduler.timesteps.flip(dims=[-1])[3:-1]
        for i, add_t in enumerate(times):
            for j, est_t in enumerate(times, start=i):
                noisy_latents = self.scheduler.add_noise(latents, noise, add_t)
            
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, est_t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    est_t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                denoised_latents = noisy_latents - noise_pred
                image = self.decode_latents(denoised_latents)
                image = self.numpy_to_pil(image)
                image[0].save(f"out0/{i:03d}_{j:03d}.png")

        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)
        image[0].save(f"out0/initial.png")
        pdb.set_trace()