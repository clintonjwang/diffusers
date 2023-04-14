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
from diffusers import EulerDiscreteScheduler

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
        lr = 1e-3,
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

        n_freqs = 7
        self.network = nn.Sequential(nn.Linear(n_freqs*4, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, num_channels_latents)).to(device=device)
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,96), torch.linspace(-1,1,96)), dim=-1)
        freqs = [torch.sin(2**L * coords*torch.pi) for L in range(n_freqs)]
        freqs += [torch.cos(2**L * coords*torch.pi) for L in range(n_freqs)]
        freqs = torch.cat(freqs, dim=-1).unsqueeze(0)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )

        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        
        self.min_step = num_inference_steps-1
        self.mid_step = num_inference_steps-1
        self.max_step = num_inference_steps
        step_reduction_schedule = [i for i in range(0,num_sds_steps,1) if i < 120 or i > 160]
        mini_scheduler = EulerDiscreteScheduler.from_config(self.scheduler.config)

        # 7. Denoising loop
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=device, dtype=prompt_embeds.dtype)
        for i in range(num_sds_steps):#i, t in enumerate(timesteps):
            latents = self.network(freqs.to(device=device)).to(dtype=prompt_embeds.dtype)
            latents = latents.permute(0,3,1,2)

            with torch.no_grad():
                t_ix = np.random.randint(self.mid_step, self.max_step)
                t = timesteps[t_ix].unsqueeze(0)
                add_t = timesteps[np.random.randint(self.min_step, t_ix+1)].unsqueeze(0)
                if i in step_reduction_schedule:
                    self.min_step -= 1
                    self.mid_step -= 1
                    self.max_step -= 1

                shape = latents.shape
                noise = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
                noisy_latents = self.scheduler.add_noise(latents, noise, add_t)
                
                if i == 140:
                    self.sweep2d(latents, generator, device, prompt_embeds, guidance_scale, cross_attention_kwargs)
                
                if i > 10:
                    # mini_scheduler.alphas_cumprod = mini_scheduler.alphas_cumprod.cpu()
                    mini_scheduler.set_timesteps(20, device=device)
                    T_local = mini_scheduler.timesteps[mini_scheduler.timesteps < t]
                    x_cur = noisy_latents

                    for i, t in enumerate(T_local):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([x_cur] * 2) if do_classifier_free_guidance else x_cur
                        latent_model_input = mini_scheduler.scale_model_input(latent_model_input, t)

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
                        x_cur = mini_scheduler.step(noise_pred, t, x_cur).prev_sample
                    denoised_latents = x_cur
                else:
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
                    latent_model_input = latent_model_input / latent_model_input.std()
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, add_t)
                    
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    model_output = noise_pred
                    sample = noisy_latents
                    step_index = (self.scheduler.timesteps == t).nonzero().item()
                    sigma = self.scheduler.sigmas[step_index]
                    pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
                    derivative = (sample - pred_original_sample) / sigma
                    if True: # first order (Euler)
                        dt = -sigma #self.scheduler.sigmas[step_index+1] - sigma
                        denoised_latents = sample + derivative * dt
                    else: # two Euler steps
                        ix_next = (len(self.scheduler.sigmas) - step_index)//2 + step_index
                        t_next = self.scheduler.timesteps[ix_next]
                        sigma_next = self.scheduler.sigmas[ix_next]
                        dt = sigma_next - sigma
                        if i % 22 == 0:
                            image = self.decode_latents(sample - derivative * sigma)
                            image = self.numpy_to_pil(image)
                            image[0].save(f"out0/1step_{i:02d}.png")
                        intermediate_latents = sample + derivative * dt
                        latent_model_input = torch.cat([intermediate_latents] * 2)
                        # latent_model_input = latent_model_input / latent_model_input.std()
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_next)
                        noise_pred = self.unet(
                            latent_model_input,
                            t_next,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        pred_sample = noise_pred * (-sigma_next / (sigma_next**2 + 1) ** 0.5) + (intermediate_latents / (sigma_next**2 + 1))
                        next_deriv = (intermediate_latents - pred_sample) / sigma_next
                        dt = - sigma_next
                        denoised_latents = intermediate_latents + next_deriv * dt
                    # latents = denoised_latents
                
            if i % 2 == 0:
                with torch.no_grad():
                    image = self.decode_latents(denoised_latents.detach())
                    image = self.numpy_to_pil(image)
                    image[0].save(f"out0/denoised_{i:02d}.png")
                    image = self.decode_latents(latents.detach())
                    image = self.numpy_to_pil(image)
                    image[0].save(f"out0/latents_{i:02d}.png")

            if i > 10:
                n_reps = 1000
            else:
                n_reps = 100
            # if i > 50 and n_reps > 20:
            #     n_reps -= 1
            for _ in range(n_reps):
                matching_error = (denoised_latents - latents).pow(2).mean()
                matching_error.backward()
                optimizer.step()
                optimizer.zero_grad()
                latents = self.network(freqs.to(device=device)).to(dtype=prompt_embeds.dtype)
                latents = latents.permute(0,3,1,2)
            scheduler.step()

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        with torch.no_grad():
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
        times = self.scheduler.timesteps.flip(dims=[-1])[::5]
        for i, add_t in enumerate(times):
            for j, est_t in enumerate(times[i:], start=i):
                noisy_latents = self.scheduler.add_noise(latents, noise, add_t.unsqueeze(0))
            
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

                # prev_timestep = 0
                # alpha_prod_t = self.scheduler.alphas_cumprod[est_t]
                # alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
                # beta_prod_t = 1 - alpha_prod_t
                # pred_original_sample = (alpha_prod_t**0.5) * noisy_latents - (beta_prod_t**0.5) * noise_pred
                # pred_epsilon = (alpha_prod_t**0.5) * noise_pred + (beta_prod_t**0.5) * noisy_latents
                # pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon
                # denoised_latents = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                model_output = noise_pred
                sample = noisy_latents
                step_index = (self.scheduler.timesteps == est_t).nonzero().item()
                sigma = self.scheduler.sigmas[step_index]
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
                derivative = (sample - pred_original_sample) / sigma
                dt = -sigma
                denoised_latents = sample + derivative * dt

                image = self.decode_latents(denoised_latents)
                image = self.numpy_to_pil(image)
                image[0].save(f"out0/{i:02d}_{j:02d}.png")

        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)
        image[0].save(f"out0/initial.png")