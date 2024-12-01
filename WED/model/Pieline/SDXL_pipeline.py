import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
    PipelineImageInput,
    rescale_noise_cfg,
)
import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.attention import Attention
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def degrade_proportionally(i, max_value=1, num_inference_steps=49,  gamma=0):
    v = max_value * (1 - i / num_inference_steps)**gamma
    return v if (v>0) else v*0.3

class LeditsAttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        if attn.shape[1] <= self.max_size:
            bs = 1 + int(PnP) + editing_prompts
            skip = 2 if PnP else 1  # skip PnP & unconditional
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            source_batch_size = int(attn.shape[1] // bs)
            self.forward(attn[:, skip * source_batch_size :], is_cross, place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        self.step_store[key].append(attn)

    def between_steps(self, store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_attention(self, step: int):
        if self.average:
            attention = {
                key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
            }
        else:
            assert step is not None
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(
        self, attention_maps, prompts, res: Union[int, Tuple[int]], from_where: List[str], is_cross: bool, select: int
    ):
        out = [[] for x in range(self.batch_size)]
        if isinstance(res, int):
            num_pixels = res**2
            resolution = (res, res)
        else:
            num_pixels = res[0] * res[1]
            resolution = res[:2]

        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(prompts), -1, *resolution, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out

    def __init__(self, average: bool, batch_size=1, max_resolution=16, max_size: int = None):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size
        if max_size is None:
            self.max_size = max_resolution**2
        elif max_size is not None and max_resolution is None:
            self.max_size = max_size
        else:
            raise ValueError("Only allowed to set one of max_resolution or max_size")

import math
# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LeditsGaussianSmoothing
class LeditsGaussianSmoothing:
    def __init__(self, device):
        kernel_size = [3, 3]
        sigma = [0.5, 0.5]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel.to(device)

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))

# Copied from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEDITSCrossAttnProcessor
class LEDITSCrossAttnProcessor:
    def __init__(self, attention_store, place_in_unet, pnp, editing_prompts):
        self.attnstore = attention_store
        self.place_in_unet = place_in_unet
        self.editing_prompts = editing_prompts
        self.pnp = pnp

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        temb=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(
            attention_probs,
            is_cross=True,
            place_in_unet=self.place_in_unet,
            editing_prompts=self.editing_prompts,
            PnP=self.pnp,
        )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class StableDiffusionXLDecompositionPipeline(StableDiffusionXLImg2ImgPipeline):
    

    
    
    def prepare_text_embeddings(
        self,
        prompt_embeds,
        negative_prompt_embeds,
        batch_size,
        num_images_per_prompt,
        device
    ):
        # 1. Validate input dimensions
        if prompt_embeds.shape[-2] != 640:  # Expected sequence length
            # Reshape or truncate to correct size
            prompt_embeds = prompt_embeds[:, :640, :]
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds[:, :640, :]
        
        # 2. Ensure correct batch size
        prompt_embeds = prompt_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)

        # 3. Move to correct device and dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=self.unet.dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=self.unet.dtype)

        # 4. Validate final shapes
        expected_shape = (batch_size * num_images_per_prompt, 640, -1)
        if prompt_embeds.shape[:2] != expected_shape[:2]:
            raise ValueError(
                f"Unexpected prompt_embeds shape: {prompt_embeds.shape}. "
                f"Expected: {expected_shape}"
            )

        return prompt_embeds, negative_prompt_embeds
    
    def prepare_unet(self, attention_store, PnP: bool = False):
        attn_procs = {}
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            if "attn2" in name and place_in_unet != "mid":
                attn_procs[name] = LEDITSCrossAttnProcessor(
                    attention_store=attention_store,
                    place_in_unet=place_in_unet,
                    pnp=PnP,
                    editing_prompts=self.enabled_editing_prompts,
                )
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)
    
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        omega = 1,
        gamma = 0,
        inv_latents = None,
        prompt_embeds_ref = None,
        added_cond_kwargs_ref = None,
        mask = None,
        t_exit=15,
        sem_guidance: Optional[List[torch.Tensor]] = [],
        store_averaged_over_steps: bool = True,
        edit_threshold: Optional[Union[float, List[float]]] = 0.9,
        edit_guidance_scale: Optional[Union[float, List[float]]] = [5.0, 10.0],
        reverse_editing_direction: Optional[Union[bool, List[bool]]] = False,
        **kwargs,
    ):
        

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        self.smoothing = LeditsGaussianSmoothing(self.device)
        device = self._execution_device
        self.enabled_editing_prompts = len(prompt)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2, 
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )


        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(self.denoising_end, float) and 0 < dnv < 1

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        add_noise = True if self.denoising_start is None else False
        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            add_noise,
        )
        batch_sizeI = image.shape[0]

        self.attention_store = LeditsAttentionStore(
                average=store_averaged_over_steps,
                batch_size=batch_sizeI,
                max_size=(latents.shape[-2] / 4.0) * (latents.shape[-1] / 4.0),
                max_resolution=None,
            )
        
        self.prepare_unet(self.attention_store)
        resolution = latents.shape[-2:]
        att_res = (int(resolution[0] / 4), int(resolution[1] / 4))
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        # add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        # add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
        # add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
        self.text_cross_attention_maps = prompt

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )


        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 9.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and self.denoising_start is not None
            and denoising_value_valid(self.denoising_end)
            and denoising_value_valid(self.denoising_start)
            and self.denoising_start >= self.denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {self.denoising_end} when using type float."
            )
        elif self.denoising_end is not None and denoising_value_valid(self.denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        reference_latents = latents
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                 
                reference_latents = inv_latents[num_inference_steps - i]
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * (1 + self.enabled_editing_prompts)) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # get the inversion latents from list
                reference_latents = (0.5 * latents + 0.5 * reference_latents) 

                reference_model_input = torch.cat([reference_latents] * 2) if self.do_classifier_free_guidance else reference_latents

                reference_model_input = self.scheduler.scale_model_input(reference_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds, # null prompt
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                scaling_factor = degrade_proportionally(i, omega, num_inference_steps-1, gamma)
               
                #############
                ###############
                noise_pred_out = noise_pred.chunk(1 + self.enabled_editing_prompts)  # [b,4, 64, 64]
                noise_pred_uncond = noise_pred_out[0]
                noise_pred_edit_concepts = noise_pred_out[1:]
                

                noise_guidance_edit = torch.zeros(
                    noise_pred_uncond.shape,
                    device=self.device,
                    dtype=noise_pred_uncond.dtype,
                )
                self.sem_guidance = []
                if not hasattr(self, "activation_mask"):
                    self.activation_mask = {}  # Tạo dạng dictionary

                # print("svsbs: ", noise_pred_edit_concepts, noise_pred.shape, self.enabled_editing_prompts, timesteps)

            

                # if self.sem_guidance is None:
                #     self.sem_guidance = torch.zeros((len(timesteps), *noise_pred_uncond.shape))

                for c in range(len(prompt)):
                    noise_guidance_edit_tmp = noise_pred*0.2

                    if reverse_editing_direction[c]:
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1

                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * edit_guidance_scale[c]
                    ######### attn mask
                    out = self.attention_store.aggregate_attention(
                        attention_maps=self.attention_store.step_store,
                        prompts=self.text_cross_attention_maps,
                        res=att_res,
                        from_where=["up", "down"],
                        is_cross=True,
                        select=self.text_cross_attention_maps.index(prompt[c]),
                    )
                    # Sử dụng toàn bộ attention map
                    attn_map = out[:, :, :, 1:]
                    # Trung bình trên tất cả các token
                    attn_map = torch.sum(attn_map, dim=3)
                    attn_map = F.pad(attn_map.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
                    attn_map = self.smoothing(attn_map).squeeze(1)

                    # torch.quantile function expects float32
                    if attn_map.dtype == torch.float32:
                        tmp = torch.quantile(attn_map.flatten(start_dim=1), edit_threshold[c], dim=1)
                    else:
                        tmp = torch.quantile(
                            attn_map.flatten(start_dim=1).to(torch.float32), edit_threshold[c], dim=1
                        ).to(attn_map.dtype)
                    attn_mask = torch.where(
                        attn_map >= tmp.unsqueeze(1).unsqueeze(1).repeat(1, *att_res), 1.0, 0.0
                    )

                    # resolution must match latent space dimension
                    attn_mask = F.interpolate(
                        attn_mask.unsqueeze(1),
                        noise_guidance_edit_tmp.shape[-2:],  # 64,64
                    ).repeat(1, 4, 1, 1)
                    self.activation_mask[i, c] = attn_mask.detach().cpu()
                    
                    ########### intersect_mask
                    noise_guidance_edit_tmp_quantile = torch.abs(noise_guidance_edit_tmp)
                    noise_guidance_edit_tmp_quantile = torch.sum(
                        noise_guidance_edit_tmp_quantile, dim=1, keepdim=True
                    )
                    noise_guidance_edit_tmp_quantile = noise_guidance_edit_tmp_quantile.repeat(
                        1, self.unet.config.in_channels, 1, 1
                    )

                    # torch.quantile function expects float32
                    if noise_guidance_edit_tmp_quantile.dtype == torch.float32:
                        tmp = torch.quantile(
                            noise_guidance_edit_tmp_quantile.flatten(start_dim=2),
                            edit_threshold[c],
                            dim=2,
                            keepdim=False,
                        )
                    else:
                        tmp = torch.quantile(
                            noise_guidance_edit_tmp_quantile.flatten(start_dim=2).to(torch.float32),
                            edit_threshold[c],
                            dim=2,
                            keepdim=False,
                        ).to(noise_guidance_edit_tmp_quantile.dtype)

                    intersect_mask = (
                        torch.where(
                            noise_guidance_edit_tmp_quantile >= tmp[:, :, None, None],
                            torch.ones_like(noise_guidance_edit_tmp),
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )
                        * attn_mask # attn mask
                    )
                    print(intersect_mask.sum(), noise_guidance_edit_tmp.sum(), c)

                    self.activation_mask[i, c] = intersect_mask.detach().cpu()

                    noise_guidance_edit_tmp = noise_guidance_edit_tmp * intersect_mask
                    
                    mask = intersect_mask
                    
                    noise_guidance_edit += noise_guidance_edit_tmp

                self.sem_guidance.append(noise_guidance_edit.detach().cpu())
                

                ################
                ###########

                # Combine masks
                print ("   ",scaling_factor)
                scaling_factor = scaling_factor * noise_guidance_edit
                #print (scaling_factor)
                # print('------------------')
                

                
                
                if  i < t_exit :
                    noise_pred_recon = self.unet(
                        reference_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_ref, # original latent
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs_ref,
                        return_dict=False,
                    )[0]
                    noise_pred = (noise_pred + scaling_factor * (noise_pred_recon - noise_pred))
                else:
                    noise_pred_fwd = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds_ref, # c prompt
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs_ref,
                    return_dict=False,
                    )[0]
                    noise_pred = noise_pred + scaling_factor * (noise_pred_fwd - noise_pred) 

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred_uncond, noise_pred_text_recon = noise_pred_recon.chunk(2)
                    noise_pred_recon = noise_pred_uncond + self.guidance_scale * (noise_pred_text_recon - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                    noise_pred_recon = rescale_noise_cfg(noise_pred_recon, noise_pred_text_recon, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop("add_neg_time_ids", add_neg_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                mask = mask.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            mask = self.vae.decode(mask / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image), StableDiffusionXLPipelineOutput(images=mask)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)
            mask = self.watermark.apply_watermark(mask)

        image = self.image_processor.postprocess(image, output_type=output_type)
        mask = self.image_processor.postprocess(mask, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, mask,)

        return StableDiffusionXLPipelineOutput(images=image), StableDiffusionXLDecompositionPipeline(images=mask)
