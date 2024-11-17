from diffusers import AutoPipelineForText2Image
import torch
from WED.shemas.genI_schemas import TextInput

pipelineT2i = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")
generator = torch.Generator("cuda").manual_seed(31)

