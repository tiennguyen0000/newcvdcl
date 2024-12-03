import torch

from WED.model.eunms import Model_Type, Scheduler_Type
from WED.model.Utils.enums_utils import get_pipes
from WED.model.config import RunConfig
from WED.schemas.genI_schemas import base64_to_image, TextInput
from WED.model.main import run


def add_interrupt_attribute(self):
    self.interrupt = False  # Set to False by default

def crdeimg(scal,
            pipe_inversion, 
            pipe_inference,
            omega = 0,
            edit_threshold = [0.9],
            edit_guidance_scale = [10],
            reverse_editing_direction = [False, False, False],
            t_exit = 55,
            ):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    # pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
    input_image = base64_to_image(scal.img_base64)
    original_shape = input_image.size
    input_image = input_image.resize((1024, 1024))
    prompt = scal.prompt # 'smile' for "009698.jpg", 'anime' for "Arknight.jpg"

    config = RunConfig(model_type = model_type,
                        num_inference_steps = int(scal.numts),
                        num_inversion_steps = int(scal.numts),
                        num_renoise_steps = int(scal.numrs),
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 7865)
    
    _, inv_latent, _, all_latents, other_kwargs = run(input_image,
                                          prompt,
                                          config,
                                          pipe_inversion=pipe_inversion,
                                          pipe_inference=pipe_inference,
                                          do_reconstruction=False)


    pipe_inference.__class__.interrupt
    if t_exit == 0:
        t_exit = scal.numts // 3
    
    rec_image = pipe_inference(image = inv_latent,
                            prompt = scal.prompt_fw,
                            denoising_start=0.0,
                            num_inference_steps = config.num_inference_steps,
                            guidance_scale = 1.0,
                            omega=omega, # omega=3 for "009698.jpg", omega=5 for "Arknight.jpg"
                            gamma=3, # gamma=3 for "009698.jpg", gamma=3 for "Arknight.jpg"
                            inv_latents=all_latents,
                            prompt_embeds_ref=other_kwargs[0],
                            added_cond_kwargs_ref=other_kwargs[1],
                            edit_threshold = edit_threshold,
                            edit_guidance_scale = edit_guidance_scale,
                            reverse_editing_direction = reverse_editing_direction,
                            t_exit = t_exit, # t_exit=15 for "009698.jpg", t_exit=25 for "Arknight.jpg"
                            ).images[0]
    # rec_image = rec_image.images[0]
    # mask = mask.images[0]
    # mask.save("../mask.jpg")
    return rec_image.resize((1024, int(1024 * original_shape[1] / original_shape[0])))

def genI(txt: TextInput):
    from WED.model.Pieline.TTI import pipelineT2i, generator
    return pipelineT2i(txt.txt, generator=generator).images[0]
