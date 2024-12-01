import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import APIRouter
from WED.schemas.genI_schemas import tai_rps, TextInput, image_to_base64
from WED.model.call import crdeimg

router = APIRouter()
device = 'cuda'
model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_typedevice=device)
@router.post('/increase')
def predict(scal : tai_rps):
    response = crdeimg(scal, pipe_inversion, pipe_inference, omega=3)

    return image_to_base64(response)

@router.post('/decrease')
def predictcx(scal : tai_rps):
    response = crdeimg(scal, pipe_inversion, pipe_inference, omega=-3)
    # image = Image.open('/kaggle/input/dsgssdd/images.jpg')   
    return image_to_base64(response)
    # return 0

@router.post('/test')
def predictcx(txt: TextInput):
    # fix bug
    return {'txt': txt.txt}
