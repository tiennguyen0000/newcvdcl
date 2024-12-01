import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import APIRouter
from WED.schemas.genI_schemas import tai_rps, TextInput, image_to_base64
from WED.model.call import crdeimg

router = APIRouter()

@router.post('/increase')
def predict(scal : tai_rps):
    response = crdeimg(scal, omega=3)

    return image_to_base64(response)

@router.post('/decrease')
def predictcx(scal : tai_rps):
    response = crdeimg(scal, omega=-3)
    # image = Image.open('/kaggle/input/dsgssdd/images.jpg')   
    return image_to_base64(response)
    # return 0

@router.post('/test')
def predictcx(txt: TextInput):
    # fix bug
    return {'txt': txt.txt}
