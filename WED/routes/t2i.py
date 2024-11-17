import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from fastapi import APIRouter
from WED.schemas.genI_schemas import TextInput, image_to_base64
from WED.model.call import genI

router = APIRouter()

@router.post('/gen')
def predict(txt : TextInput):
    response = genI(txt)
    return image_to_base64(response)
    # return 0

@router.post('/test')
def predictcx(txt: TextInput):
    # fix bug
    return {'txt': txt.txt}