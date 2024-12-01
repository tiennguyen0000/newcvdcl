from io import BytesIO
import base64
from PIL import Image
from pydantic import BaseModel

class TextInput(BaseModel):
    txt: str
    
class tai_rps(BaseModel):
  prompt : str
  prompt_fw : str
  img_base64 : str

# support input
def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return {'img_base64': img_str}

def base64_to_image(base64_str: str) -> Image.Image:

    image_data = base64.b64decode(base64_str)
    buffered = BytesIO(image_data)
    img = Image.open(buffered)
    return img
