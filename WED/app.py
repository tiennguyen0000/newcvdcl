import sys 
from pathlib import Path
# sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path.cwd()))
from fastapi import FastAPI
from WED.middleware import  setup_cors
from WED.routes.base import router
import nest_asyncio
import uvicorn
import ngrok

app = FastAPI()
setup_cors(app)
app.include_router(router)
# Khai báo port mặc định
port = 8000
ngrok.set_auth_token("2oeQ1IbMhVUGy8RjTOtdXMX50cI_gQkjMK5ctG2L1Srdz1ZU")
public_url = ngrok.connect(port).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, port))
# for gpu onl
nest_asyncio.apply()
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
# if local you just run: uvicorn app:app --reload