from fastapi import APIRouter
from WED.routes.i2i import i2i, t2i

router = APIRouter()
router.include_router(t2i, prefix="/t2i")
router.include_router(i2i, prefix="/i2i")
