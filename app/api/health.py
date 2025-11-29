from fastapi import APIRouter
import time
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    return {"status":"ok","service":"transcribe-service","timestamp": time.time()}
