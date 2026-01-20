from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ..services.camera import CameraService
import shutil
import os

router = APIRouter()
camera_service = CameraService()

@router.post("/upload-identity")
async def upload_identity(file: UploadFile = File(...)):
    try:
        content = await file.read()
        success = camera_service.update_identity(content)
        
        if success:
            return JSONResponse(content={"message": "Identity updated successfully", "filename": file.filename})
        else:
            raise HTTPException(status_code=500, detail="Failed to reload detector with new image")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
