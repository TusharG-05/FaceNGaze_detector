from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..services.camera import CameraService
import time

router = APIRouter()
camera_service = CameraService()

def frame_generator():
    last_processed_id = -1
    
    while True:
        frame, current_id = camera_service.get_frame()
        
        if frame is None:
            time.sleep(0.01)
            continue
            
        if current_id == last_processed_id:
            # No new frame yet, fast sleep to check again
            time.sleep(0.001)
            continue
            
        last_processed_id = current_id
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@router.get("/video_feed")
async def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
