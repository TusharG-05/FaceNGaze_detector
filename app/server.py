import contextlib
import os
from fastapi import FastAPI
from .routers import video, site, settings
from .services.camera import CameraService

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    video_path = "video.mp4"
    src = video_path if os.path.exists(video_path) else 0
    
    print(f"Starting CameraService with source: {src}")
    service = CameraService()
    service.start(src)
    
    yield
    
    # Shutdown
    print("Stopping CameraService...")
    service.stop()

app = FastAPI(lifespan=lifespan)

app.include_router(site.router)
app.include_router(video.router)
app.include_router(settings.router)


