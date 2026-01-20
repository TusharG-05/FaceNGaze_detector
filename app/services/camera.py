import cv2
import threading
import time
import os
from .face import FaceDetector
from .gaze import GazeDetector

class CameraService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CameraService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._initialized = True
        
        self.camera = None
        self.face_detector = None
        self.gaze_detector = None
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.frame_id = 0
        self.frame_lock = threading.Lock()

    def start(self, video_source=0):
        if self.running: return
        
        # Init Detectors
        print("Initializing Detectors...")
        known_path = "app/assets/known_person.jpg"
        if os.path.exists(known_path):
            try:
                self.face_detector = FaceDetector(known_person_path=known_path)
                print("FaceDetector ready.")
            except Exception as e:
                print(f"FaceDetector init failed: {e}")
        else:
            print(f"Warning: {known_path} not found. Face Auth disabled until upload.")
        
        gaze_path = "app/assets/face_landmarker.task"
        print(f"Checking Gaze Model at: {os.path.abspath(gaze_path)}")
        if os.path.exists(gaze_path):
            try:
                self.gaze_detector = GazeDetector(model_path=gaze_path, max_faces=1)
                print("GazeDetector init successful.")
            except Exception as e:
                print(f"GazeDetector init failed: {e}")
        else:
            print(f"ERROR: Gaze model not found at {gaze_path}")

        # Open Camera
        self.camera = cv2.VideoCapture(video_source)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
            
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Camera processing thread started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.camera:
            self.camera.release()
        if self.face_detector:
            self.face_detector.close()
        if self.gaze_detector:
            self.gaze_detector.close()

    def _process_loop(self):
        last_face_status = (False, 1.0, 0, [])
        last_gaze_status = "Initializing..."
        
        while self.running:
            success, frame = self.camera.read()
            if not success:
                # Loop if file
                if self.camera.get(cv2.CAP_PROP_POS_FRAMES) == self.camera.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.1)
                continue

            # 1. Detection
            if self.face_detector:
                f_res = self.face_detector.process_frame(frame)
                if f_res[0] is not None:
                    last_face_status = f_res
            
            if self.gaze_detector:
                g_res = self.gaze_detector.process_frame(frame)
                if g_res:
                    last_gaze_status = g_res

            # 2. Annotation
            found, dist, num_faces, locations = last_face_status
            display_status = last_gaze_status
            
            if num_faces > 1: display_status = "ERROR: Multiple Faces"
            elif num_faces == 0: display_status = "No Face"

            auth_color = (0, 255, 0) if (found and num_faces == 1) else (0, 0, 255)
            gaze_color = (0, 0, 255) if "WARNING" in str(display_status) else (255, 255, 0)
            
            cv2.putText(frame, f"Auth: {found} ({dist:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auth_color, 2)
            cv2.putText(frame, f"Gaze: {display_status}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
            
            for (top, right, bottom, left) in locations:
                cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0) if found else (0,0,255), 2)

            # 3. Store
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                with self.frame_lock:
                    self.latest_frame = buffer.tobytes()
                    self.frame_id += 1

    def update_identity(self, image_bytes):
        """Updates the known person identity and reloads the detector."""
        filepath = "app/assets/known_person.jpg"
        
        # Save new file
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        print(f"Identity updated. Reloading FaceDetector from {filepath}...")
        
        # Stop old detector
        if self.face_detector:
            self.face_detector.close()
            self.face_detector = None
            
        # Start new detector
        try:
            self.face_detector = FaceDetector(known_person_path=filepath)
            print("FaceDetector reloaded successfully.")
            return True
        except Exception as e:
            print(f"Failed to reload FaceDetector: {e}")
            return False

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame, self.frame_id
