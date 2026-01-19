import cv2
import multiprocessing
import numpy as np
import mediapipe as mp
import time

def gaze_worker(frame_queue, result_queue, max_faces):
    mp_face_mesh = mp.solutions.face_mesh
    # Set max_num_faces to a high number to detect everyone, so we can enforce our own limit.
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=10, 
        refine_landmarks=True,
        min_detection_confidence=0.3, # Lowered for long distance
        min_tracking_confidence=0.3
    )

    while True:
        try:
            # Wait for frame
            try:
                frame_data = frame_queue.get(timeout=1)
            except:
                continue
            
            if frame_data is None:
                break
                
            rgb_frame = frame_data
            
            try:
                results = face_mesh.process(rgb_frame)
                
                status_message = "No Face Detected"
                
                if results.multi_face_landmarks:
                    num_faces = len(results.multi_face_landmarks)
                    
                    if num_faces > max_faces:
                        status_message = f"ERROR: Too many faces ({num_faces} > {max_faces})"
                    else:
                        # Process the first face (assuming the main user is the primary face)
                        # In a future iteration, we could select the largest face.
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        mesh_points = np.array([np.multiply([p.x, p.y], [rgb_frame.shape[1], rgb_frame.shape[0]]).astype(int) for p in face_landmarks.landmark])
                        
                        # --- IRIS RATIO LOGIC ---
                        # Left Eye: 33 (Left), 133 (Right), 468 (Iris)
                        # Right Eye: 362 (Left), 263 (Right), 473 (Iris)
                        
                        def get_ratio(eye_points, iris_center):
                            lc = eye_points[0]
                            rc = eye_points[1]
                            full_width = np.linalg.norm(rc - lc)
                            if full_width == 0: return 0.5
                            dist_to_left = np.linalg.norm(iris_center - lc)
                            return dist_to_left / full_width

                        p33, p133, p468 = mesh_points[33], mesh_points[133], mesh_points[468]
                        ratio_left = get_ratio([p33, p133], p468)
                        
                        p362, p263, p473 = mesh_points[362], mesh_points[263], mesh_points[473]
                        ratio_right = get_ratio([p362, p263], p473)
                        
                        avg_ratio = (ratio_left + ratio_right) / 2
                        
                        # --- 30 DEGREE ANGLE LOGIC ---
                        # Calibrating: 
                        # 0.5 is Center.
                        # We assume Safe Zone is roughly 0.35 to 0.65.
                        # < 0.35 -> Looking Left (Camera Right) > 30 deg
                        # > 0.65 -> Looking Right (Camera Left) > 30 deg
                        # Note: Directions are mirrored relative to the person vs camera.
                        
                        SAFE_MIN = 0.35
                        SAFE_MAX = 0.65
                        
                        if avg_ratio < SAFE_MIN:
                            status_message = "WARNING: Gaze > 30 deg (Right)"
                        elif avg_ratio > SAFE_MAX:
                            status_message = "WARNING: Gaze > 30 deg (Left)"
                        else:
                            status_message = "Safe: Center"

                # ALWAYS put result to clear the "Initializing" state
                # Clear queue if full to ensure fresh data
                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except:
                        pass
                result_queue.put(status_message)

            except Exception as e:
                print(f"Gaze Processing Error: {e}")
                
        except Exception as e:
            print(f"Gaze Worker Error: {e}")

    face_mesh.close()

class GazeDetector:
    def __init__(self, max_faces=1):
        print("Initializing GazeDetector...")
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        
        self.worker = multiprocessing.Process(
            target=gaze_worker,
            args=(self.frame_queue, self.result_queue, max_faces)
        )
        self.worker.daemon = True
        self.worker.start()
        print("Gaze Worker started.")
        
    def process_frame(self, frame_bgr):
        # MediaPipe expects RGB
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            
            # Non-blocking put
            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)
                
            # Non-blocking get
            try:
                return self.result_queue.get_nowait()
            except:
                return None # Return None if no NEW result, main loop keeps old one
        except Exception as e:
            print(f"Gaze Main Error: {e}")
            return None
            
    def close(self):
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()
