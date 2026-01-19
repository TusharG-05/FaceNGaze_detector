import cv2
import time
import numpy as np
import multiprocessing
import os
import face_recognition

def face_recognition_worker(frame_queue, result_queue, known_encoding):
    while True:
        try:
            rgb_small_frame = frame_queue.get(timeout=1)
        except:
            continue

        if rgb_small_frame is None:
            break

        try:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            num_faces = len(face_encodings)
            
            match_found = False
            min_dist = 1.0
            
            if num_faces > 0:
                # Still check the first face for match
                distances = face_recognition.face_distance([known_encoding], face_encodings[0])
                min_dist = distances[0]
                if min_dist <= 0.5: # Threshold
                    match_found = True
            
            # Send result back (including face count and locations)
            if not result_queue.full():
                result_queue.put((match_found, min_dist, num_faces, face_locations))
                
        except Exception as e:
            print(f"Worker Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Initializing FaceDetector...")
        if not os.path.exists(known_person_path):
            raise FileNotFoundError(f"Error: {known_person_path} not found")
            
        # Load Known Face
        known_image = face_recognition.load_image_file(known_person_path)
        self.known_encoding = face_recognition.face_encodings(known_image)[0]
        print("Known face loaded.")

        # Setup Queues
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)

        # Start Worker
        self.worker = multiprocessing.Process(
            target=face_recognition_worker, 
            args=(self.frame_queue, self.result_queue, self.known_encoding)
        )
        self.worker.daemon = True
        self.worker.start()
        print("Worker process started.")

    def process_frame(self, frame_bgr):
        # 1. Resize & Convert to RGB (Increase resolution for distant faces)
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 2. Put in Queue
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small_frame)

        # 3. Check for Result
        try:
            return self.result_queue.get_nowait()
        except:
            return None, None, 0, []

    def close(self):
        # Send Sentinel to stop the worker gracefully
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate() # Fallback

# =============================
# MOCK FRONTEND (Simulating a Server)
# =============================
if __name__ == "__main__":
    # It opens the camera locally to test the Class.
    
    # Stop Button State
    stop_button_pressed = [False] # Use a list for mutable reference in callback

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in the STOP button area (bottom right)
            # We'll define the area in the draw step
            if hasattr(on_mouse_click, 'button_rect'):
                rx, ry, rw, rh = on_mouse_click.button_rect
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    stop_button_pressed[0] = True

    detector = FaceDetector()
    
    # Try to initialize GazeDetector with better error handling
    gaze_detector = None
    if not os.path.exists('face_landmarker.task'):
        print("WARNING: face_landmarker.task not found. Gaze detection will be disabled.")
        print("Please download it from MediaPipe: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    else:
        try:
            from gaze_detector import GazeDetector
            gaze_detector = GazeDetector()
        except (ImportError, Exception) as e:
            print(f"Could not initialize GazeDetector: {e}")
            gaze_detector = None
    
    # Simulate Camera Input
    video_path = "video.mp4"
    if os.path.exists(video_path):
        video = cv2.VideoCapture(video_path)
    else:
        video = cv2.VideoCapture(0)

    print("--- SIMULATING FRONTEND CONNECTION ---")
    start_time = time.time()
    
    last_found = False
    last_dist = 1.0
    last_num_faces = 0
    last_locations = []
    last_gaze = "Gaze Disabled (Missing Model)" if gaze_detector is None else "Initializing..."

    while True:
        ret, frame = video.read()
        if not ret: break

        # === THE API CALL ===
        face_result = detector.process_frame(frame)
        
        gaze_result = None
        if gaze_detector:
            gaze_result = gaze_detector.process_frame(frame)
        
        # Update State if we got a fresh result
        found, dist, n_faces, locs = face_result
        if found is not None:
            last_found = found
            last_dist = dist
            last_num_faces = n_faces
            last_locations = locs
            if found:
                print(f"API Result: Found! ({dist:.2f})")
        
        # Override status if multiple faces detected (Cheating Prevention)
        display_status = last_gaze
        if last_num_faces > 1:
            display_status = "ERROR: Multiple Faces Detected!"
        elif last_num_faces == 0:
             display_status = "No Face Detected"

        if gaze_result:
            last_gaze = gaze_result
            # Only update if no multi-face error
            if last_num_faces <= 1:
                display_status = last_gaze

        # Display (Only for testing)
        color = (0, 255, 0) if (last_found and last_num_faces == 1) else (0, 0, 255)
        text_found = f"Found: {last_found}" if last_num_faces <= 1 else "Found: False (Multi-Face)"
        
        cv2.putText(frame, f"{text_found} ({last_dist:.2f})", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Color for gaze status (Red for warnings)
        gaze_color = (255, 255, 0) # Default Cyan-ish
        if "WARNING" in display_status or "ERROR" in display_status:
            gaze_color = (0, 0, 255) # Red
            
        cv2.putText(frame, f"Status: {display_status}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
        
        # Draw STOP Button (Bottom Right)
        h, w = frame.shape[:2]
        bw, bh = 100, 40
        bx, by = w - bw - 20, h - bh - 20
        on_mouse_click.button_rect = (bx, by, bw, bh)
        
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), -1)
        cv2.putText(frame, "STOP", (bx + 15, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw Face Squares (Rescaled from 0.5x)
        for i, (top, right, bottom, left) in enumerate(last_locations):
            # Scale back up by 2 (since we now resize to 0.5)
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Use Green for the primary/matched face, Cyan for others
            box_color = (0, 255, 0) if (i == 0 and last_found and last_num_faces == 1) else (255, 255, 0)
            if last_num_faces > 1:
                box_color = (0, 0, 255) # Red if multi-face cheating detected
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, f"Face {i+1}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        cv2.imshow("Frontend Simulation", frame)
        cv2.setMouseCallback("Frontend Simulation", on_mouse_click)
        
        if cv2.waitKey(1) == ord('q') or stop_button_pressed[0]: 
            if stop_button_pressed[0]:
                print("Stop button clicked.")
            break

    total_time = time.time() - start_time
    print(f"Session ended. Total time: {total_time:.2f}s")

    detector.close()
    if gaze_detector:
        gaze_detector.close()
    video.release()
    cv2.destroyAllWindows()
