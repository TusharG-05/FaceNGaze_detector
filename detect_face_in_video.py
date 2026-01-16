import cv2
import time
import numpy as np
import multiprocessing
import os
import face_recognition

# =============================
# WORKER PROCESS
# =============================
def face_recognition_worker(frame_queue, result_queue, known_encoding):
    """
    Runs in a separate process.
    Constantly pulls frames from the queue and performs face recognition.
    """
    while True:
        try:
            # Get frame (blocking with timeout to allow clean exit)
            rgb_small_frame = frame_queue.get(timeout=1)
        except:
            continue

        if rgb_small_frame is None:
            break # Sentinel to stop

        # Perform the heavy lifting
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame)
            
            match_found = False
            min_dist = 1.0
            
            if len(face_encodings) > 0:
                # Compare detected faces to known face
                # face_distance returns an array of distances
                distances = face_recognition.face_distance([known_encoding], face_encodings[0])
                min_dist = distances[0]
                if min_dist <= 0.5: # Threshold
                    match_found = True
            
            # Send result back
            # Format: (found_boolean, distance_float)
            if not result_queue.full():
                result_queue.put((match_found, min_dist))
                
        except Exception as e:
            print(f"Worker Error: {e}")

# =============================
# FACE DETECTOR CLASS (API READY)
# =============================
class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        """
        Initialize the separate process and load the known face.
        """
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
        """
        Accepts a BGR frame (from OpenCV/Camera), sends it to the worker,
        and returns the LATEST known result (non-blocking).
        Returns: (is_found, distance) or (None, None) if no new result.
        """
        # 1. Resize & Convert to RGB (Optimization)
        # We do this here to lighten the load on the IPC (Inter-Process Communication)
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # 2. Put in Queue (Non-blocking drop)
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small_frame)

        # 3. Check for Result (Non-blocking peek)
        try:
            return self.result_queue.get_nowait()
        except:
            return None, None # No new result ready yet

    def close(self):
        """Cleanup resources"""
        self.worker.terminate()
        self.worker.join()

# =============================
# MOCK FRONTEND (Simulating a Server)
# =============================
if __name__ == "__main__":
    # This block simulates what your "FastAPI" or "Flask" app would do.
    # It opens the camera locally to test the Class.
    
    detector = FaceDetector()
    
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

    while True:
        ret, frame = video.read()
        if not ret: break

        # === THE API CALL ===
        # In a real app, 'frame' comes from the webpage.
        result = detector.process_frame(frame) 
        
        # Update State if we got a fresh result
        found, dist = result
        if found is not None:
            last_found = found
            last_dist = dist
            if found:
                print(f"API Result: Found! ({dist:.2f})")

        # Display (Only for testing)
        color = (0, 255, 0) if last_found else (0, 0, 255)
        cv2.putText(frame, f"Found: {last_found} ({last_dist:.2f})", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Frontend Simulation", frame)
        if cv2.waitKey(1) == ord('q'): break

    total_time = time.time() - start_time
    print(f"Session ended. Total time: {total_time:.2f}s")

    detector.close()
    video.release()
    cv2.destroyAllWindows()
