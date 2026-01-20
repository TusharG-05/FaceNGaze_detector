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
            # Face Recognition Logic
            # Upsample 2x to find smaller faces (High Sensitivity)
            face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            num_faces = len(face_encodings)
            
            match_found = False
            min_dist = 1.0
            
            if num_faces > 0:
                # We check the first face found against our known encoding
                distances = face_recognition.face_distance([known_encoding], face_encodings[0])
                min_dist = distances[0]
                if min_dist <= 0.5: # Strict threshold for a match
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
            
        # Load Reference Image
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

        # 2. Send to Worker (Non-blocking)
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small_frame)

        # 3. Retrieve Latest Result (Non-blocking)
        try:
            return self.result_queue.get_nowait()
        except:
            # Return defaults/empty if processing isn't done yet
            return None, None, 0, []

    def close(self):
        """Clean up background resources."""
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()


