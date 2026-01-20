import cv2
import time
import numpy as np
import multiprocessing
import os
import face_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import threading

def face_recognition_worker(frame_queue, result_queue, known_encoding):
    """
    Multithreaded worker: 
    - Main Loop: Fast Detection (30+ FPS)
    - Background Thread: Slow Recognition (No blocking)
    """
    state = {
        'img': None,
        'locs': [],
        'match': False,
        'conf': 1.0,
        'last_recog_time': 0,
        'lock': threading.Lock()
    }

    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path='app/assets/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=4,
        min_face_detection_confidence=0.5
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    def recognition_loop():
        while True:
            time.sleep(0.01)
            with state['lock']:
                img_copy = state['img'].copy() if state['img'] is not None else None
                locs_copy = list(state['locs'])
            
            if img_copy is not None and locs_copy:
                try:
                    # Slow encoding happens here
                    encs = face_recognition.face_encodings(img_copy, locs_copy, num_jitters=0)
                    if encs and known_encoding is not None:
                        dists = face_recognition.face_distance(encs, known_encoding)
                        match = any(d <= 0.52 for d in dists)
                        conf = float(min(dists))
                        with state['lock']:
                            state['match'] = match
                            state['conf'] = conf
                except:
                    pass
            time.sleep(0.5) # Throttle recognition thread

    recog_thread = threading.Thread(target=recognition_loop, daemon=True)
    recog_thread.start()

    while True:
        try:
            img = frame_queue.get(timeout=1)
            if img is None: break
            
            ih, iw = img.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            results = detector.detect(mp_image)
            
            new_locs = []
            if results.face_landmarks:
                for landmarks in results.face_landmarks:
                    xs, ys = [lm.x for lm in landmarks], [lm.y for lm in landmarks]
                    t, l, b, r = int(min(ys)*ih), int(min(xs)*iw), int(max(ys)*ih), int(max(xs)*iw)
                    hp, wp = int((b-t)*0.1), int((r-l)*0.1)
                    new_locs.append((max(0, t-hp), min(iw, r+wp), min(ih, b+hp), max(0, l-wp)))

            with state['lock']:
                state['img'] = img
                state['locs'] = new_locs
                match_val, conf_val = state['match'], state['conf']

            if not result_queue.full():
                result_queue.put((match_val, conf_val, len(new_locs), new_locs))
        except Exception as e:
            print(f"Face Worker Error: {e}")

class FaceDetector:
    def __init__(self, known_person_path="known_person.jpg"):
        print("Starting Zero-Lag Face Service...")
        try:
            known_image = face_recognition.load_image_file(known_person_path)
            encs = face_recognition.face_encodings(known_image)
            self.known_encoding = encs[0] if encs else None
        except:
            self.known_encoding = None

        self.frame_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue(maxsize=1)
        self.worker = multiprocessing.Process(
            target=face_recognition_worker, 
            args=(self.frame_queue, self.result_queue, self.known_encoding)
        )
        self.worker.daemon = True
        self.worker.start()

    def process_frame(self, frame_bgr):
        # RESIZE IN MAIN THREAD - Crucial for zero delay
        h, w = frame_bgr.shape[:2]
        s = 360.0 / h if h > 360 else 1.0
        
        # Convert to RGB and Resize
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (0,0), fx=s, fy=s) if s < 1.0 else img_rgb

        if not self.frame_queue.full():
            self.frame_queue.put(img_small)
        
        try:
            match, conf, n_faces, locs = self.result_queue.get_nowait()
            # Scale coordinates back up
            scaled_locs = [(int(t/s), int(r/s), int(b/s), int(l/s)) for (t,r,b,l) in locs]
            return match, conf, n_faces, scaled_locs
        except:
            return None, None, 0, []

    def close(self):
        try:
            self.frame_queue.put(None)
            self.worker.join()
        except:
            self.worker.terminate()


