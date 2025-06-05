import sys
import numpy as np
import cv2
import time

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as tflite

MODEL_PATH = "hand_landmark_lite.tflite"

def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return interpreter, inp, out

def preprocess(frame, inp):
    h, w = inp['shape'][1], inp['shape'][2]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, 0).astype(inp['dtype'])

def run_pipe(width=640, height=480):
    interp, inp, out = load_interpreter(MODEL_PATH)
    frame_size = width * height * 3  # BGR24: 3 bytes per pixel
    print(f"[DEBUG] Expecting frames of size: {frame_size} bytes ({width}x{height})")
    frame_count = 0
    start_time = time.time()

    while True:
        raw = sys.stdin.buffer.read(frame_size)
        if not raw or len(raw) != frame_size:
            print(f"[ERROR] Failed to read frame ({len(raw) if raw else 0} bytes). Exiting.")
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        frame = frame.copy()  # ← добави този ред!

        t0 = time.time()
        data = preprocess(frame, inp)
        interp.set_tensor(inp['index'], data)
        interp.invoke()
        lm = interp.get_tensor(out['index'])[0]
        inf_ms = (time.time() - t0) * 1000

        x_norm, y_norm = lm[0], lm[1]
        px, py = int(x_norm * width), int(y_norm * height)

        cv2.circle(frame, (px, py), 7, (0, 255, 0), -1)
        cv2.putText(frame, f"Wrist: ({px},{py})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inf_ms:.1f} ms", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Hand Landmark Lite (libcamera)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("[DEBUG] ESC pressed, exiting.")
            break


    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipe(640, 480)
