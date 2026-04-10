import cv2
import numpy as np
import argparse
import time
import os

def quantize_color(img, k):
    if k < 2:
        return img.copy()
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.shape)
    return quantized

def cartoonify_frame(frame, d, sigma_color, sigma_space, k, block_size, C, use_quant):
    if frame is None:
        return None
    # 1. Color Layer
    color = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    if use_quant:
        color = quantize_color(color, k)
    # 2. Edge Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  block_size, C)
    # 3. Merge (as per project overview)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def auto_suggest_k(img, max_k=20):
    unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
    k = max(4, min(max_k, unique_colors // 800))
    return k

def main():
    parser = argparse.ArgumentParser(description="SmartCartoonify - Novel CV Project")
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'webcam', 'video'],
                        help='image / webcam / video')
    parser.add_argument('--input', type=str, default='input.jpg',
                        help='Path to image or video file')
    args = parser.parse_args()

    cv2.namedWindow("SmartCartoonify Controls", cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar("Bilateral d", "SmartCartoonify Controls", 9, 15, lambda x: None)
    cv2.createTrackbar("Sigma Color", "SmartCartoonify Controls", 75, 300, lambda x: None)
    cv2.createTrackbar("Sigma Space", "SmartCartoonify Controls", 75, 300, lambda x: None)
    cv2.createTrackbar("K Colors", "SmartCartoonify Controls", 8, 32, lambda x: None)
    cv2.createTrackbar("Block Size", "SmartCartoonify Controls", 9, 15, lambda x: None)
    cv2.createTrackbar("C", "SmartCartoonify Controls", 2, 10, lambda x: None)
    cv2.createTrackbar("Quantize (0/1)", "SmartCartoonify Controls", 1, 1, lambda x: None)

    if args.mode == 'image':
        # ... (same as before - unchanged)
        original = cv2.imread(args.input)
        if original is None:
            print("❌ Error: Cannot load image")
            return
        suggested_k = auto_suggest_k(original)
        cv2.setTrackbarPos("K Colors", "SmartCartoonify Controls", suggested_k)
        print(f"🔥 Auto-suggested K = {suggested_k}")

        while True:
            d = cv2.getTrackbarPos("Bilateral d", "SmartCartoonify Controls")
            sc = cv2.getTrackbarPos("Sigma Color", "SmartCartoonify Controls")
            ss = cv2.getTrackbarPos("Sigma Space", "SmartCartoonify Controls")
            k = cv2.getTrackbarPos("K Colors", "SmartCartoonify Controls")
            bs = cv2.getTrackbarPos("Block Size", "SmartCartoonify Controls")
            c_val = cv2.getTrackbarPos("C", "SmartCartoonify Controls")
            quant = bool(cv2.getTrackbarPos("Quantize (0/1)", "SmartCartoonify Controls"))
            if bs % 2 == 0: bs += 1; cv2.setTrackbarPos("Block Size", "SmartCartoonify Controls", bs)

            cartoon = cartoonify_frame(original.copy(), d, sc, ss, k, bs, c_val, quant)
            h, w = original.shape[:2]
            combined = np.hstack((cv2.resize(original, (w, h)), cv2.resize(cartoon, (w, h))))
            cv2.putText(combined, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Cartoon", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("SmartCartoonify - Original | Cartoon", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                filename = f"cartoon_{int(time.time())}.jpg"
                cv2.imwrite(filename, cartoon)
                print(f"✅ Saved: {filename}")
            elif key == ord('q'):
                break

    elif args.mode == 'webcam':
        # ... (same as before - unchanged)
        cap = cv2.VideoCapture(0)
        print("🎥 Live Webcam Mode - 's'=save frame, 'q'=quit")
        while True:
            ret, frame = cap.read()
            if not ret: break
            # ... (same trackbar + cartoonify logic as before)
            d = cv2.getTrackbarPos("Bilateral d", "SmartCartoonify Controls")
            sc = cv2.getTrackbarPos("Sigma Color", "SmartCartoonify Controls")
            ss = cv2.getTrackbarPos("Sigma Space", "SmartCartoonify Controls")
            k = cv2.getTrackbarPos("K Colors", "SmartCartoonify Controls")
            bs = cv2.getTrackbarPos("Block Size", "SmartCartoonify Controls")
            c_val = cv2.getTrackbarPos("C", "SmartCartoonify Controls")
            quant = bool(cv2.getTrackbarPos("Quantize (0/1)", "SmartCartoonify Controls"))
            if bs % 2 == 0: bs += 1

            cartoon = cartoonify_frame(frame, d, sc, ss, k, bs, c_val, quant)
            combined = np.hstack((cv2.resize(frame, (640, 480)), cv2.resize(cartoon, (640, 480))))
            cv2.imshow("Live Cartoonify - Original | Cartoon", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                filename = f"live_cartoon_{int(time.time())}.jpg"
                cv2.imwrite(filename, cartoon)
                print(f"✅ Saved: {filename}")
            elif key == ord('q'):
                break
        cap.release()

    elif args.mode == 'video':
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print("❌ Error: Cannot open video file")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"🎬 Video Mode Loaded: {args.input} | {width}x{height} @ {fps}fps")
        print("   's' = save current frame | 'r' = start/stop recording full cartoon video | 'q' = quit")

        out = None
        recording = False
        output_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("✅ End of video reached.")
                break

            d = cv2.getTrackbarPos("Bilateral d", "SmartCartoonify Controls")
            sc = cv2.getTrackbarPos("Sigma Color", "SmartCartoonify Controls")
            ss = cv2.getTrackbarPos("Sigma Space", "SmartCartoonify Controls")
            k = cv2.getTrackbarPos("K Colors", "SmartCartoonify Controls")
            bs = cv2.getTrackbarPos("Block Size", "SmartCartoonify Controls")
            c_val = cv2.getTrackbarPos("C", "SmartCartoonify Controls")
            quant = bool(cv2.getTrackbarPos("Quantize (0/1)", "SmartCartoonify Controls"))
            if bs % 2 == 0:
                bs += 1
                cv2.setTrackbarPos("Block Size", "SmartCartoonify Controls", bs)

            cartoon = cartoonify_frame(frame, d, sc, ss, k, bs, c_val, quant)

            # Side-by-side live preview
            combined = np.hstack((cv2.resize(frame, (640, 480)), cv2.resize(cartoon, (640, 480))))
            cv2.imshow("SmartCartoonify Video - Original | Cartoon", combined)

            # Recording logic
            if recording and out is not None:
                out.write(cartoon)

            key = cv2.waitKey(25) & 0xFF   # 25ms ≈ 40 fps preview speed

            if key == ord('s'):
                filename = f"video_cartoon_{int(time.time())}.jpg"
                cv2.imwrite(filename, cartoon)
                print(f"✅ Saved frame: {filename}")

            elif key == ord('r'):
                if not recording:
                    output_path = f"cartoon_output_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    recording = True
                    print(f"🔴 Recording started → {output_path}")
                else:
                    out.release()
                    out = None
                    recording = False
                    print(f"🟢 Recording saved: {output_path}")

            elif key == ord('q'):
                break

        cap.release()
        if out is not None:
            out.release()
        print("✅ Video mode closed.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()