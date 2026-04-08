import cv2
import numpy as np

def record_cartoon_webcam(output_filename="my_cartoon_video.mp4"):
    """
    Captures video from the webcam, applies a real-time cartoon effect, 
    and saves the result to an MP4 file.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # ---------------------------------------------------------
    # NEW: Setup the VideoWriter to save the video
    # ---------------------------------------------------------
    # 1. Get the default width and height of the webcam's frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Define the codec using VideoWriter_fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # 3. Create the VideoWriter object (filename, codec, frames per second, resolution)
    # Note: 20.0 FPS is a good baseline, but you can tweak it based on your camera
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

    print(f"Starting webcam and recording to '{output_filename}'...")
    print("Press 'q' to quit and save the video.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        frame = cv2.flip(frame, 1)

        # 1. Edge Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )

        # 2. Color Simplification (Pyramid down -> Bilateral -> Pyramid up)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        for _ in range(5): 
            small_frame = cv2.bilateralFilter(small_frame, d=9, sigmaColor=9, sigmaSpace=7)
        color = cv2.resize(small_frame, (frame.shape[1], frame.shape[0]))

        # 3. Final Composite
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        # ---------------------------------------------------------
        # NEW: Write the processed frame to our video file
        # ---------------------------------------------------------
        out.write(cartoon)

        # Display the result on screen
        cv2.imshow("Recording Cartoonify...", cartoon)

        # Listen for the 'q' key to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and finalize the video file
    cap.release()
    out.release() # IMPORTANT: This saves and closes the MP4 file!
    cv2.destroyAllWindows()
    print("Webcam closed. Video successfully saved!")

# ==========================================
# Run the function
# ==========================================
if __name__ == "__main__":
    record_cartoon_webcam()