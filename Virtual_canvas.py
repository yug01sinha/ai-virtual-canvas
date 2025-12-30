import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- 1. CONFIGURATION ---
# Define DESIRED size, but the code will use the ACTUAL size the camera provides
DESIRED_WIDTH = 1000
DESIRED_HEIGHT = 700 
# Threshold for considering the index and middle fingers as "pinched" (in pixels)
PINCH_DISTANCE_THRESHOLD = 50 

# Brush settings
BRUSH_COLOR = (0, 255, 0)      # Green for Drawing
ERASER_COLOR = (0, 0, 0)       # Black (matches canvas background)
BRUSH_SIZE = 8
ERASER_SIZE = 40

# MediaPipe Hand Model Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# --- 2. INITIAL SETUP ---

def run_virtual_canvas():
    
    cap = cv2.VideoCapture(0)
    
    # 1. Try to set the size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 2. READ the ACTUAL dimensions the camera successfully opened at
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera successfully opened at actual resolution: {actual_width}x{actual_height}")

    # Initialize the canvas using the ACTUAL camera dimensions! (Fixes the -209 error)
    canvas = np.zeros((actual_height, actual_width, 3), dtype=np.uint8) 

    # Track the previous position of the index finger tip for smooth lines
    prev_index_pos = None
    
    print("\nVirtual Paintbrush Running. Use Index Finger to draw. Pinch Index & Middle finger to erase. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a natural view
        frame = cv2.flip(frame, 1)
        
        # Prepare image for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        
        mode = "IDLE" 

        # --- Drawing and Mode Logic ---
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Get the coordinates for the Index Finger Tip (Landmark 8) 
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # And Middle Finger Tip (Landmark 12)
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # 3. Convert normalized coordinates (0 to 1) to pixel coordinates using the ACTUAL dimensions
                index_x = int(index_tip.x * actual_width)
                index_y = int(index_tip.y * actual_height)
                middle_x = int(middle_tip.x * actual_width)
                middle_y = int(middle_tip.y * actual_height)
                
                current_index_pos = (index_x, index_y)

                # 4. Calculate the distance between the two tips
                distance = math.hypot(index_x - middle_x, index_y - middle_y)
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 5. Determine Mode (Erase vs. Draw)
                if distance < PINCH_DISTANCE_THRESHOLD:
                    mode = "ERASE"
                    brush_color = ERASER_COLOR
                    brush_size = ERASER_SIZE
                    cv2.circle(frame, current_index_pos, brush_size, (0, 255, 255), 2)
                else:
                    mode = "DRAW"
                    brush_color = BRUSH_COLOR
                    brush_size = BRUSH_SIZE
                    cv2.circle(frame, current_index_pos, brush_size, (0, 255, 0), 2)
                    
                # 6. Draw on the Canvas
                if prev_index_pos is not None:
                    cv2.line(canvas, prev_index_pos, current_index_pos, brush_color, brush_size)
                
                prev_index_pos = current_index_pos
        else:
            prev_index_pos = None

        # --- Display and Overlay ---
        
        # This line will now work because 'frame' and 'canvas' have identical dimensions.
        frame = cv2.addWeighted(frame, 1, canvas, 0.8, 0)
        
        # Display the current mode
        cv2.putText(frame, f"Mode: {mode} (C=Clear)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Virtual Canvas', frame)
        
        # Exit loop if 'q' is pressed or clear canvas if 'c' is pressed
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas[:] = 0 

    # --- 4. CLEANUP ---
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_virtual_canvas()