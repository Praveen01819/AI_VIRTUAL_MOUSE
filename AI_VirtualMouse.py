import cv2
import mediapipe as mp
import pyautogui
import math
import time  

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class Gest:
    PALM = 31  # Stop cursor when palm is detected
    V_GEST = 33  # Cursor movement
    PINCH = 35  # Left-click
    RIGHT_CLICK = 36  # Right-click
    DOUBLE_CLICK = 37  # Double-click
    FIST = 0  # Stop cursor when fist is detected
    PINCH_SCROLL_LEFT = 44
    PINCH_SCROLL_RIGHT = 45
    PINCH_SCROLL_UP = 46
    PINCH_SCROLL_DOWN = 47
    DRAG = 50

class Controller:
    prev_x, prev_y = 0, 0  # Store previous cursor position
    last_click_time = 0  # Store time of last index-middle finger touch
    last_left_click_time = 0  # Store time of last left click to prevent false double clicks
    dragging = False

    @staticmethod
    def get_position(hand_result):
        """Gets the position of the index finger base for cursor movement."""
        point = 9  
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x = int(position[0] * sx)
        y = int(position[1] * sy)
        return x, y

    @staticmethod
    def smooth_cursor(x, y, alpha=0.2):
        """Smooth cursor movement using an exponential moving average."""
        Controller.prev_x = int(Controller.prev_x * (1 - alpha) + x * alpha)
        Controller.prev_y = int(Controller.prev_y * (1 - alpha) + y * alpha)
        return Controller.prev_x, Controller.prev_y

    @staticmethod
    def handle_controls(gesture, hand_result):
        """Handles different gestures and executes corresponding mouse actions."""
        #if gesture == Gest.PALM or gesture == Gest.FIST:
            #return  # Stop cursor movement when palm or fist is detected
        
        if gesture == Gest.V_GEST:
            x, y = Controller.get_position(hand_result)
            smooth_x, smooth_y = Controller.smooth_cursor(x, y)
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

        elif gesture == Gest.PINCH and not Controller.dragging:
            current_time = time.time()
            if current_time - Controller.last_left_click_time > 0.3:
                pyautogui.click()
                Controller.last_left_click_time = current_time

        elif gesture == Gest.RIGHT_CLICK:
            pyautogui.click(button="right")  # Right-click
        
        elif gesture == Gest.DOUBLE_CLICK:
            current_time = time.time()
            # Detect double tap (index-middle touch two times within 0.5 seconds)
            if Controller.last_click_time and (current_time - Controller.last_click_time < 0.4):
                pyautogui.doubleClick()  # Perform double-click
                Controller.last_click_time = 0
            else:
                Controller.last_click_time = current_time  # Update last click time

        elif gesture == Gest.PINCH_SCROLL_LEFT:
            pyautogui.hscroll(-30)  # Scroll left
        elif gesture == Gest.PINCH_SCROLL_RIGHT:
            pyautogui.hscroll(30)  # Scroll right
        elif gesture == Gest.PINCH_SCROLL_UP:
            pyautogui.scroll(30)  # Scroll up
        elif gesture == Gest.PINCH_SCROLL_DOWN:
            pyautogui.scroll(-30)  # Scroll down
        
        elif gesture == Gest.DRAG:
            x, y = Controller.get_position(hand_result)
            smooth_x, smooth_y = Controller.smooth_cursor(x, y)

            if not Controller.dragging:
                Controller.dragging = True
                pyautogui.mouseDown()  # Start dragging

                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

# Stop drag when palm is detected
        elif gesture == Gest.PALM or gesture == Gest.FIST:
            if Controller.dragging:
                Controller.dragging = False
                pyautogui.mouseUp()  # Stop dragging

class HandRecog:
    def __init__(self):
        """Initializes the Hand Recognition system."""
        self.hand_result = None
        self.drag_started = False 

    def update_hand_result(self, hand_result):
        """Updates the detected hand landmarks."""
        self.hand_result = hand_result

    def get_gesture(self):
        """Identifies the hand gesture based on finger positions."""
        if self.hand_result is None:
            return Gest.PALM  # If no hand detected, assume palm is open

        # Get landmarks for fingertips
        thumb_tip = self.hand_result.landmark[4]
        index_tip = self.hand_result.landmark[8]
        middle_tip = self.hand_result.landmark[12]
        ring_tip = self.hand_result.landmark[16]
        pinky_tip = self.hand_result.landmark[20]

        # Get landmarks for finger bases
        index_base = self.hand_result.landmark[5]
        middle_base = self.hand_result.landmark[9]
        ring_base = self.hand_result.landmark[13]
        pinky_base = self.hand_result.landmark[17]

        # **1️⃣ Palm Detection**
        open_fingers = sum([
            math.dist([index_tip.x, index_tip.y], [index_base.x, index_base.y]) > 0.12,
            math.dist([middle_tip.x, middle_tip.y], [middle_base.x, middle_base.y]) > 0.12,
            math.dist([ring_tip.x, ring_tip.y], [ring_base.x, ring_base.y]) > 0.12,
            math.dist([pinky_tip.x, pinky_tip.y], [pinky_base.x, pinky_base.y]) > 0.12
        ])
        if open_fingers == 4:
            self.drag_started = False
            return Gest.PALM  

        # **2️⃣ Fist Detection**
        closed_fingers = sum([
            math.dist([index_tip.x, index_tip.y], [index_base.x, index_base.y]) < 0.04,
            math.dist([middle_tip.x, middle_tip.y], [middle_base.x, middle_base.y]) < 0.04,
            math.dist([ring_tip.x, ring_tip.y], [ring_base.x, ring_base.y]) < 0.04,
            math.dist([pinky_tip.x, pinky_tip.y], [pinky_base.x, pinky_base.y]) < 0.04
        ])
        if closed_fingers == 4 and math.dist([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y]) > 0.01:
            self.drag_started = True
            pyautogui.mouseDown()  # Start dragging
            return Gest.DRAG  # Return immediately, preventing double-click misdetection



        #Left Click Detection
        pinch_distance_left = math.dist([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y])
        if pinch_distance_left < 0.05:
            return Gest.PINCH  # Left-click

        #Right Click Detection
        pinch_distance_right = math.dist([thumb_tip.x, thumb_tip.y], [middle_tip.x, middle_tip.y])
        if pinch_distance_right < 0.03:
            return Gest.RIGHT_CLICK  # Right-click
        
        # Double Click Detection
        distance_double = math.dist([index_tip.x, index_tip.y], [middle_tip.x, middle_tip.y])
        current_time = time.time()
        if distance_double < 0.03:
            if not Controller.last_click_time:
                Controller.last_click_time = current_time
            elif (current_time - Controller.last_click_time) < 0.3:
                Controller.last_click_time = 0
                return Gest.DOUBLE_CLICK  # NEW: Double Click
        else:
            Controller.last_click_time = 0
        
                # Pinch detection for scrolling
        pinch_distance = math.dist([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y])

        # If thumb and index finger are pinched together (close enough)
        if pinch_distance < 0.05:
            # Compare x-coordinates for left/right scroll
            if index_tip.x < index_base.x - 0.05:  # Move left
                return Gest.PINCH_SCROLL_LEFT
            elif index_tip.x > index_base.x + 0.05:  # Move right
                return Gest.PINCH_SCROLL_RIGHT


        return Gest.V_GEST  # Default cursor movement

class GestureController:
    def __init__(self):
        """Initializes the Gesture Controller."""
        self.cap = cv2.VideoCapture(0)
        self.hand_recog = HandRecog()

    def start(self):
        """Starts the AI Virtual Mouse system."""
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.hand_recog.update_hand_result(hand_landmarks)
                        gesture = self.hand_recog.get_gesture()
                        Controller.handle_controls(gesture, hand_landmarks)
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('AI Virtual Mouse', image)
                if cv2.waitKey(5) & 0xFF == 13:
                    break

        self.cap.release()
        cv2.destroyAllWindows()

# Run the AI Virtual Mouse system
gc = GestureController()
gc.start()