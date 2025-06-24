import tkinter as tk
import customtkinter as ctk
import threading
import cv2
import time
import numpy as np
import HandTrackingModule as htm
from PIL import Image, ImageTk

# CustomTkinter Configuration
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class VideoCaptureAndProcessingThread(threading.Thread):
    def __init__(self, finger_display_var, detector_confidence=0.75):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        self.detector = htm.handDetector(detectionCon=detector_confidence)
        self.finger_display_var = finger_display_var
        self.running = True
        self.current_frame = None

    def run(self):
        pTime = 0
        while self.running:
            success, img = self.cap.read()
            if not success:
                print("Failed to read from camera in thread. Exiting.")
                self.running = False
                break

            img = cv2.flip(img, 1)

            img = self.detector.findHands(img)
            lmList, bbox = self.detector.findPosition(img, draw=False)
            
            # Create a gesture state string to pass to the UI
            # Format: "index_middle_pinky_thumb" where each can be 0 or 1
            gesture_state = "0_0_0_0"

            if lmList:
                fingersUp = self.detector.fingersUp()
                
                # Check individual fingers
                index_up = fingersUp[1] == 1  # Index finger
                middle_up = fingersUp[2] == 1  # Middle finger
                pinky_up = fingersUp[4] == 1  # Pinky finger  
                thumb_up = fingersUp[0] == 1  # Thumb
                
                # Create gesture state string
                gesture_state = f"{int(index_up)}_{int(middle_up)}_{int(pinky_up)}_{int(thumb_up)}"
            
            self.finger_display_var.set(gesture_state)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (500, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

            self.current_frame = img

        self.cap.release()
        print("Video capture thread stopped.")

    def get_current_frame(self):
        return self.current_frame

    def stop(self):
        self.running = False

class AlphaNumericKeyboardApp(ctk.CTkFrame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("🖐️ Hand Controlled Keyboard")
        self.master.geometry("1400x800")  # Increased window size for better layout
        self.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights for responsive layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.finger_recognized_var = tk.StringVar(self.master, value="0_0_0_0")
        self.video_thread = None
        self.current_photo_image = None
        
        # --- Keyboard State Variables ---
        self.input_text = tk.StringVar(self.master, value="")
        self.selected_column = -1
        self.selected_row = -1
        
        # --- New gesture state tracking ---
        self._last_gesture_state = "0_0_0_0"
        self._index_column_counter = 0
        self._pinky_row_counter = 0

        self.create_widgets()
        self.start_video_processing()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # --- Left Panel: Video Feed ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Video title
        video_title = ctk.CTkLabel(self.video_frame, text="📹 Live Camera Feed", 
                                  font=ctk.CTkFont(size=18, weight="bold"))
        video_title.pack(pady=(20, 10))
        
        self.video_label = tk.Label(self.video_frame, bg="#212121")
        self.video_label.pack(padx=20, pady=(0, 20), fill=tk.BOTH, expand=True)

        # --- Right Panel: Keyboard Control ---
        self.control_panel = ctk.CTkFrame(self, corner_radius=15)
        self.control_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        # Configure control panel grid
        self.control_panel.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(self.control_panel, text="🎮 Gesture Control Panel", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")
        
        # --- Gesture State Display ---
        self.gesture_frame = ctk.CTkFrame(self.control_panel, corner_radius=10)
        self.gesture_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        gesture_title = ctk.CTkLabel(self.gesture_frame, text="👆 Gesture State (Index_Middle_Pinky_Thumb)", 
                                    font=ctk.CTkFont(size=14, weight="bold"))
        gesture_title.pack(pady=(15, 5))
        
        self.gesture_display = ctk.CTkLabel(self.gesture_frame, textvariable=self.finger_recognized_var,
                                           font=ctk.CTkFont(size=24, weight="bold"),
                                           text_color="#1f538d")
        self.gesture_display.pack(pady=(0, 15))
        
        # Status label
        self.action_status_label = ctk.CTkLabel(self.control_panel, text="🎯 Last Action: None", 
                                               font=ctk.CTkFont(size=14))
        self.action_status_label.grid(row=2, column=0, pady=10, sticky="ew")

        # --- Selection Status ---
        self.status_frame = ctk.CTkFrame(self.control_panel, corner_radius=10)
        self.status_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        status_title = ctk.CTkLabel(self.status_frame, text="📍 Selection Status", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        status_title.pack(pady=(15, 10))
        
        # Status labels in a horizontal frame
        status_container = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        status_container.pack(pady=(0, 15))
        
        self.column_status_label = ctk.CTkLabel(status_container, text="📋 Column: None", 
                                               font=ctk.CTkFont(size=12),
                                               text_color="#1f538d")
        self.column_status_label.pack(side=tk.LEFT, padx=10)
        
        self.row_status_label = ctk.CTkLabel(status_container, text="📊 Row: None", 
                                            font=ctk.CTkFont(size=12),
                                            text_color="#1f8d3a")
        self.row_status_label.pack(side=tk.LEFT, padx=10)
        
        # --- Text Input ---
        self.input_frame = ctk.CTkFrame(self.control_panel, corner_radius=10)
        self.input_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        input_title = ctk.CTkLabel(self.input_frame, text="✏️ Text Input", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        input_title.pack(pady=(15, 10))
        
        self.text_entry = ctk.CTkEntry(self.input_frame, textvariable=self.input_text,
                                      font=ctk.CTkFont(size=18),
                                      placeholder_text="Type with gestures...",
                                      height=40)
        self.text_entry.pack(padx=20, pady=(0, 15), fill=tk.X)
        
        # --- Keyboard Layout ---
        self.keyboard_frame = ctk.CTkFrame(self.control_panel, corner_radius=10)
        self.keyboard_frame.grid(row=5, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        keyboard_title = ctk.CTkLabel(self.keyboard_frame, text="⌨️ Virtual Keyboard", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        keyboard_title.pack(pady=(15, 10))
        
        # Keyboard buttons container
        self.keyboard_buttons_frame = ctk.CTkFrame(self.keyboard_frame, fg_color="transparent")
        self.keyboard_buttons_frame.pack(pady=(0, 15), padx=20)
        
        # Symmetric 3x9 layout with only letters (27 letters total)
        self.keys_layout = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'], 
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'P', '']
        ]

        self.key_buttons = {}  # Store button references for highlighting

        for r_idx, row_keys in enumerate(self.keys_layout):
            for c_idx, key_char in enumerate(row_keys):
                # Skip empty keys
                if key_char == '':
                    continue
                btn = ctk.CTkButton(self.keyboard_buttons_frame, text=key_char, 
                                   font=ctk.CTkFont(size=14, weight="bold"),
                                   width=45, height=35,
                                   command=lambda char=key_char: self.type_character(char))
                btn.grid(row=r_idx, column=c_idx, padx=2, pady=2)
                self.key_buttons[(r_idx, c_idx)] = btn  # Store button by its grid position
        
        # --- Schedule gesture checking ---
        self.master.after(100, self.check_and_trigger_keyboard_action)

    def start_video_processing(self):
        if self.video_thread is None or not self.video_thread.is_alive():
            self.video_thread = VideoCaptureAndProcessingThread(self.finger_recognized_var)
            self.video_thread.daemon = True
            self.video_thread.start()
            self.update_video_feed()

    def update_video_feed(self):
        frame = self.video_thread.get_current_frame()
        if frame is not None:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            pil_img = Image.fromarray(cv2image)
            self.current_photo_image = ImageTk.PhotoImage(image=pil_img)
            self.video_label.config(image=self.current_photo_image)
            self.video_label.image = self.current_photo_image
        
        self.master.after(10, self.update_video_feed)

    # --- Core Keyboard Interaction Methods ---
    def type_character(self, char):
        current_text = self.input_text.get()
        self.input_text.set(current_text + char)
        print(f"Typed: {char}")

    def clear_selection(self):
        """Clear all selections"""
        self.selected_column = -1
        self.selected_row = -1
        self._index_column_counter = 0
        self._pinky_row_counter = 0
        self.highlight_selection()
        self.update_status_labels()

    def update_column_selection(self):
        """Update column selection based on index finger"""
        self.selected_column = self._index_column_counter % len(self.keys_layout[0])
        print(f"Column selected: {self.selected_column}")
        self.highlight_selection()
        self.update_status_labels()

    def update_row_selection(self):
        """Update row selection based on pinky finger"""
        self.selected_row = self._pinky_row_counter % len(self.keys_layout)
        print(f"Row selected: {self.selected_row}")
        self.highlight_selection()
        self.update_status_labels()

    def update_status_labels(self):
        """Update the status labels to show current selection"""
        col_text = f"📋 Column: {self.selected_column}" if self.selected_column != -1 else "📋 Column: None"
        row_text = f"📊 Row: {self.selected_row}" if self.selected_row != -1 else "📊 Row: None"
        
        self.column_status_label.configure(text=col_text)
        self.row_status_label.configure(text=row_text)

    def perform_backspace(self):
        """Remove last character from input"""
        current_text = self.input_text.get()
        if current_text:
            self.input_text.set(current_text[:-1])
        print("Backspace performed")

    def perform_select_and_enter(self):
        """Enter the selected character if both row and column are selected"""
        if self.selected_row != -1 and self.selected_column != -1:
            try:
                # Ensure the selected row/column is within the bounds of the keys_layout
                if (0 <= self.selected_row < len(self.keys_layout) and 
                    0 <= self.selected_column < len(self.keys_layout[self.selected_row])):
                    char_to_type = self.keys_layout[self.selected_row][self.selected_column]
                    self.type_character(char_to_type)
                    print(f"ENTERED: {char_to_type}")
                    
                    # Clear selection after successful entry
                    self.clear_selection()
                else:
                    print("Error: Selection out of bounds.")
            except IndexError:
                print("Error: Invalid selection coordinates (IndexError).")
        else:
            print("No complete selection (row and column) to enter.")

    def highlight_selection(self):
        """Highlights the currently selected row/column/cell using CustomTkinter color system."""
        # CustomTkinter color scheme
        default_color = ["#3B8ED0", "#1F6AA5"]  # Default blue colors
        column_highlight_color = ["#36719F", "#144870"]  # Light blue for columns
        row_highlight_color = ["#3E8B3E", "#2E6B2E"]  # Green for rows
        selected_cell_color = ["#FA7970", "#C93025"]  # Red for selected cell

        # Reset all buttons to default color
        for (r_idx, c_idx), btn in self.key_buttons.items():
            btn.configure(fg_color=default_color)

        # Highlight columns
        if self.selected_column != -1:
            for r_idx in range(len(self.keys_layout)):
                if (r_idx, self.selected_column) in self.key_buttons:
                    self.key_buttons[(r_idx, self.selected_column)].configure(fg_color=column_highlight_color)

        # Highlight rows
        if self.selected_row != -1:
            for c_idx in range(len(self.keys_layout[0])):
                if (self.selected_row, c_idx) in self.key_buttons:
                    self.key_buttons[(self.selected_row, c_idx)].configure(fg_color=row_highlight_color)
        
        # Highlight the specific selected cell if both row and column are selected
        if self.selected_row != -1 and self.selected_column != -1:
            if (self.selected_row, self.selected_column) in self.key_buttons:
                self.key_buttons[(self.selected_row, self.selected_column)].configure(fg_color=selected_cell_color)

    def check_and_trigger_keyboard_action(self):
        """Main logic to handle finger gestures and trigger keyboard actions"""
        current_gesture_state = self.finger_recognized_var.get()
        
        try:
            # Parse gesture state: "index_middle_pinky_thumb"
            index_up, middle_up, pinky_up, thumb_up = map(int, current_gesture_state.split('_'))
        except (ValueError, AttributeError):
            index_up, middle_up, pinky_up, thumb_up = 0, 0, 0, 0
        
        # Parse previous state
        try:
            prev_index, prev_middle, prev_pinky, prev_thumb = map(int, self._last_gesture_state.split('_'))
        except (ValueError, AttributeError):
            prev_index, prev_middle, prev_pinky, prev_thumb = 0, 0, 0, 0

        # Detect rising edges (finger going from down to up)
        index_rising_edge = index_up == 1 and prev_index == 0
        middle_rising_edge = middle_up == 1 and prev_middle == 0
        pinky_rising_edge = pinky_up == 1 and prev_pinky == 0
        thumb_rising_edge = thumb_up == 1 and prev_thumb == 0

        # Handle index finger (column selection)
        if index_rising_edge:
            # Select based on current counter before increment to start at first column
            self.update_column_selection()
            self._index_column_counter += 1
            self.action_status_label.configure(text="🎯 Action: Column Selected")

        # Handle middle finger (backspace)
        if middle_rising_edge:
            self.perform_backspace()
            self.action_status_label.configure(text="🎯 Action: Backspace")

        # Handle pinky finger (row selection)
        if pinky_rising_edge:
            # Select based on current counter before increment to start at first row
            self.update_row_selection()
            self._pinky_row_counter += 1
            self.action_status_label.configure(text="🎯 Action: Row Selected")

        # Handle thumb (enter character)
        if thumb_rising_edge:
            self.perform_select_and_enter()
            self.action_status_label.configure(text="🎯 Action: Character Entered")

        # Update status based on current finger states
        status_parts = []
        if index_up:
            status_parts.append("👆 Index (Column)")
        if middle_up:
            status_parts.append("🖕 Middle (Backspace)")
        if pinky_up:
            status_parts.append("🤙 Pinky (Row)")
        if thumb_up:
            status_parts.append("👍 Thumb (Enter)")
        
        if not status_parts:
            self.action_status_label.configure(text="🎯 Status: No gesture detected")
        else:
            self.action_status_label.configure(text=f"🎯 Active: {', '.join(status_parts)}")

        # Update the last gesture state
        self._last_gesture_state = current_gesture_state

        # Schedule the next check
        self.master.after(100, self.check_and_trigger_keyboard_action)

    def on_closing(self):
        """Handles graceful shutdown when the Tkinter window is closed."""
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.stop()
            self.video_thread.join() # Wait for the thread to finish its execution
        self.master.destroy() # Destroy the Tkinter window

# --- Main application execution ---
if __name__ == "__main__":
    main_window = ctk.CTk()  # Use CTk instead of tk.Tk()
    app = AlphaNumericKeyboardApp(main_window)
    main_window.mainloop()
