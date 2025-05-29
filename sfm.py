# sfm.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import threading
import queue
import os
import tempfile
import shutil
import cv2

from core import StructureFromMotion
from camera_calibration import CameraCalibrator

class SFMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Structure from Motion")
        self.geometry("800x700")
        self.temp_frame_dir = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Redirect stdout
        self.output_queue = queue.Queue()
        self.original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.output_queue)
        
        self.after(100, self.process_output)
        
        # Cleanup temp directory on exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        self.input_widgets = []
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # SFM Tab
        sfm_frame = ttk.Frame(notebook)
        notebook.add(sfm_frame, text="Structure from Motion")
        self.create_sfm_widgets(sfm_frame)
        
        # Camera Calibration Tab
        calib_frame = ttk.Frame(notebook)
        notebook.add(calib_frame, text="Camera Calibration")
        self.create_calibration_widgets(calib_frame)

    def create_sfm_widgets(self, parent):
        # Input Type Selection
        input_type_frame = ttk.Frame(parent)
        input_type_frame.pack(pady=5, fill=tk.X)

        ttk.Label(input_type_frame, text="Input Type:").pack(side=tk.LEFT, padx=5)
        
        self.input_type_var = tk.StringVar(value="image")
        ttk.Radiobutton(
            input_type_frame, 
            text="Image Directory", 
            variable=self.input_type_var, 
            value="image", 
            command=self.update_input_ui
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            input_type_frame, 
            text="Video File", 
            variable=self.input_type_var, 
            value="video", 
            command=self.update_input_ui
        ).pack(side=tk.LEFT, padx=5)

        # Input Path selection
        self.input_path_frame = ttk.Frame(parent)
        self.input_path_frame.pack(pady=10, fill=tk.X)

        self.input_path_label = ttk.Label(self.input_path_frame, text="Select Image Directory:")
        self.input_path_label.pack(anchor=tk.W, padx=5)
        
        self.input_path_entry = ttk.Entry(self.input_path_frame, width=50)
        self.input_path_entry.pack(side=tk.LEFT, padx=5)
        self.input_widgets.append(self.input_path_entry)
        
        self.browse_input_btn = ttk.Button(
            self.input_path_frame, 
            text="Browse", 
            command=self.browse_image_directory
        )
        self.browse_input_btn.pack(side=tk.LEFT)
        self.input_widgets.append(self.browse_input_btn)

        # Directory K Matrix selection
        dir_k_frame = ttk.Frame(parent)
        dir_k_frame.pack(pady=10, fill=tk.X)

        ttk.Label(dir_k_frame, text="Select K Matrix File:").pack(anchor=tk.W, padx=5)
        
        self.dir_k_entry = ttk.Entry(dir_k_frame, width=50)
        self.dir_k_entry.pack(side=tk.LEFT, padx=5)
        self.input_widgets.append(self.dir_k_entry)
        
        browse_k_btn = ttk.Button(dir_k_frame, text="Browse", command=self.browse_k_file)
        browse_k_btn.pack(side=tk.LEFT)
        self.input_widgets.append(browse_k_btn)

        # Result format selection
        result_format_frame = ttk.Frame(parent)
        result_format_frame.pack(pady=10, fill=tk.X)

        ttk.Label(result_format_frame, text="Select Result Format:").pack(anchor=tk.W, padx=5)
        
        self.result_format_var = tk.StringVar()
        self.result_format_var.set("ply")  # Default value
        ply_radio_btn = ttk.Radiobutton(result_format_frame, text="PLY", variable=self.result_format_var, value="ply")
        ply_radio_btn.pack(anchor=tk.W, padx=5)
        self.input_widgets.append(ply_radio_btn)
        obj_radio_btn = ttk.Radiobutton(result_format_frame, text="OBJ", variable=self.result_format_var, value="obj")
        obj_radio_btn.pack(anchor=tk.W, padx=5)
        self.input_widgets.append(obj_radio_btn)
        
        # Run button
        run_btn = ttk.Button(parent, text="Run SFM", command=self.run_sfm)
        run_btn.pack(pady=10)
        self.input_widgets.append(run_btn)

    def create_calibration_widgets(self, parent):
        # Calibration input type
        calib_input_frame = ttk.Frame(parent)
        calib_input_frame.pack(pady=5, fill=tk.X)

        ttk.Label(calib_input_frame, text="Calibration Source:").pack(side=tk.LEFT, padx=5)
        
        self.calib_input_type_var = tk.StringVar(value="images")
        ttk.Radiobutton(
            calib_input_frame, 
            text="Image Directory", 
            variable=self.calib_input_type_var, 
            value="images", 
            command=self.update_calib_ui
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            calib_input_frame, 
            text="Video File", 
            variable=self.calib_input_type_var, 
            value="video", 
            command=self.update_calib_ui
        ).pack(side=tk.LEFT, padx=5)

        # Calibration input path
        self.calib_path_frame = ttk.Frame(parent)
        self.calib_path_frame.pack(pady=10, fill=tk.X)

        self.calib_path_label = ttk.Label(self.calib_path_frame, text="Select Calibration Images Directory:")
        self.calib_path_label.pack(anchor=tk.W, padx=5)
        
        self.calib_path_entry = ttk.Entry(self.calib_path_frame, width=50)
        self.calib_path_entry.pack(side=tk.LEFT, padx=5)
        
        self.browse_calib_btn = ttk.Button(
            self.calib_path_frame, 
            text="Browse", 
            command=self.browse_calib_images
        )
        self.browse_calib_btn.pack(side=tk.LEFT)

        # Chessboard size settings
        chessboard_frame = ttk.Frame(parent)
        chessboard_frame.pack(pady=10, fill=tk.X)

        ttk.Label(chessboard_frame, text="Chessboard Size (inner corners):").pack(anchor=tk.W, padx=5)
        
        size_input_frame = ttk.Frame(chessboard_frame)
        size_input_frame.pack(anchor=tk.W, padx=20)
        
        ttk.Label(size_input_frame, text="Width:").pack(side=tk.LEFT, padx=5)
        self.chessboard_width_var = tk.StringVar(value="10")
        width_entry = ttk.Entry(size_input_frame, textvariable=self.chessboard_width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(size_input_frame, text="Height:").pack(side=tk.LEFT, padx=5)
        self.chessboard_height_var = tk.StringVar(value="7")
        height_entry = ttk.Entry(size_input_frame, textvariable=self.chessboard_height_var, width=5)
        height_entry.pack(side=tk.LEFT, padx=5)

        # Frame skip for video (only shown when video is selected)
        self.frame_skip_frame = ttk.Frame(parent)
        
        ttk.Label(self.frame_skip_frame, text="Process every Nth frame:").pack(anchor=tk.W, padx=5)
        self.frame_skip_var = tk.StringVar(value="30")
        frame_skip_entry = ttk.Entry(self.frame_skip_frame, textvariable=self.frame_skip_var, width=10)
        frame_skip_entry.pack(anchor=tk.W, padx=20)

        # Run calibration button
        run_calib_btn = ttk.Button(parent, text="Run Camera Calibration", command=self.run_calibration)
        run_calib_btn.pack(pady=10)

        # Log text area (shared between tabs)
        log_frame = ttk.Frame(self)
        log_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        ttk.Label(log_frame, text="Output Log:").pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=15)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_input_ui(self):
        """Update UI elements based on input type selection"""
        if self.input_type_var.get() == "image":
            self.input_path_label.config(text="Select Image Directory:")
            self.browse_input_btn.config(command=self.browse_image_directory)
        else:
            self.input_path_label.config(text="Select Video File:")
            self.browse_input_btn.config(command=self.browse_video_file)

    def update_calib_ui(self):
        """Update calibration UI based on input type"""
        if self.calib_input_type_var.get() == "images":
            self.calib_path_label.config(text="Select Calibration Images Directory:")
            self.browse_calib_btn.config(command=self.browse_calib_images)
            self.frame_skip_frame.pack_forget()
        else:
            self.calib_path_label.config(text="Select Calibration Video File:")
            self.browse_calib_btn.config(command=self.browse_calib_video)
            self.frame_skip_frame.pack(pady=10, fill=tk.X)

    def browse_image_directory(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, dir_path)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, file_path)

    def browse_calib_images(self):
        dir_path = filedialog.askdirectory(title="Select Calibration Images Directory")
        if dir_path:
            self.calib_path_entry.delete(0, tk.END)
            self.calib_path_entry.insert(0, dir_path)

    def browse_calib_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Calibration Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if file_path:
            self.calib_path_entry.delete(0, tk.END)
            self.calib_path_entry.insert(0, file_path)

    def browse_k_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            self.dir_k_entry.delete(0, tk.END)
            self.dir_k_entry.insert(0, path)

    def run_calibration(self):
        """Run camera calibration in a separate thread"""
        calib_path = self.calib_path_entry.get()
        calib_type = self.calib_input_type_var.get()
        
        # Validate inputs
        if calib_type == "images":
            if not calib_path or not os.path.isdir(calib_path):
                messagebox.showerror("Error", "Please select a valid calibration images directory")
                return
        else:
            if not calib_path or not os.path.isfile(calib_path):
                messagebox.showerror("Error", "Please select a valid calibration video file")
                return

        # Get chessboard size
        try:
            width = int(self.chessboard_width_var.get())
            height = int(self.chessboard_height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid positive integers for chessboard size")
            return

        # Get frame skip (for video only)
        frame_skip = 30
        if calib_type == "video":
            try:
                frame_skip = int(self.frame_skip_var.get())
                if frame_skip <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid positive integer for frame skip")
                return

        # Start calibration
        self.status_var.set("Running camera calibration...")
        self.log_text.insert(tk.END, "\n" + "="*50 + "\n")
        self.log_text.see(tk.END)
        
        calib_thread = threading.Thread(
            target=self.execute_calibration,
            args=(calib_path, calib_type, (width, height), frame_skip),
            daemon=True
        )
        calib_thread.start()

    def execute_calibration(self, calib_path, calib_type, chessboard_size, frame_skip):
        """Execute camera calibration"""
        try:
            calibrator = CameraCalibrator(chessboard_size)
            
            if calib_type == "images":
                mtx = calibrator.calibrate_from_images(calib_path)
            else:
                mtx = calibrator.calibrate_from_video(calib_path, frame_skip)
            
            if mtx is None:
                self.output_queue.put("\nCamera calibration failed!\n")
                
        except Exception as e:
            self.output_queue.put(f"\nCalibration error: {str(e)}\n")
        finally:
            self.status_var.set("Ready")

    def run_sfm(self):
        input_path = self.input_path_entry.get()
        k_path = self.dir_k_entry.get()
        result_format = self.result_format_var.get()
        input_type = self.input_type_var.get()

        # Validate inputs
        if input_type == "image":
            if not input_path or not os.path.isdir(input_path):
                messagebox.showerror("Error", "Invalid image directory path")
                return
        else:
            if not input_path or not os.path.isfile(input_path):
                messagebox.showerror("Error", "Invalid video file path")
                return

        if not k_path or not os.path.isfile(k_path):
            messagebox.showerror("Error", "Invalid K matrix file path")
            return

        # Disable UI during processing
        self.status_var.set("Processing...")
        self.input_state(tk.DISABLED)
        
        # Start processing thread
        sfm_thread = threading.Thread(
            target=self.execute_sfm,
            args=(input_path, k_path, result_format, input_type),
            daemon=True
        )
        sfm_thread.start()

    def execute_sfm(self, input_path, k_path, result_format, input_type):
        try:
            # Handle video input
            if input_type == "video":
                self.output_queue.put("Extracting frames from video...\n")
                self.temp_frame_dir = tempfile.mkdtemp()
                success = self.extract_frames(input_path, self.temp_frame_dir)
                if not success:
                    raise ValueError("Failed to extract frames from video")
                input_path = self.temp_frame_dir

            # Run main SFM process with either image dir or temp frame dir
            sfm = StructureFromMotion(input_path, k_path)
            sfm.run(result_format)
            self.output_queue.put("\nProcessing completed successfully!\n")
            
        except Exception as e:
            self.output_queue.put(f"\nError: {str(e)}\n")
        finally:
            self.input_state(tk.NORMAL)
            self.status_var.set("Ready")
            # Cleanup temp directory
            if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
                shutil.rmtree(self.temp_frame_dir)
                self.temp_frame_dir = None

    def extract_frames(self, video_path, output_dir):
        try:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                return False

            count = 0
            success, image = vidcap.read()
            while success:
                frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
                cv2.imwrite(frame_path, image)
                count += 1
                success, image = vidcap.read()
                
            self.output_queue.put(f"Extracted {count} frames\n")
            return count > 0
        except Exception as e:
            self.output_queue.put(f"Frame extraction error: {str(e)}\n")
            return False

    def input_state(self, state):
        for widget in self.input_widgets:
            widget.config(state=state)

    def process_output(self):
        while not self.output_queue.empty():
            msg = self.output_queue.get_nowait()
            self.log_text.insert(tk.END, msg)
            self.log_text.see(tk.END)
        self.after(100, self.process_output)

    def on_close(self):
        """Cleanup temp directory when closing"""
        if self.temp_frame_dir and os.path.exists(self.temp_frame_dir):
            shutil.rmtree(self.temp_frame_dir)
        self.destroy()

class StdoutRedirector:
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

if __name__ == "__main__":
    app = SFMApp()
    app.mainloop()