import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import threading
import queue
import os
import tempfile
import shutil
import cv2  # Requires opencv-python package

from main import run

class SFMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Structure from Motion")
        self.geometry("800x600")
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
        
        # Input Type Selection
        input_type_frame = ttk.Frame(self)
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
        self.input_path_frame = ttk.Frame(self)
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
        dir_k_frame = ttk.Frame(self)
        dir_k_frame.pack(pady=10, fill=tk.X)

        ttk.Label(dir_k_frame, text="Select K Matrix File:").pack(anchor=tk.W, padx=5)
        
        self.dir_k_entry = ttk.Entry(dir_k_frame, width=50)
        self.dir_k_entry.pack(side=tk.LEFT, padx=5)
        self.input_widgets.append(self.dir_k_entry)
        
        browse_k_btn = ttk.Button(dir_k_frame, text="Browse", command=self.browse_k_file)
        browse_k_btn.pack(side=tk.LEFT)
        self.input_widgets.append(browse_k_btn)

        # Result format selection
        result_format_frame = ttk.Frame(self)
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
        run_btn = ttk.Button(self, text="Run SFM", command=self.run_sfm)
        run_btn.pack(pady=10)
        self.input_widgets.append(run_btn)
        
        # Log text area
        self.log_text = tk.Text(self, wrap=tk.WORD)
        self.log_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def update_input_ui(self):
        """Update UI elements based on input type selection"""
        if self.input_type_var.get() == "image":
            self.input_path_label.config(text="Select Image Directory:")
            self.browse_input_btn.config(command=self.browse_image_directory)
        else:
            self.input_path_label.config(text="Select Video File:")
            self.browse_input_btn.config(command=self.browse_video_file)

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

    def browse_k_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if path:
            self.dir_k_entry.delete(0, tk.END)
            self.dir_k_entry.insert(0, path)

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
            run(input_path, k_path, result_format)
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