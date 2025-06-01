import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import threading
import queue
import os
import tempfile
import shutil
import cv2

from sfm import StructureFromMotion
from camera_calibration import CameraCalibrator

class StdoutRedirector:
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass

class InstructionsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Instructions - 3D Reconstruction Tools")
        self.geometry("800x600")
        self.resizable(True, True)
        self.transient(parent)
        self.grab_set()
        
        # Center the window
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        self.create_widgets()

    def create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="User Instructions", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create scrollable text widget
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget with scrollbar
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            bg='white',
            fg='black',
            padx=15,
            pady=15,
            relief='solid',
            borderwidth=1
        )
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Instructions content
        instructions_text = self.load_instructions_text()
        text_widget.insert('1.0', instructions_text)
        text_widget.config(state='disabled')
        
        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.destroy,
            style='Action.TButton'
        )
        close_btn.pack(anchor=tk.CENTER)

    def load_instructions_text(self):
        """Load instructions text from external file"""
        instructions_file = "instructions.txt"
        
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            instructions_path = os.path.join(script_dir, instructions_file)
            
            if os.path.exists(instructions_path):
                with open(instructions_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif os.path.exists(instructions_file):
                with open(instructions_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return "Instructions file not found. Please ensure 'instructions.txt' is in the same directory as this application."
                    
        except Exception as e:
            print(f"Warning: Could not load instructions from file: {e}")
            return "Error loading instructions file."

class SFMTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.input_widgets = []
        self.create_widgets()

    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Top frame for side-by-side layout
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Left column - Input Configuration
        left_frame = ttk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        input_section = self.app.create_section_frame(left_frame, "Input Configuration")
        input_section.pack(fill=tk.BOTH, expand=True)

        # Input type with modern radio buttons
        type_frame = ttk.Frame(input_section)
        type_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(type_frame, text="Data Source:", style='Heading.TLabel').pack(anchor=tk.W)
        
        radio_frame = ttk.Frame(type_frame)
        radio_frame.pack(anchor=tk.W, pady=(8, 0))
        
        self.input_type_var = tk.StringVar(value="image")
        
        image_radio = ttk.Radiobutton(
            radio_frame, 
            text="Image Directory", 
            variable=self.input_type_var, 
            value="image", 
            command=self.update_input_ui
        )
        image_radio.pack(anchor=tk.W, pady=2)
        
        video_radio = ttk.Radiobutton(
            radio_frame, 
            text="Video File", 
            variable=self.input_type_var, 
            value="video", 
            command=self.update_input_ui
        )
        video_radio.pack(anchor=tk.W, pady=2)

        # Input Path selection with improved layout
        path_frame = ttk.Frame(input_section)
        path_frame.pack(fill=tk.X, pady=(0, 15))

        self.input_path_label = ttk.Label(path_frame, text="Select Image Directory:", style='Heading.TLabel')
        self.input_path_label.pack(anchor=tk.W, pady=(0, 8))
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(fill=tk.X)
        
        self.input_path_entry = ttk.Entry(path_input_frame, font=('Segoe UI', 9))
        self.input_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_widgets.append(self.input_path_entry)
        
        self.browse_input_btn = ttk.Button(
            path_input_frame, 
            text="Browse...", 
            command=self.browse_image_directory,
            width=12
        )
        self.browse_input_btn.pack(side=tk.RIGHT)
        self.input_widgets.append(self.browse_input_btn)

        # Directory K Matrix selection
        dir_k_frame = ttk.Frame(input_section)
        dir_k_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(dir_k_frame, text="Camera Matrix File (K):", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        k_input_frame = ttk.Frame(dir_k_frame)
        k_input_frame.pack(fill=tk.X)
        
        self.dir_k_entry = ttk.Entry(k_input_frame, font=('Segoe UI', 9))
        self.dir_k_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_widgets.append(self.dir_k_entry)
        
        browse_k_btn = ttk.Button(k_input_frame, text="Browse...", command=self.browse_k_file, width=12)
        browse_k_btn.pack(side=tk.RIGHT)
        self.input_widgets.append(browse_k_btn)

        # Right column - Output Configuration
        right_frame = ttk.Frame(top_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        output_section = self.app.create_section_frame(right_frame, "Output Configuration")
        output_section.pack(fill=tk.BOTH, expand=True)

        # Result format selection
        format_frame = ttk.Frame(output_section)
        format_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(format_frame, text="Output Format:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        format_radio_frame = ttk.Frame(format_frame)
        format_radio_frame.pack(anchor=tk.W)
        
        self.result_format_var = tk.StringVar()
        self.result_format_var.set("ply")
        
        ply_radio_btn = ttk.Radiobutton(
            format_radio_frame, 
            text="PLY Format", 
            variable=self.result_format_var, 
            value="ply"
        )
        ply_radio_btn.pack(anchor=tk.W, pady=2)
        self.input_widgets.append(ply_radio_btn)
        
        obj_radio_btn = ttk.Radiobutton(
            format_radio_frame, 
            text="OBJ Format", 
            variable=self.result_format_var, 
            value="obj"
        )
        obj_radio_btn.pack(anchor=tk.W, pady=2)
        self.input_widgets.append(obj_radio_btn)
        
        # Run button - centered at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        run_btn = ttk.Button(
            button_frame, 
            text="Start 3D Reconstruction", 
            command=self.run_sfm,
            style='Action.TButton'
        )
        run_btn.pack(anchor=tk.CENTER)
        self.input_widgets.append(run_btn)

    def update_input_ui(self):
        """Update UI elements based on input type selection"""
        if self.input_type_var.get() == "image":
            self.input_path_label.config(text="Select Image Directory:")
            self.browse_input_btn.config(command=self.browse_image_directory)
        else:
            self.input_path_label.config(text="Select Video File:")
            self.browse_input_btn.config(command=self.browse_video_file)

    def browse_image_directory(self):
        dir_path = filedialog.askdirectory(title="Select Image Directory")
        if dir_path:
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, dir_path)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_path_entry.delete(0, tk.END)
            self.input_path_entry.insert(0, file_path)

    def browse_k_file(self):
        path = filedialog.askopenfilename(
            title="Select Camera Matrix File",
            filetypes=[("Text Files", "*.txt"), ("NumPy Files", "*.npy"), ("All Files", "*.*")]
        )
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
                messagebox.showerror("Input Error", "Please select a valid image directory")
                return
        else:
            if not input_path or not os.path.isfile(input_path):
                messagebox.showerror("Input Error", "Please select a valid video file")
                return

        if not k_path or not os.path.isfile(k_path):
            messagebox.showerror("Input Error", "Please select a valid camera matrix file")
            return

        # Disable UI during processing
        self.app.status_var.set("Processing 3D reconstruction...")
        self.set_input_state(tk.DISABLED)
        
        # Add processing start message
        self.app.log_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.app.log_text.insert(tk.END, "Starting 3D reconstruction process...\n")
        self.app.log_text.see(tk.END)
        
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
                self.app.output_queue.put("Extracting frames from video...\n")
                self.app.temp_frame_dir = tempfile.mkdtemp()
                success = self.extract_frames(input_path, self.app.temp_frame_dir)
                if not success:
                    raise ValueError("Failed to extract frames from video")
                input_path = self.app.temp_frame_dir

            # Run main SFM process with either image dir or temp frame dir
            sfm = StructureFromMotion(input_path, k_path)
            sfm.run(result_format)
            self.app.output_queue.put("\n3D reconstruction completed successfully!\n")
            
        except Exception as e:
            self.app.output_queue.put(f"\nError: {str(e)}\n")
        finally:
            self.set_input_state(tk.NORMAL)
            self.app.status_var.set("Ready")
            # Cleanup temp directory
            if self.app.temp_frame_dir and os.path.exists(self.app.temp_frame_dir):
                shutil.rmtree(self.app.temp_frame_dir)
                self.app.temp_frame_dir = None

    def extract_frames(self, video_path, output_dir):
        try:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                return False

            count = 0
            success, image = vidcap.read()
            while success:
                frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
                cv2.imwrite(frame_path, image)
                count += 1
                success, image = vidcap.read()
                
            self.app.output_queue.put(f"Extracted {count} frames successfully\n")
            return count > 0
        except Exception as e:
            self.app.output_queue.put(f"Frame extraction error: {str(e)}\n")
            return False

    def set_input_state(self, state):
        for widget in self.input_widgets:
            widget.config(state=state)

class CalibrationTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Top frame for side-by-side layout
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Left column - Calibration Source
        left_frame = ttk.Frame(top_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        input_section = self.app.create_section_frame(left_frame, "Calibration Source")
        input_section.pack(fill=tk.BOTH, expand=True)

        # Calibration input type
        type_frame = ttk.Frame(input_section)
        type_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(type_frame, text="Data Source:", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        radio_frame = ttk.Frame(type_frame)
        radio_frame.pack(anchor=tk.W)
        
        self.calib_input_type_var = tk.StringVar(value="images")
        
        images_radio = ttk.Radiobutton(
            radio_frame, 
            text="Calibration Images Directory", 
            variable=self.calib_input_type_var, 
            value="images", 
            command=self.update_calib_ui
        )
        images_radio.pack(anchor=tk.W, pady=2)
        
        video_radio = ttk.Radiobutton(
            radio_frame, 
            text="Calibration Video File", 
            variable=self.calib_input_type_var, 
            value="video", 
            command=self.update_calib_ui
        )
        video_radio.pack(anchor=tk.W, pady=2)

        # Calibration input path
        path_frame = ttk.Frame(input_section)
        path_frame.pack(fill=tk.X, pady=(0, 15))

        self.calib_path_label = ttk.Label(path_frame, text="Select Calibration Images Directory:", style='Heading.TLabel')
        self.calib_path_label.pack(anchor=tk.W, pady=(0, 8))
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(fill=tk.X)
        
        self.calib_path_entry = ttk.Entry(path_input_frame, font=('Segoe UI', 9))
        self.calib_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.browse_calib_btn = ttk.Button(
            path_input_frame, 
            text="Browse...", 
            command=self.browse_calib_images,
            width=12
        )
        self.browse_calib_btn.pack(side=tk.RIGHT)

        # Right column - Calibration Settings
        right_frame = ttk.Frame(top_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        settings_section = self.app.create_section_frame(right_frame, "Calibration Settings")
        settings_section.pack(fill=tk.BOTH, expand=True)

        # Chessboard size settings
        chessboard_frame = ttk.Frame(settings_section)
        chessboard_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(chessboard_frame, text="Chessboard Pattern (inner corners):", style='Heading.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        size_input_frame = ttk.Frame(chessboard_frame)
        size_input_frame.pack(anchor=tk.W)
        
        ttk.Label(size_input_frame, text="Width:").pack(side=tk.LEFT, padx=(0, 5))
        self.chessboard_width_var = tk.StringVar(value="10")
        width_entry = ttk.Entry(size_input_frame, textvariable=self.chessboard_width_var, width=8)
        width_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(size_input_frame, text="Height:").pack(side=tk.LEFT, padx=(0, 5))
        self.chessboard_height_var = tk.StringVar(value="7")
        height_entry = ttk.Entry(size_input_frame, textvariable=self.chessboard_height_var, width=8)
        height_entry.pack(side=tk.LEFT)

        # Frame skip for video (only shown when video is selected)
        self.frame_skip_frame = ttk.Frame(settings_section)
        
        skip_label_frame = ttk.Frame(self.frame_skip_frame)
        skip_label_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(skip_label_frame, text="Frame Processing:", style='Heading.TLabel').pack(anchor=tk.W)
        ttk.Label(skip_label_frame, text="Process every Nth frame:", foreground='#6c757d').pack(anchor=tk.W)
        
        self.frame_skip_var = tk.StringVar(value="30")
        frame_skip_entry = ttk.Entry(self.frame_skip_frame, textvariable=self.frame_skip_var, width=15)
        frame_skip_entry.pack(anchor=tk.W)

        # Run calibration button - centered at bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        run_calib_btn = ttk.Button(
            button_frame, 
            text="Run Camera Calibration", 
            command=self.run_calibration,
            style='Action.TButton'
        )
        run_calib_btn.pack(anchor=tk.CENTER)

    def update_calib_ui(self):
        """Update calibration UI based on input type"""
        if self.calib_input_type_var.get() == "images":
            self.calib_path_label.config(text="Select Calibration Images Directory:")
            self.browse_calib_btn.config(command=self.browse_calib_images)
            self.frame_skip_frame.pack_forget()
        else:
            self.calib_path_label.config(text="Select Calibration Video File:")
            self.browse_calib_btn.config(command=self.browse_calib_video)
            self.frame_skip_frame.pack(fill=tk.X, pady=(0, 15))

    def browse_calib_images(self):
        dir_path = filedialog.askdirectory(title="Select Calibration Images Directory")
        if dir_path:
            self.calib_path_entry.delete(0, tk.END)
            self.calib_path_entry.insert(0, dir_path)

    def browse_calib_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Calibration Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.calib_path_entry.delete(0, tk.END)
            self.calib_path_entry.insert(0, file_path)

    def run_calibration(self):
        """Run camera calibration in a separate thread"""
        calib_path = self.calib_path_entry.get()
        calib_type = self.calib_input_type_var.get()
        
        # Validate inputs
        if calib_type == "images":
            if not calib_path or not os.path.isdir(calib_path):
                messagebox.showerror("Input Error", "Please select a valid calibration images directory")
                return
        else:
            if not calib_path or not os.path.isfile(calib_path):
                messagebox.showerror("Input Error", "Please select a valid calibration video file")
                return

        # Get chessboard size
        try:
            width = int(self.chessboard_width_var.get())
            height = int(self.chessboard_height_var.get())
            if width <= 0 or height <= 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid positive integers for chessboard size")
            return

        # Get frame skip (for video only)
        frame_skip = 30
        if calib_type == "video":
            try:
                frame_skip = int(self.frame_skip_var.get())
                if frame_skip <= 0:
                    raise ValueError()
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid positive integer for frame skip")
                return

        # Start calibration
        self.app.status_var.set("Running camera calibration...")
        self.app.log_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.app.log_text.insert(tk.END, "Starting camera calibration process...\n")
        self.app.log_text.see(tk.END)
        
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
                self.app.output_queue.put("\nCamera calibration failed!\n")
            else:
                self.app.output_queue.put("\nCamera calibration completed successfully!\n")
                
        except Exception as e:
            self.app.output_queue.put(f"\nCalibration error: {str(e)}\n")
        finally:
            self.app.status_var.set("Ready")

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Structure from Motion - 3D Reconstruction Tools")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.temp_frame_dir = None
        
        # Configure modern styling
        self.configure_styling()
        
        # Create GUI elements
        self.create_widgets()
        
        # Redirect stdout
        self.output_queue = queue.Queue()
        self.original_stdout = sys.stdout
        sys.stdout = StdoutRedirector(self.output_queue)
        
        self.after(100, self.process_output)
        
        # Cleanup temp directory on exit
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_styling(self):
        """Configure modern styling for the application"""
        # Configure ttk styles
        style = ttk.Style()
        
        # Use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 10, 'bold'))
        style.configure('Action.TButton', font=('Segoe UI', 10, 'bold'))
        
        # Configure colors
        self.configure(bg='#f8f9fa')

    def create_widgets(self):
        # Main container with padding
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Header with instructions button
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title and subtitle - centered
        title_section = ttk.Frame(header_frame)
        title_section.pack(expand=True, fill=tk.X)
        
        title_label = ttk.Label(
            title_section, 
            text="3D Reconstruction Tools", 
            style='Title.TLabel'
        )
        title_label.pack(anchor=tk.CENTER)
        
        subtitle_label = ttk.Label(
            title_section, 
            text="Structure from Motion & Camera Calibration",
            foreground='#6c757d'
        )
        subtitle_label.pack(anchor=tk.CENTER, pady=(5, 0))
        
        # Instructions button - positioned absolutely to top right
        instructions_btn = ttk.Button(
            header_frame,
            text="Instructions",
            command=self.show_instructions,
            width=15
        )
        instructions_btn.place(relx=1.0, rely=0.0, anchor=tk.NE)
        
        # Create notebook for tabs with better styling
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # SFM Tab
        self.sfm_tab = SFMTab(self.notebook, self)
        self.notebook.add(self.sfm_tab, text="  Structure from Motion  ")
        
        # Camera Calibration Tab
        self.calib_tab = CalibrationTab(self.notebook, self)
        self.notebook.add(self.calib_tab, text="  Camera Calibration  ")
        
        # Log section - full width at bottom
        self.create_log_section(main_container)
        
        # Status bar
        self.create_status_bar()

    def create_section_frame(self, parent, title):
        """Create a styled section frame with title"""
        section_frame = ttk.LabelFrame(parent, text=title, padding=(15, 10))
        return section_frame

    def create_log_section(self, parent):
        """Create the output log section - full width and larger"""
        log_section = self.create_section_frame(parent, "Processing Output")
        log_section.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create text widget with scrollbar - increased height
        text_frame = ttk.Frame(log_section)
        text_frame.pack(expand=True, fill=tk.BOTH)
        
        self.log_text = tk.Text(
            text_frame, 
            wrap=tk.WORD, 
            height=18,  # Increased from 12 to 18
            font=('Consolas', 9),
            bg='#2d3748',
            fg='#e2e8f0',
            insertbackground='#4fd1c7',
            selectbackground='#4a5568'
        )
        
        log_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add welcome message
        welcome_msg = "Welcome to the 3D Reconstruction Tools!\nSelect your input data and configuration, then click the run button to begin.\n" + "="*60 + "\n"
        self.log_text.insert(tk.END, welcome_msg)

    def create_status_bar(self):
        """Create modern status bar"""
        status_frame = ttk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
        
        separator = ttk.Separator(status_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=(0, 8))
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            status_frame, 
            textvariable=self.status_var,
            font=('Segoe UI', 9),
            foreground='#495057'
        )
        status_bar.pack(anchor=tk.W)

    def show_instructions(self):
        """Show instructions popup window"""
        InstructionsDialog(self)

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

if __name__ == "__main__":
    app = GUI()
    app.mainloop()