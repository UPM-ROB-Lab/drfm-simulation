import json
import threading
import pathlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import yaml
import os
import subprocess
import sys
import io
import re
from PIL import Image, ImageTk
from run.run_single_simulation import run_single_simulation


class SimulationGUI(tk.Tk):
    """Tkinter-based GUI for running a single steering simulation."""

    DEFAULT_PARAMS_PATH = pathlib.Path('run') / 'simulation_params.json'
    DEFAULT_CONFIG_PATH = pathlib.Path('config.yaml')
    DEFAULT_OUTPUT_DIR = pathlib.Path('single_result')
    
    # Parameter name mapping from technical names to user-friendly labels
    PARAM_LABELS = {
        'left_steering_angle': 'Left Steering Angle (deg)',
        'right_steering_angle': 'Right Steering Angle (deg)',
        'beta_FR': 'Front Right Friction Coefficient',
        'beta_RL': 'Rear Left Friction Coefficient', 
        'beta_RR': 'Rear Right Friction Coefficient',
        'max_speed': 'Maximum Speed (rad/s)',
        'turning_radius': 'Turning Radius (m)',
        'total_time': 'Total Simulation Time (s)',
        'dt': 'Time Step Size (s)',
        'arc_angle': 'Arc Angle (degrees)',
        'arc_fraction': 'Arc Fraction (0-1)',
        'accel_time': 'Acceleration Time (s)',
        'const_time': 'Constant Speed Time (s)'
    }

    def __init__(self):
        super().__init__()
        self.title('Single Simulation Runner')
        self.geometry('900x700')

        # Simulation control variables
        self.simulation_running = False
        self.simulation_thread = None
        self.stop_requested = False

        # Main containers
        self._build_param_frame()
        self._build_file_select_frame()
        self._build_control_frame()
        self._build_status_frame()

        # Load defaults
        self.default_params = self._load_default_params()
        self.param_vars = {}
        self._populate_param_entries(self.default_params)

    # --------------------------- UI BUILDERS --------------------------- #
    def _build_param_frame(self):
        self.param_container = ttk.Labelframe(self, text='Simulation Parameters')
        self.param_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Scrollable canvas inside container
        canvas = tk.Canvas(self.param_container, borderwidth=0)
        vscroll = ttk.Scrollbar(self.param_container, orient='vertical', command=canvas.yview)
        self.param_inner = ttk.Frame(canvas)
        self.param_inner.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        canvas.create_window((0, 0), window=self.param_inner, anchor='nw')
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side='left', fill='both', expand=True)
        vscroll.pack(side='right', fill='y')

    def _build_file_select_frame(self):
        self.file_frame = ttk.Labelframe(self, text='Files & Directories')
        self.file_frame.pack(fill='x', padx=10, pady=5)

        # Config.yaml selector
        ttk.Label(self.file_frame, text='Config YAML:').grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.config_path_var = tk.StringVar(value=str(self.DEFAULT_CONFIG_PATH))
        ttk.Entry(self.file_frame, textvariable=self.config_path_var, width=50).grid(row=0, column=1, padx=5, pady=3)
        ttk.Button(self.file_frame, text='Browse', command=self._choose_config_file).grid(row=0, column=2, padx=5, pady=3)
        ttk.Button(self.file_frame, text='Edit Config', command=self._edit_config_file).grid(row=0, column=3, padx=5, pady=3)

        # Output dir selector
        ttk.Label(self.file_frame, text='Output Directory:').grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.output_dir_var = tk.StringVar(value=str(self.DEFAULT_OUTPUT_DIR))
        ttk.Entry(self.file_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5, pady=3)
        ttk.Button(self.file_frame, text='Browse', command=self._choose_output_dir).grid(row=1, column=2, padx=5, pady=3)

    def _build_control_frame(self):
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(fill='x', padx=10, pady=5)

        # Progress Bar - Start in determinate mode
        self.progress = ttk.Progressbar(self.control_frame, mode='determinate', maximum=100, value=0)
        self.progress.pack(fill='x', expand=True, side='left', padx=5)
        
        # Progress Label
        self.progress_label = ttk.Label(self.control_frame, text="Ready")
        self.progress_label.pack(side='left', padx=5)

        # Button frame for run and stop buttons
        btn_frame = ttk.Frame(self.control_frame)
        btn_frame.pack(side='right', padx=5)
        
        # Run button
        self.run_btn = ttk.Button(btn_frame, text='Run Simulation', command=self._on_run_clicked)
        self.run_btn.pack(side='left', padx=2)
        
        # Stop button
        self.stop_btn = ttk.Button(btn_frame, text='Stop', command=self._on_stop_clicked, state='disabled')
        self.stop_btn.pack(side='left', padx=2)

    def _build_status_frame(self):
        self.status_frame = ttk.Labelframe(self, text='Status / Logs')
        self.status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.status_text = scrolledtext.ScrolledText(self.status_frame, height=10, state='disabled', wrap='word')
        self.status_text.pack(fill='both', expand=True)

    # --------------------------- UTILITIES --------------------------- #
    def _load_default_params(self):
        try:
            with open(self.DEFAULT_PARAMS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            messagebox.showerror('Error', f"Default params file not found: {self.DEFAULT_PARAMS_PATH}")
            return {}
        except json.JSONDecodeError as e:
            messagebox.showerror('Error', f"Failed to parse default params: {e}")
            return {}

    def _populate_param_entries(self, params):
        # Filter out arc_fraction and modify time parameters
        filtered_params = {}
        for key, val in params.items():
            if key == 'arc_fraction':
                continue  # Skip arc_fraction
            elif key in ['accel_time', 'const_time']:
                continue  # Skip these, they'll be auto-calculated
            else:
                filtered_params[key] = val
        
        row = 0
        for key, val in filtered_params.items():
            # Use user-friendly label if available, otherwise use original key
            label_text = self.PARAM_LABELS.get(key, key.replace('_', ' ').title())
            ttk.Label(self.param_inner, text=label_text).grid(row=row, column=0, sticky='e', padx=5, pady=2)
            var = tk.StringVar(value=str(val))
            entry = ttk.Entry(self.param_inner, textvariable=var, width=20)
            entry.grid(row=row, column=1, sticky='w', padx=5, pady=2)
            self.param_vars[key] = (var, type(val))
            row += 1

    def _choose_config_file(self):
        path = filedialog.askopenfilename(title='Select config.yaml', filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')])
        if path:
            self.config_path_var.set(path)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title='Select output directory')
        if path:
            self.output_dir_var.set(path)
    
    def _edit_config_file(self):
        """Open config editor window"""
        config_path = self.config_path_var.get()
        if not os.path.exists(config_path):
            messagebox.showerror('Error', f'Config file not found: {config_path}')
            return
        
        ConfigEditorWindow(self, config_path)

    def _validate_and_collect_params(self):
        params = {}
        for key, (var, expected_type) in self.param_vars.items():
            raw_val = var.get()
            try:
                if expected_type is bool:
                    params[key] = raw_val.lower() in ('1', 'true', 'yes', 'y')
                elif expected_type is int:
                    params[key] = int(float(raw_val))  # handle 3.0 → 3
                elif expected_type is float:
                    params[key] = float(raw_val)
                else:
                    params[key] = raw_val
            except ValueError:
                raise ValueError(f"Invalid value for {key}: {raw_val}")
        
        # Auto-calculate accel_time and const_time based on total_time
        if 'total_time' in params:
            params['accel_time'] = 0
            params['const_time'] = params['total_time']
        
        return params

    def _on_run_clicked(self):
        try:
            params = self._validate_and_collect_params()
        except ValueError as e:
            messagebox.showerror('Input Error', str(e))
            return

        config_path = self.config_path_var.get()
        output_dir = self.output_dir_var.get()

        # Set simulation state
        self.simulation_running = True
        self.stop_requested = False
        
        # Update UI state
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.configure(value=0)
        self.progress_label.config(text="Starting...")
        self._log('Simulation started...')

        # Start simulation thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(params, config_path, output_dir),
            daemon=True
        )
        self.simulation_thread.start()

    def _run_simulation_thread(self, params, config_path, output_dir):
        try:
            # Capture stdout to monitor progress
            old_stdout = sys.stdout
            
            class ProgressCapture:
                def __init__(self, stop_check_callback):
                    self.stop_check_callback = stop_check_callback
                    self.original_stdout = old_stdout

                def write(self, text):
                    self.original_stdout.write(text)

                    # 如果用户请求停止，抛出异常以中断仿真
                    if self.stop_check_callback():
                        raise KeyboardInterrupt("Simulation stopped by user")

                    return len(text)

                def flush(self):
                    self.original_stdout.flush()
            
            def update_progress(percentage):
                self.after(0, self._update_progress_bar, percentage)
            
            def check_stop():
                return self.stop_requested
            
            sys.stdout = ProgressCapture(check_stop)
            
            try:
                generated_files = run_single_simulation(params=params, config_path=config_path, output_dir=output_dir, progress_callback=update_progress)
                if self.stop_requested:
                    self.after(0, self._on_simulation_stopped)
                else:
                    self.after(0, self._on_simulation_finished, generated_files)
            finally:
                sys.stdout = old_stdout
                
        except KeyboardInterrupt:
            sys.stdout = old_stdout
            self.after(0, self._on_simulation_stopped)
        except Exception as e:
            sys.stdout = old_stdout
            self.after(0, self._on_simulation_failed, e)
    
    def _update_progress_bar(self, percentage):
        """Update progress bar with percentage"""
        if not self.stop_requested:  # Only update if not stopped
            self.progress.configure(value=percentage)
            self.progress_label.config(text=f"{percentage}%")

    def _on_simulation_finished(self, files):
        # Reset simulation state
        self.simulation_running = False
        self.simulation_thread = None
        
        # Update UI state
        self.progress.configure(value=100)
        self.progress_label.config(text="Completed")
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self._log('Simulation completed.')
        if files:
            self._log('Generated files:')
            for f in files:
                self._log(f'  - {f}')
            # Show results window
            ResultsWindow(self, files)
        else:
            self._log('No files generated (simulation may have failed silently).')

    def _on_simulation_failed(self, error):
        # Reset simulation state
        self.simulation_running = False
        self.simulation_thread = None
        
        # Update UI state
        self.progress_label.config(text="Failed")
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        messagebox.showerror('Simulation Error', str(error))
        self._log(f'Error: {error}')
    
    def _on_stop_clicked(self):
        """Handle stop button click"""
        if self.simulation_running:
            self.stop_requested = True
            self.progress_label.config(text="Stopping...")
            self._log('Stop requested - simulation will terminate...')
            
            # Note: This is a graceful stop request. The actual stopping
            # depends on the simulation checking the stop_requested flag
    
    def _on_simulation_stopped(self):
        """Handle simulation stopped by user"""
        # Reset simulation state
        self.simulation_running = False
        self.simulation_thread = None
        self.stop_requested = False
        
        # Update UI state
        self.progress_label.config(text="Stopped")
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self._log('Simulation stopped by user.')

    def _log(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert('end', message + '\n')
        self.status_text.see('end')
        self.status_text.configure(state='disabled')


class ConfigEditorWindow(tk.Toplevel):
    """Config.yaml editor window"""
    
    def __init__(self, parent, config_path):
        super().__init__(parent)
        self.parent = parent
        self.config_path = config_path
        self.title('Config Editor')
        self.geometry('600x500')
        self.grab_set()  # Make window modal
        
        # Load config
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load config: {e}')
            self.destroy()
            return
        
        self._build_ui()
        
    def _build_ui(self):
        # Text editor
        text_frame = ttk.Frame(self)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.text_editor = scrolledtext.ScrolledText(text_frame, wrap='none')
        self.text_editor.pack(fill='both', expand=True)
        
        # Load current config content
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.text_editor.insert('1.0', content)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read config file: {e}')
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(btn_frame, text='Save', command=self._save_config).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='Cancel', command=self.destroy).pack(side='left', padx=5)
        
    def _save_config(self):
        try:
            content = self.text_editor.get('1.0', 'end-1c')
            # Validate YAML
            yaml.safe_load(content)
            # Save to file
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            messagebox.showinfo('Success', 'Config saved successfully!')
            self.destroy()
        except yaml.YAMLError as e:
            messagebox.showerror('YAML Error', f'Invalid YAML syntax: {e}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to save config: {e}')


class ResultsWindow(tk.Toplevel):
    """Results viewer window"""
    
    def __init__(self, parent, files):
        super().__init__(parent)
        self.parent = parent
        self.files = files
        self.title('Simulation Results')
        self.geometry('500x400')
        self.grab_set()  # Make window modal
        
        self._build_ui()
        
    def _build_ui(self):
        # Image selection
        select_frame = ttk.Labelframe(self, text='Select Image to View')
        select_frame.pack(fill='x', padx=10, pady=5)
        
        # Find PNG files only
        self.image_files = [f for f in self.files if f.lower().endswith('.png')]
        
        if not self.image_files:
            ttk.Label(select_frame, text='No PNG files found in results.').pack(pady=10)
            return
        
        # Create user-friendly names for the images - only show specific types
        self.image_display_names = {}
        display_names = []
        
        # Define the specific image types we want to show
        allowed_image_types = [
            'trajectory_comparison',
            'wheel_forces_X', 
            'wheel_forces_Y',
            'wheel_forces_Z',
            'wheel_forces',  # This might be Z-axis or combined forces
            'wheel_torques'
        ]
        
        for f in self.image_files:
            filename = os.path.basename(f)
            display_name = None
            
            # Only process files that match our allowed types
            if 'trajectory_comparison' in filename:
                display_name = 'Trajectory Comparison'
            elif 'wheel_forces_X' in filename:
                display_name = 'Wheel Forces (X Axis)'
            elif 'wheel_forces_Y' in filename:
                display_name = 'Wheel Forces (Y Axis)'
            elif 'wheel_forces_Z' in filename:
                display_name = 'Wheel Forces (Z Axis)'
            elif 'wheel_forces' in filename and 'X' not in filename and 'Y' not in filename and 'Z' not in filename:
                display_name = 'Wheel Forces (Z Axis)'  # This is likely the Z-axis forces
            elif 'wheel_torques' in filename:
                display_name = 'Wheel Torques'
            
            # Only add to display list if it's one of our allowed types
            if display_name:
                self.image_display_names[display_name] = f
                display_names.append(display_name)
        
        self.selected_image = tk.StringVar()
        self.image_combo = ttk.Combobox(select_frame, textvariable=self.selected_image, 
                                       values=display_names,
                                       state='readonly')
        self.image_combo.pack(fill='x', padx=10, pady=5)
        self.image_combo.set(display_names[0])  # Select first image
        
        # Buttons
        btn_frame = ttk.Frame(select_frame)
        btn_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(btn_frame, text='View Image', command=self._view_image).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='Open Folder', command=self._open_folder).pack(side='left', padx=5)
        
        # File list
        list_frame = ttk.Labelframe(self, text='All Generated Files')
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.file_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        for file in self.files:
            self.file_listbox.insert('end', os.path.basename(file))
        
        self.file_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Close button
        ttk.Button(self, text='Close', command=self.destroy).pack(pady=10)
        
    def _view_image(self):
        if not self.selected_image.get():
            return
        
        # Find full path using display name mapping
        selected_display_name = self.selected_image.get()
        full_path = self.image_display_names.get(selected_display_name)
        
        if not full_path or not os.path.exists(full_path):
            messagebox.showerror('Error', 'Image file not found')
            return
        
        # Open image viewer window
        ImageViewerWindow(self, full_path)
        
    def _open_folder(self):
        if self.files:
            folder_path = os.path.dirname(self.files[0])
            if os.path.exists(folder_path):
                try:
                    # Open folder in system file manager
                    if os.name == 'nt':  # Windows
                        os.startfile(folder_path)
                    elif os.name == 'posix':  # macOS and Linux
                        subprocess.call(['open', folder_path])  # macOS
                        # subprocess.call(['xdg-open', folder_path])  # Linux
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to open folder: {e}')


class ImageViewerWindow(tk.Toplevel):
    """Image viewer window"""
    
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.parent = parent
        self.image_path = image_path
        self.title(f'Image Viewer - {os.path.basename(image_path)}')
        self.geometry('800x600')
        
        self._build_ui()
        
    def _build_ui(self):
        try:
            # Load and display image
            image = Image.open(self.image_path)
            # Resize image to fit window if needed
            display_size = (750, 550)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image)
            
            # Create canvas for image
            canvas = tk.Canvas(self, bg='white')
            canvas.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Add image to canvas
            canvas.create_image(canvas.winfo_reqwidth()//2, canvas.winfo_reqheight()//2, 
                              anchor='center', image=self.photo)
            
            # Update canvas scroll region
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox('all'))
            
        except Exception as e:
            ttk.Label(self, text=f'Error loading image: {e}').pack(pady=20)
        
        # Close button
        ttk.Button(self, text='Close', command=self.destroy).pack(pady=10)


if __name__ == '__main__':
    app = SimulationGUI()
    app.mainloop() 