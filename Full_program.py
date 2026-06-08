import os
import time
import threading
import numpy as np
import warnings
import yaml
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from glob import glob
from PIL import Image, ImageTk
import matplotlib
from astropy.io import ascii
from PIL import Image, ImageTk
matplotlib.use("Agg")

import FULL_synthesis_def
from param_GUI import WizardApp

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SynthesisGUI:

    def __init__(self, root, yaml_path):
        self.root = root
        self.root.title("Synthesis Runner")
        self.root.geometry("1000x900")
        self.yaml_path = yaml_path

        # Keep references to PhotoImage to avoid GC
        self._photo_refs = []
        # ==========================================
        # Popup image viewer variables
        # ==========================================
        self.analysis_image_path = None
        self.zoom_factor = 1.0
        self.popup_window = None

        # --- YAML Path ---
        tk.Label(root, text="Selected YAML parameter file:").pack(pady=5)
        self.path_entry = tk.Entry(root, width=110)
        self.path_entry.pack(pady=5)
        self.path_entry.insert(0, yaml_path)

        self.run_button = ttk.Button(root, text="Run Synthesis", command=self.start_synthesis)
        self.run_button.pack(pady=5)

        # Status
        self.status_label = tk.Label(root, text="Status: Waiting")
        self.status_label.pack(pady=5)

        # 📝 Log box
        tk.Label(root, text="Processing Log:").pack(pady=2)
        self.log_box = scrolledtext.ScrolledText(root, height=18, state='disabled')
        self.log_box.pack(fill="both", padx=20, pady=5, expand=True)
        # ==========================================
        # Open analysis image button
        # ==========================================
        self.open_image_button = ttk.Button(
            root,
            text="Open Analysis Figure",
            command=self.open_analysis_popup,
            state="disabled"
        )

        self.open_image_button.pack(pady=5)
        

        

        self.log_box.tag_config('running', foreground='orange')
        self.log_box.tag_config('completed', foreground='green')
        self.log_box.tag_config('error', foreground='red')

        self.end_button = tk.Button(
            root, text="END", command=self.close_program,
            bg="red", fg="white", font=("Arial", 12, "bold")
        )
        self.end_button.pack(side="bottom", pady=15)

    # ---------- UI-thread helpers ----------
    def ui(self, func, *args, **kwargs):
        self.root.after(0, lambda: func(*args, **kwargs))


    def set_status(self, text):
        self.ui(self.status_label.config, text=text)
    def open_analysis_popup(self):

        if self.analysis_image_path is None:
            return

        # ==========================================
        # Destroy old popup if exists
        # ==========================================
        if self.popup_window is not None:

            try:
                self.popup_window.destroy()

            except Exception:
                pass

        # ==========================================
        # Open image first
        # ==========================================
        img = Image.open(self.analysis_image_path)

        img_width = img.width
        img_height = img.height

        # ==========================================
        # Screen size
        # ==========================================
        screen_width = self.root.winfo_screenwidth()

        screen_height = self.root.winfo_screenheight()

        # ==========================================
        # Maximum popup size
        # Leave margins around screen
        # ==========================================
        max_width = int(screen_width * 0.85)

        max_height = int(screen_height * 0.85)

        # ==========================================
        # Compute scaling
        # ==========================================
        scale = min(
            max_width / img_width,
            max_height / img_height,
            1.0
        )

        popup_width = int(img_width * scale)

        popup_height = int(img_height * scale)

        # Add space for toolbar/buttons
        popup_height += 80

        # ==========================================
        # Create popup
        # ==========================================
        self.popup_window = tk.Toplevel(self.root)

        self.popup_window.title("Analysis Figure Viewer")

        self.popup_window.geometry(
            f"{popup_width}x{popup_height}"
        )

        self.popup_window.configure(bg="black")

        # ==========================================
        # Control Frame
        # ==========================================
        control_frame = tk.Frame(self.popup_window, bg="white")

        control_frame.pack(fill="x", pady=5)

        ttk.Button(control_frame, text="Zoom In", command=self.zoom_in).pack(side="left", padx=5)

        ttk.Button(control_frame, text="Zoom Out", command=self.zoom_out).pack(side="left", padx=5)

        ttk.Button(control_frame, text="Reset", command=self.reset_zoom).pack(side="left", padx=5)

        # ==========================================
        # Canvas
        # ==========================================
        self.canvas = tk.Canvas(self.popup_window, bg="black", highlightthickness=0)

        hbar = tk.Scrollbar(self.popup_window, orient="horizontal", command=self.canvas.xview)

        vbar = tk.Scrollbar(
            self.popup_window,
            orient="vertical",
            command=self.canvas.yview
        )

        self.canvas.configure(
            xscrollcommand=hbar.set,
            yscrollcommand=vbar.set
        )

        hbar.pack(
            side="bottom",
            fill="x"
        )

        vbar.pack(
            side="right",
            fill="y"
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        # ==========================================
        # Mouse wheel zoom
        # ==========================================
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)

        # Linux support
        self.canvas.bind("<Button-4>", lambda e: self.zoom_in())
        self.canvas.bind("<Button-5>", lambda e: self.zoom_out())

        # ==========================================
        # Initial zoom factor
        # ==========================================
        self.zoom_factor = scale

        self.render_popup_image()

    def render_popup_image(self):

        if self.analysis_image_path is None:
            return

        img = Image.open(self.analysis_image_path)

        width = int(img.width * self.zoom_factor)

        height = int(img.height * self.zoom_factor)

        img = img.resize(
            (width, height),
            Image.LANCZOS
        )

        photo = ImageTk.PhotoImage(img)

        self._photo_refs.clear()

        self._photo_refs.append(photo)

        self.canvas.delete("all")

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=photo
        )

        self.canvas.config(
            scrollregion=(0, 0, width, height)
        )

    def zoom_in(self):

        self.zoom_factor *= 1.2

        self.render_popup_image()


    def zoom_out(self):

        self.zoom_factor /= 1.2

        self.render_popup_image()


    def reset_zoom(self):

        self.zoom_factor = 1.0

        self.render_popup_image()

    def mouse_zoom(self, event):

        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1

        self.render_popup_image()


    # ==========================================================
    # Safe logging
    # ==========================================================
    def safe_log(self, text, tag='completed', text_color=None):

        def _append():

            self.log_box.config(state='normal')

            # -----------------------------------------
            # Optional custom color
            # -----------------------------------------
            if text_color is not None:

                custom_tag = f"custom_{text_color}"

                # Create tag if needed
                if custom_tag not in self.log_box.tag_names():

                    self.log_box.tag_config(
                        custom_tag,
                        foreground=text_color
                    )

                use_tag = custom_tag

            else:

                use_tag = tag

            self.log_box.insert(
                tk.END,
                text + "\n",
                use_tag
            )

            self.log_box.see(tk.END)

            self.log_box.config(state='disabled')

        self.ui(_append)


    # ==========================================================
    # Close program safely
    # ==========================================================
    def close_program(self):

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
               
     

    # ---------- Run pipeline ----------
    def start_synthesis(self):
        yaml_path = self.path_entry.get().strip()
        if not yaml_path or not os.path.exists(yaml_path):
            messagebox.showerror("Error", "Invalid YAML file!")
            return

        # Disable matplotlib plotting in worker thread if supported by your module
        try:
            FULL_synthesis_def.PLOT_IN_WORKER = False
        except Exception:
            pass

        self.run_button.config(state="disabled")
        self.set_status("Status: Running…")
        threading.Thread(target=self.run_synthesis, daemon=True).start()

    def run_synthesis(self):
        self.start_time = time.time()

        # --- Load YAML ---
        try:
            with open(self.yaml_path, 'r') as f:
                parameters = yaml.safe_load(f) or {}
        except Exception as e:
            self.ui(messagebox.showerror, "Error", f"Failed to load YAML:\n{e}")
            self.safe_log(f"Error: Failed to load YAML: {e}", 'error')
            self.ui(self.run_button.config, state="normal")
            return

        # Validate/derive arrays
        try:
            m_detection_arr = np.arange(
                parameters['mag_detectionband_start'],
                parameters['mag_detectionband_final'] + parameters['mag_detectionband_bin'],
                parameters['mag_detectionband_bin'],
            )
        except Exception as e:
            self.safe_log(f"YAML missing or invalid magnitude grid: {e}", 'error')
            self.ui(messagebox.showerror, "Error", f"YAML missing or invalid magnitude grid: {e}")
            self.ui(self.run_button.config, state="normal")
            return

        # Make sure output path exists
        program_path = parameters.get('enzo_path')
        output_path = os.path.join(program_path, "Results", parameters.get('stellar_model'))

        mh_list = parameters.get('mh_list') or [0.0]

        enabled = {
            "Generating Data": bool(parameters.get('redo_mockcatalog', False)),
            "EAZY": bool(parameters.get('redo_EAZY', False)),
            "Color Selection": True,
            "Number Density": bool(parameters.get('Calculate_numdens', False)),
            "Analysis": bool(parameters.get('analysis_program', False)),
        }

        # ---------- Begin pipeline over MH values ----------
        for mh in mh_list:
            # Generating Data (includes extracting fluxes by stellar model if requested)
            if enabled["Generating Data"]:
                try:
                    self.safe_log(f"[Generating Data] Started (MH={mh})", 'running')

                    # Optional pre-step depending on stellar_model
                    stellar_model = parameters.get('stellar_model', '')

                    # If it's a list, take the first item
                    if isinstance(stellar_model, list) and stellar_model:
                        stellar_model = stellar_model[0]

                    # Now clean it
                    stellar_model = str(stellar_model).replace("_", "").strip().lower()
                    
                    if stellar_model == 'elfowl':
                        try:
                            FULL_synthesis_def.sort_flux(parameters)
                        except Exception as e:
                            self.safe_log(f"extracts_flux_elfowl failed: {e}", 'error')

                    elif stellar_model == 'bobcat':
                        try:
                            FULL_synthesis_def.extracts_flux_bobcat(parameters)
                        except Exception as e:
                            self.safe_log(f"extracts_flux_bobcat failed: {e}", 'error')

                    for idx, m_detec_set in enumerate(m_detection_arr):
                        FULL_synthesis_def.generate_mock_catalog(parameters, m_detec_set, output_path, mh)
                    self.safe_log("[Generating Data] Completed", 'completed')
                except Exception as e:
                    self.safe_log(f"[Generating Data] Error: {e}", 'error')
            else:
                self.safe_log("Generating Data: NOT RUN")

            for idx, m_detec_set in enumerate(m_detection_arr):
                output_folder = output_path+f"/{parameters['fields_name']}"+f'/mag_{m_detec_set}'
                # Default EAZY folder path
                eazy_folder = os.path.join(output_folder, "Eazy")
                # EAZY
                if enabled["EAZY"]:
                    try:
                        self.safe_log(f"[EAZY] Started (MH={mh})", 'running')
                        eazy_folder = FULL_synthesis_def.eazy(parameters, m_detec_set, output_folder, mh)
                        self.safe_log(f"[EAZY] at mag {m_detec_set}: Completed", 'completed')
                    except Exception as e:
                        self.safe_log(f"[EAZY] Error: {e}", 'error')
                else:
                    self.safe_log("EAZY: NOT RUN", text_color='Brown')

                # Color Selection
                if enabled["Color Selection"]:
                    try:
                        if idx == 0:  # Only log once per MH, not every magnitude step
                            self.safe_log(f"[Color Selection] Started (MH={mh})", 'running')
                        output_path = os.path.join(program_path, "Results", parameters.get('stellar_model'))
                        save_log = os.path.join(output_path, parameters['fields_name'])
                        os.makedirs(save_log, exist_ok=True)
                        log_file_path = os.path.join(save_log, f"{mh}_color_selection_log.txt")
                        flux_path = f"{output_folder}/{mh}_flux25_move_to_mag{m_detec_set}.dat"
                        photoz_path = f"{eazy_folder}/{mh}_bestfit_{m_detec_set}.dat"
                        color_folder = output_folder+'/color'       
                        os.makedirs(color_folder, exist_ok=True)

                        if os.path.exists(log_file_path):
                            try:
                                os.remove(log_file_path)
                            except OSError:
                                pass

                        try:
                            crit = str(parameters.get('color_criteria', '')).lower()
                            if crit == 'borsani_2022':
                                z_cri = FULL_synthesis_def.color_borsani(parameters)
                                
                            elif crit == 'bouwens_2015':
                                z_cri = FULL_synthesis_def.color_bouwen(parameters)
        
                            else:
                                self.safe_log(f"Unknown color_criteria '{crit}', skipping this step.", 'error')
                                
                            FULL_synthesis_def.color_selection(z_cri, parameters, flux_path, m_detec_set, 
                                                output_folder, mh, log_file_path, photoz_path)

                        except Exception as e:
                            self.safe_log(f"Error in color selection (Mag {m_detec_set}): {e}", 'error')
                        self.safe_log(f"[Color Selection] at mag {m_detec_set}: Completed", 'completed')
                    except Exception as e:
                        self.safe_log(f"[Color Selection] Error: {e}", 'error')
                else:
                    self.safe_log("Color Selection: NOT RUN")

            # Number Density
            if enabled["Number Density"]:
                try:
                    self.safe_log("[Number Density] Started", 'running')
                    output_path = os.path.join(program_path, "Results", parameters.get('stellar_model'))
                    output_path2 = os.path.join(output_path, parameters['fields_name'])
                    os.makedirs(output_path2, exist_ok=True)

                    flux_ori_path = f"{output_path}/{parameters['fields_name']}/flux_zp25_{parameters['filters_program']}_m{mh}.dat"            
                    flux_ori_table = ascii.read(flux_ori_path)
                    FULL_synthesis_def.distance_original(flux_ori_table, parameters)
                    FULL_synthesis_def.number_density(m_detection_arr, parameters, output_path2)
                    self.safe_log("[Number Density] Completed", 'completed')

                except Exception as e:
                    self.safe_log(f"[Number Density] Error: {e}", 'error')
            else:
                self.safe_log("Number Density: NOT RUN")

            # Analysis
            analysis_path = None

            if enabled["Analysis"]:

                try:

                    self.safe_log("[Analysis] Started", 'running')
                    output_path = os.path.join(program_path, "Results", parameters.get('stellar_model'))
                    output_path2 = os.path.join( output_path, parameters['fields_name'])

                    analysis_path = os.path.join(output_path2, 'Analysis')

                    os.makedirs(analysis_path, exist_ok=True)

                    # ==========================================
                    # RUN ANALYSIS
                    # ==========================================
                    result = FULL_synthesis_def.analysis(
                        parameters,
                        m_detection_arr,
                        mh,
                        output_path2
                    )

                    # ==========================================
                    # DISPLAY RESULT
                    # ==========================================
                    if result is not None:

                        image_path, summary_message = result

                        self.analysis_image_path = image_path

                        self.ui(
                            self.open_image_button.config,
                            state="normal"
                        )

                        # Automatically open popup
                        self.ui(self.open_analysis_popup)

                    self.safe_log("[Analysis] Completed", 'completed')

                except Exception as e:

                    self.safe_log(
                        f"[Analysis] Error: {e}",
                        'error'
                    )

            else:
                self.safe_log("Analysis: NOT RUN")

        self.set_status("All stages completed")
        self.safe_log("Synthesis finished successfully! \n Open the analysis results to view the output.", 'completed')


# ---------------- Launcher ----------------
if __name__ == "__main__":
    wizard_root = tk.Tk()
    wizard_root.geometry("800x650")
    wizard = WizardApp(wizard_root)
    wizard_root.mainloop()

    if hasattr(wizard, 'selected_yaml_path') and wizard.selected_yaml_path:
        yaml_path = wizard.selected_yaml_path
        root = tk.Tk()
        app = SynthesisGUI(root, yaml_path)
        root.mainloop()
    else:
        print("No YAML selected. Exiting.")