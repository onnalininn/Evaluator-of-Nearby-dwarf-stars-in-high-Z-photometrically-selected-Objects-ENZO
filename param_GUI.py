import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
import re

# =========================================================
# Scrollable Frame
# =========================================================
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=canvas.yview
        )

        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas_window = canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )

        def resize_frame(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", resize_frame)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel support
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):

            canvas.bind_all("<MouseWheel>", _on_mousewheel)

            canvas.bind_all(
                "<Button-4>",
                lambda ev: canvas.yview_scroll(-1, "units")
            )

            canvas.bind_all(
                "<Button-5>",
                lambda ev: canvas.yview_scroll(1, "units")
            )


        def _unbind_mousewheel(event):

            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")


        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)


# =========================================================
# Wizard App
# =========================================================
class WizardApp:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "ENZO : Evaluator of Nearby dwarf stars in high-Z photometrically selected Objects"
        )

        self.root.geometry("1000x750")
        self.root.minsize(900, 650)

        # =================================================
        # VARIABLES
        # =================================================
        self.frames = []
        self.current_frame = 0

        self.start_mode = None
        self.yaml_path = None
        self.yaml_data = {}

        self.selected_yaml_path = None
        self.yaml_text_widget = None

        self.indent = 30

        # =================================================
        # HEADER
        # =================================================
        self.header_frame = tk.Frame(
            self.root,
            bg="#1E3A5F",
            height=70
        )
        self.header_frame.pack(side="top", fill="x")

        self.header_title = tk.Label(
            self.header_frame,
            text="ENZO : Evaluator of Nearby dwarf stars in high-Z photometrically selected Objects",
            font=("Arial", 18, "bold"),
            fg="white",
            bg="#1E3A5F"
        )

        self.header_title.pack(pady=15)

        # =================================================
        # PAGE CONTAINER
        # =================================================
        self.page_container = tk.Frame(self.root)
        self.page_container.pack(fill="both", expand=True)

        # =================================================
        # VARIABLES FOR PAGE 4
        # =================================================
        self.redo_mockcatalog_manual = tk.BooleanVar(value=False)
        self.redo_EAZY = tk.BooleanVar(value=False)
        # EAZY percenties selection
        self.eazy_percenties = tk.StringVar(
            value="35% : P(z>xx) > 0.65"
        )
        
        # Template selections
        self.galaxy_template_var = tk.StringVar(value="v1p3")
        self.stellar_template_var = tk.StringVar(value="spex")
        # Color criteria selections
        self.color_criteria_var = tk.StringVar(value="Bouwens_2015")
        self.criteria_set_var = tk.StringVar(value="set_b")

        self.calc_numdens = tk.BooleanVar(value=True)
        self.analysis_program = tk.BooleanVar(value=True)

        # =================================================
        # CREATE PAGES
        # =================================================
        self.create_yaml_check_frame()       # Page 0
        self.create_survey_frame()           # Page 1
        self.create_filter_frame()           # Page 2
        self.create_mag_filepath_frame()     # Page 3
        self.create_manual_frame()           # Page 4
        self.create_yaml_editor_frame()

        self.show_frame(0)

    # =====================================================
    # Placeholder
    # =====================================================
    @staticmethod
    def get_entry_value(entry):

        value = entry.get().strip()

        if not value:
            raise ValueError(
                "A required field is empty."
            )

        return value
    
    def add_placeholder(self, entry, placeholder):

        entry.insert(0, placeholder)
        entry.config(fg="gray")

        def on_focus_in(event):
            if entry.get() == placeholder:
                entry.delete(0, tk.END)
                entry.config(fg="black")

        def on_focus_out(event):
            if entry.get() == "":
                entry.insert(0, placeholder)
                entry.config(fg="gray")

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
    
    def clean_entry(self, entry):
        value = entry.get().strip()

        if entry.cget("fg") == "gray":
            return ""

        return value

    # =====================================================
    # Navigation buttons
    # =====================================================
    def add_nav_buttons(self, frame, back_cmd=None, next_cmd=None):

        nav_frame = tk.Frame(frame)
        nav_frame.pack(pady=15)

        if back_cmd:
            tk.Button(
                nav_frame,
                text="Back",
                width=12,
                command=back_cmd
            ).pack(side="left", padx=5)

        if next_cmd:
            tk.Button(
                nav_frame,
                text="Next",
                width=12,
                command=next_cmd
            ).pack(side="left", padx=5)
    # =====================================================
    # YAML EDITOR PAGE
    # =====================================================
    def reload_yaml(self):

        if not self.yaml_path:
            return

        self.open_yaml_editor(self.yaml_path)

    def create_yaml_editor_frame(self):

        frame = tk.Frame(self.page_container)

        self.yaml_editor_frame = frame

        tk.Label(
            frame,
            text="YAML Editor",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        text_frame = tk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        yscroll = tk.Scrollbar(text_frame)

        xscroll = tk.Scrollbar(
            frame,
            orient="horizontal"
        )

        yscroll.pack(side="right", fill="y")

        self.yaml_text_widget = tk.Text(
                                        text_frame,
                                        wrap="none",
                                        undo=True,
                                        font=("Courier New", 11),
                                        yscrollcommand=yscroll.set,
                                        xscrollcommand=xscroll.set
                                    )
        yscroll.config(
            command=self.yaml_text_widget.yview
        )

        xscroll.config(
            command=self.yaml_text_widget.xview
        )

        xscroll.pack(
            fill="x"
        )

        self.yaml_text_widget.pack(
            side="left",
            fill="both",
            expand=True
        )

        yscroll.config(
            command=self.yaml_text_widget.yview
        )

        # -------------------------------------------------
        # Buttons
        # -------------------------------------------------
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame,
            text="Save YAML",
            width=15,
            command=self.save_yaml_editor
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Reload",
            width=15,
            command=self.reload_yaml
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Save & Run Synthesis",
            width=20,
            command=self.save_and_run_synthesis
        ).pack(side="left", padx=5)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=15,
            command=lambda: self.show_frame(
                self.frames.index(self.page0_frame)
            )
        ).pack(side="left", padx=5)

        self.frames.append(frame)

    def open_yaml_editor(self, filepath):

        try:

            with open(filepath, "r") as f:
                content = f.read()

            self.yaml_text_widget.delete(
                "1.0",
                tk.END
            )

            self.yaml_text_widget.insert(
                "1.0",
                content
            )

            self.show_frame(
                self.frames.index(
                    self.yaml_editor_frame
                )
            )
            self.root.title(
                f"Editing: {os.path.basename(filepath)}"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Cannot open YAML:\n{e}"
            )
    def save_yaml_editor(self):

        try:

            content = self.yaml_text_widget.get(
                "1.0",
                tk.END
            )

            try:
                yaml.safe_load(content)
            except yaml.YAMLError as e:

                messagebox.showerror(
                    "YAML Syntax Error",
                    str(e)
                )
                return

            with open(self.yaml_path, "w") as f:

                f.write(content)

            messagebox.showinfo(
                "Saved",
                "YAML file saved successfully."
            )

        except Exception as e:

            messagebox.showerror(
                "YAML Error",
                str(e)
            )

    def save_and_run_synthesis(self):
        
        try:

            content = self.yaml_text_widget.get(
                "1.0",
                tk.END
            )

            yaml.safe_load(content)

            confirm = messagebox.askyesno(
                "Run Synthesis",
                "Save YAML and start synthesis?"
            )

            if not confirm:
                return

            with open(self.yaml_path, "w") as f:
                f.write(content)

            print(
                self.yaml_path,
                flush=True
            )

            self.selected_yaml_path = self.yaml_path

            self.root.destroy()

        except Exception as e:

            messagebox.showerror(
                "YAML Error",
                str(e)
            )
    # =====================================================
    # Show frame
    # =====================================================
    def show_frame(self, index):

        if index != self.frames.index(self.yaml_editor_frame):
            self.root.title(
                "ENZO : Evaluator of Nearby dwarf stars in high-Z photometrically selected Objects"
            )

        if 0 <= self.current_frame < len(self.frames):
            self.frames[self.current_frame].pack_forget()

        self.current_frame = index

        self.frames[self.current_frame].pack(
            fill="both",
            expand=True
        )

    # =====================================================
    # PAGE 0
    # =====================================================
    def create_yaml_check_frame(self):

        frame = tk.Frame(self.page_container)

        self.page0_frame = frame

        tk.Label(
            frame,
            text="Do you already have a YAML file?",
            font=("Arial", 14, "bold")
        ).pack(pady=40)

        button_frame = tk.Frame(frame)
        button_frame.pack(pady=20)

        # -------------------------------------------------
        # YES
        # -------------------------------------------------
        def yes_have_yaml():

            file_path = filedialog.askopenfilename(
                title="Select YAML file",
                filetypes=[
                    ("YAML files", "*.yaml *.yml"),
                    ("All files", "*.*")
                ]
            )

            if not file_path:
                return

            self.yaml_path = file_path

            use_direct = messagebox.askyesno(
                "Use Existing YAML",
                "Use this YAML directly without changes?"
            )

            # =============================================
            # FLOW 1
            # =============================================
            if use_direct:

                self.selected_yaml_path = self.yaml_path

                print(self.yaml_path, flush=True)

                self.root.destroy()

                return

            # =============================================
            # FLOW 2
            # =============================================
            else:

                self.open_yaml_editor(
                    self.yaml_path)

        # -------------------------------------------------
        # NO
        # -------------------------------------------------
        def no_yaml():

            self.start_mode = "wizard"

            self.show_frame(
                self.frames.index(self.page1_frame)
            )

        tk.Button(
            button_frame,
            text="Yes",
            width=15,
            command=yes_have_yaml
        ).pack(side="left", padx=10)

        tk.Button(
            button_frame,
            text="No",
            width=15,
            command=no_yaml
        ).pack(side="left", padx=10)

        self.frames.append(frame)

    # =====================================================
    # PAGE 1
    # =====================================================
    def create_survey_frame(self):

        frame = tk.Frame(self.page_container)

        self.page1_frame = frame

        tk.Label(
            frame,
            text="1. General Survey Parameters",
            font=("Arial", 13, "bold")
        ).pack(pady=10)

        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)

        container = scroll.scrollable_frame

        fields = [
            ("Field Name:", "BORG_0314-6712", "field_entry"),
            ("RA (field center):", "48.4335", "ra_entry"),
            ("DEC (field center):", "-67.2032", "dec_entry"),
            ("Solid Angle [value, unit]:", "4.1,arcmin2", "solid_entry"),
            ("Filter Program:", "HST", "filter_prog_entry"),
            ("Stellar Evolution Model:", "Elf_Owl, Bobcat", "stellar_model_entry"),
            ("ENZO path:",
             "/Users/.../ENZO program",
             "enzo_path_entry"),
            ("MH List:",
             "+0.0,+0.5,-0.5,+0.7,+1.0,-1.0",
             "mh_entry"),
        ]

        for label, placeholder, attr in fields:

            tk.Label(
                container,
                text=label
            ).pack(anchor="w", padx=self.indent, pady=2)

            entry = tk.Entry(container, width=50)

            self.add_placeholder(entry, placeholder)

            entry.pack(anchor="w", padx=self.indent, pady=2)

            setattr(self, attr, entry)

        self.add_nav_buttons(
            frame,
            back_cmd=lambda: self.show_frame(
                self.frames.index(self.page0_frame)
            ),
            next_cmd=lambda: self.show_frame(
                self.frames.index(self.page2_frame)
            )
        )

        self.frames.append(frame)

    # =====================================================
    # PAGE 2
    # =====================================================
    def create_filter_frame(self):

        frame = tk.Frame(self.page_container)

        self.page2_frame = frame

        tk.Label(
            frame,
            text="2. Filter Parameters",
            font=("Arial", 13, "bold")
        ).pack(pady=10)

        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)

        self.filter_container = scroll.scrollable_frame

        header1 = [
            "Filter",
            "Depth",
            "PHOTFLAM",
            "PHOTPLAM",
            "EXPTIME",
            ""
        ]

        header2 = [
            "FxxxX",
            "5 sigma magnitude",
            "erg/s/cm2/A/e-",
            "Angstrom",
            "seconds",
            ""
        ]

        hf = tk.Frame(self.filter_container)
        hf.pack(pady=5)

        for col, text in enumerate(header1):

            tk.Label(
                hf,
                text=text,
                width=12,
                font=("Arial", 10, "bold")
            ).grid(row=0, column=col)

        for col, text in enumerate(header2):

            tk.Label(
                hf,
                text=text,
                width=12,
                fg="gray"
            ).grid(row=1, column=col)

        self.filter_rows = []

        tk.Button(
            self.filter_container,
            text="Add Filter",
            command=self.add_filter_row
        ).pack(pady=10)

        self.add_nav_buttons(
            frame,
            back_cmd=lambda: self.show_frame(
                self.frames.index(self.page1_frame)
            ),
            next_cmd=lambda: self.show_frame(
                self.frames.index(self.page3_frame)
            )
        )

        self.frames.append(frame)
        self.add_filter_row()

    # =====================================================
    # ADD FILTER ROW
    # =====================================================
    def add_filter_row(self, default=False):

        row_frame = tk.Frame(self.filter_container)

        row_frame.pack(pady=2)

        entries = []

        placeholders = [
            ("F125W", 10),
            ("26.3", 8),
            ("2.2483e-20", 12),
            ("12486.06", 12),
            ("92578", 10)
        ]

        for placeholder, width in placeholders:

            e = tk.Entry(row_frame, width=width)

            self.add_placeholder(e, placeholder)

            e.pack(side="left", padx=2)

            entries.append(e)

        tk.Button(
            row_frame,
            text="X",
            bg="#fff9c4",
            fg="red",
            width=3,
            command=lambda: self.delete_filter_row(row_frame)
        ).pack(side="left", padx=2)

        self.filter_rows.append((row_frame, entries))

    # =====================================================
    # DELETE FILTER ROW
    # =====================================================
    def delete_filter_row(self, row_frame):

        row_frame.destroy()

        self.filter_rows = [
            (rf, ent)
            for rf, ent in self.filter_rows
            if rf.winfo_exists()
        ]

    # =====================================================
    # PAGE 3
    # =====================================================
    def create_mag_filepath_frame(self):

        frame = tk.Frame(self.page_container)

        self.page3_frame = frame

        tk.Label(
            frame,
            text="3. Magnitude Setup",
            font=("Arial", 13, "bold")
        ).pack(pady=10)

        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)

        container = scroll.scrollable_frame

        fields = [
            ("Detection Band:", "F160W", "detband_entry"),
            ("Mag Start:", "24.5", "mag_start_entry"),
            ("Mag Final:", "27.5", "mag_final_entry"),
            ("Mag Bin:", "0.5", "mag_bin_entry")
        ]

        for label, placeholder, attr in fields:

            tk.Label(
                container,
                text=label
            ).pack(anchor="w", padx=self.indent, pady=2)

            entry = tk.Entry(container, width=55)

            self.add_placeholder(entry, placeholder)

            entry.pack(anchor="w", padx=self.indent, pady=2)

            setattr(self, attr, entry)

        self.add_nav_buttons(
            frame,
            back_cmd=lambda: self.show_frame(
                self.frames.index(self.page2_frame)
            ),
            next_cmd=lambda: self.show_frame(
                self.frames.index(self.page4_frame)
            )
        )

        self.frames.append(frame)

    # =====================================================
    # PAGE 4
    # =====================================================
    def create_manual_frame(self):

        frame = tk.Frame(self.page_container)

        self.page4_frame = frame

        scroll = ScrollableFrame(frame)
        scroll.pack(fill="both", expand=True)

        container = scroll.scrollable_frame

        self.page4_title = tk.Label(
            container,
            text="4. Manual Setting",
            font=("Arial", 13, "bold")
        )

        self.page4_title.pack(pady=10)

        # =================================================
        # REDO MOCK CATALOG
        # =================================================
        self.redo_mockcatalog_checkbox = tk.Checkbutton(
            container,
            text="Redo Mock Catalog",
            variable=self.redo_mockcatalog_manual
        )

        self.redo_mockcatalog_checkbox.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # =================================================
        # EAZY PART
        # =================================================
        tk.Label(
            container,
            text="#________EAZY PART________#",
            fg="green",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=self.indent, pady=(20, 5))

        # -------------------------------------------------
        # Redo EAZY
        # -------------------------------------------------
        self.redo_EAZY_checkbox = tk.Checkbutton(
            container,
            text="Redo EAZY",
            variable=self.redo_EAZY
        )

        self.redo_EAZY_checkbox.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # -------------------------------------------------
        # Galaxy Template
        # -------------------------------------------------
        tk.Label(
            container,
            text="Galaxy Template:"
        ).pack(anchor="w", padx=self.indent, pady=2)

        galaxy_templates = [
            "v1p3",
            "v1p0",
            "larson",
            "hainline"
        ]

        self.galaxy_template_menu = ttk.Combobox(
            container,
            textvariable=self.galaxy_template_var,
            values=galaxy_templates,
            state="readonly",
            width=35
        )

        self.galaxy_template_menu.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # -------------------------------------------------
        # Stellar Template
        # -------------------------------------------------
        tk.Label(
            container,
            text="Stellar Template:"
        ).pack(anchor="w", padx=self.indent, pady=2)

        stellar_templates = [
            "spex"
        ]

        self.stellar_template_menu = ttk.Combobox(
            container,
            textvariable=self.stellar_template_var,
            values=stellar_templates,
            state="readonly",
            width=35
        )

        self.stellar_template_menu.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # -------------------------------------------------
        # EAZY Percenties
        # -------------------------------------------------
        tk.Label(
            container,
            text="EAZY percenties:"
        ).pack(anchor="w", padx=self.indent, pady=2)

        eazy_pct_choices = [
            "20% : P(z>xx) > 0.8",
            "30% : P(z>xx) > 0.7",
            "35% : P(z>xx) > 0.65",
        ]

        self.eazy_percenties_menu = ttk.Combobox(
            container,
            textvariable=self.eazy_percenties,
            values=eazy_pct_choices,
            state="readonly",
            width=35
        )

        self.eazy_percenties_menu.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # =================================================
        # COLOR SELECTION
        # =================================================
        tk.Label(
            container,
            text="#________COLOR SELECTION________#",
            fg="orange",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=self.indent, pady=(20, 5))

        # -------------------------------------------------
        # Color Criteria
        # -------------------------------------------------
        tk.Label(
            container,
            text="Color Criteria:"
        ).pack(anchor="w", padx=self.indent, pady=2)

        criteria_list = [
            "Bouwens_2015",
            "Borsani_2022"
        ]

        self.color_criteria_menu = ttk.Combobox(
            container,
            textvariable=self.color_criteria_var,
            values=criteria_list,
            state="readonly",
            width=30
        )

        self.color_criteria_menu.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # -------------------------------------------------
        # Criteria Set
        # -------------------------------------------------
        tk.Label(
            container,
            text="Criteria Set:"
        ).pack(anchor="w", padx=self.indent, pady=2)

        self.criteria_set_menu = ttk.Combobox(
            container,
            textvariable=self.criteria_set_var,
            values=["set_a", "set_b", "set_c"],
            state="readonly",
            width=30
        )

        self.criteria_set_menu.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )
        # -------------------------------------------------
        # Criteria set explanation
        # -------------------------------------------------
        criteria_info = (
            "Set_a : XDF, HUDF09-Ps, CANDELS-GS+GN\n"
            "Set_b : ERS, BoRG/HIPPIES\n"
            "Set_c : CANDELS-UDS, COSMOS, EGS"
        )

        tk.Label(
            container,
            text=criteria_info,
            justify="left",
            fg="gray",
            font=("Arial", 12)
        ).pack(
            anchor="w",
            padx=self.indent + 20,
            pady=(0, 5)
        )


        # -------------------------------------------------
        # Dynamic update for criteria set
        # -------------------------------------------------
        def update_criteria_sets(event=None):

            criteria = self.color_criteria_var.get()

            if criteria == "Bouwens_2015":

                sets = ["set_a", "set_b", "set_c"]

            elif criteria == "Borsani_2022":

                sets = ["set_b"]

            else:

                sets = []

            self.criteria_set_menu["values"] = sets

            if sets:
                self.criteria_set_var.set(sets[0])

        self.color_criteria_menu.bind(
            "<<ComboboxSelected>>",
            update_criteria_sets
        )

        update_criteria_sets()

        # =================================================
        # ANALYSIS PART
        # =================================================
        tk.Label(
            container,
            text="#________ANALYSIS PART________#",
            fg="blue",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=self.indent, pady=(20, 5))

        self.calc_numdens_checkbox = tk.Checkbutton(
            container,
            text="Calculate Number Density (Forced ON)",
            variable=self.calc_numdens,
            state="disabled"
        )

        self.calc_numdens_checkbox.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        self.analysis_program_checkbox = tk.Checkbutton(
            container,
            text="Analysis Program (Forced ON)",
            variable=self.analysis_program,
            state="disabled"
        )

        self.analysis_program_checkbox.pack(
            anchor="w",
            padx=self.indent,
            pady=2
        )

        # =================================================
        # BUTTONS
        # =================================================
        nav_frame = tk.Frame(container)
        nav_frame.pack(pady=20)

        tk.Button(
            nav_frame,
            text="Back",
            width=12,
            command=lambda: self.show_frame(
                self.frames.index(self.page3_frame)
            )
        ).pack(side="left", padx=5)

        tk.Button(
            nav_frame,
            text="Save YAML",
            width=15,
            command=self.save_all
        ).pack(side="left", padx=5)

        self.frames.append(frame)

    # =====================================================
    # PREFILL MANUAL PAGE
    # =====================================================
    def prefill_manual_from_yaml(self, data):

        def get(d, k, default=None):
            return d[k] if k in d else default

        self.redo_mockcatalog_manual.set(
            bool(get(data, "redo_mockcatalog", False))
        )

        self.redo_EAZY.set(
            bool(get(data, "redo_EAZY", False))
        )

        # ============================================
        # Templates
        # ============================================
        templates = get(data, "templates", [])

        if isinstance(templates, list):

            if len(templates) >= 1:
                self.galaxy_template_var.set(templates[0])

            if len(templates) >= 2:
                self.stellar_template_var.set(templates[1])

        # ============================================
        # EAZY percenties
        # ============================================
        pct = float(get(data, "eazy_pct", 20))

        if pct == 20:

            self.eazy_percenties.set(
                "20% : P(z>xx) > 0.8"
            )
        elif pct == 30:
            self.eazy_percenties.set(
                "30% : P(z>xx) > 0.7"
            )

        elif pct == 35:

            self.eazy_percenties.set(
                "35% : P(z>xx) > 0.65"
            )

        # ============================================
        # Color criteria
        # ============================================
        self.color_criteria_var.set(
            str(get(data, "color_criteria", "Bouwens_2015"))
        )

        self.criteria_set_var.set(
            str(get(data, "criteria_set", "set_a"))
        )

    # =====================================================
    # SAVE ALL
    # =====================================================
    def save_all(self):

        # =================================================
        # FLOW 3 : CREATE NEW YAML
        # =================================================
        try:
            field_name = self.clean_entry(self.field_entry)

            if not field_name:
                raise ValueError("Field Name is required.")

            field_name = re.sub(
                r'[<>:"/\\|?*\s]+',
                "_",
                field_name
            )
            solid_angle = [
                x.strip()
                for x in self.clean_entry(
                    self.solid_entry
                ).split(',')
            ]

            if len(solid_angle) != 2:
                raise ValueError(
                    "Solid Angle must be 'value,unit'"
                )
            
            try:
                float(solid_angle[0])
            except ValueError:
                raise ValueError(
                    "Solid angle value must be numeric."
                )
            
            ra_text = self.clean_entry(self.ra_entry)
            if not ra_text:
                raise ValueError("RA is required.")
            
            dec_text = self.clean_entry(self.dec_entry)
            if not dec_text:
                raise ValueError("Dec is required.")
            
            mh_list = []

            for x in self.clean_entry(self.mh_entry).split(","):
                x = x.strip()
                if not x:
                    continue

                try:
                    float(x)
                except ValueError:
                    raise ValueError(
                        f"Invalid MH value: {x}"
                    )

                mh_list.append(x)

            data = {
                'fields_name': field_name,

                'RA' : float(ra_text),

                'DEC': float(dec_text),

                'solid_angle': solid_angle,

                'filters_program': self.clean_entry(self.filter_prog_entry),

                'mh_list': mh_list,

                'stellar_model': self.clean_entry(self.stellar_model_entry),

                'enzo_path': self.clean_entry(self.enzo_path_entry),
                
            }
            if not (0 <= data["RA"] <= 360):
                raise ValueError("RA must be between 0 and 360")

            if not (-90 <= data["DEC"] <= 90):
                raise ValueError("DEC must be between -90 and 90")

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Survey parameter error:\n{e}"
            )

            return

        # =================================================
        # FILTERS
        # =================================================
        bands = []

        depths = {}
        PHOTFLAM = {}
        PHOTPLAM = {}
        EXPTIME = {}

        for row_frame, entries in self.filter_rows:

            try:

                fname = self.clean_entry(entries[0])

                if not fname:
                    continue

                if fname in bands:
                    messagebox.showerror(
                        "Error",
                        f"Duplicate filter: {fname}"
                    )
                    return
                
                bands.append(fname)

                depths[fname] = float(
                    self.clean_entry(entries[1])
                )

                PHOTFLAM[fname] = float(
                    self.clean_entry(entries[2])
                )

                PHOTPLAM[fname] = float(
                    self.clean_entry(entries[3])
                )

                EXPTIME[fname] = float(
                    self.clean_entry(entries[4])
                )

            except Exception:

                messagebox.showerror(
                    "Error",
                    f"Invalid filter row: {fname}"
                )

                return
        if len(bands) == 0:

            messagebox.showerror(
                "Error",
                "Please add at least one filter."
            )
            return

        data["bands"] = bands
        data["depths"] = depths
        data["PHOTFLAM"] = PHOTFLAM
        data["PHOTPLAM"] = PHOTPLAM
        data["EXPTIME"] = EXPTIME

        # =================================================
        # MAG SETUP
        # =================================================
        detection_band = self.clean_entry(self.detband_entry)

        if not detection_band:
            messagebox.showerror(
                "Error",
                "Detection band is required."
            )
            return

        if detection_band not in bands:
            messagebox.showerror(
                "Error",
                f"Detection band '{detection_band}' is not in filter list."
            )
            return
        
        data["detection_band"] = detection_band

        try:
            data["mag_detectionband_start"] = float(self.get_entry_value(self.mag_start_entry))

            data["mag_detectionband_final"] = float(self.get_entry_value(self.mag_final_entry))

            data["mag_detectionband_bin"] = float(self.get_entry_value(self.mag_bin_entry))
            if data["mag_detectionband_bin"] <= 0:
                raise ValueError(
                    "Mag Bin must be > 0"
                )

            if data["mag_detectionband_start"] >= data["mag_detectionband_final"]:

                messagebox.showerror(
                    "Error",
                    "Mag Start must be smaller than Mag Final."
                )
                return

        except ValueError as e:
            messagebox.showerror(
                "Error",
                str(e)
            )
            return

        # =================================================
        # OUTPUT
        # =================================================

        field_name = data["fields_name"]

        stellar_models = [
                x.strip()
                for x in self.clean_entry(
                    self.stellar_model_entry
                ).split(",")
                if x.strip()
            ]
        if not stellar_models:
            messagebox.showerror(
                "Error",
                "No stellar model specified."
            )
            return

        stellar_model_name = stellar_models[0].strip()

        base_output = self.clean_entry(self.enzo_path_entry)
        if not os.path.isdir(base_output):
            messagebox.showerror(
                "Error",
                f"ENZO path does not exist:\n{base_output}"
            )
            return

        yaml_folder = os.path.join(
            base_output,
            "Results",
            stellar_model_name,
            field_name
        )
        os.makedirs(yaml_folder, exist_ok=True)

        # =================================================
        # MANUAL PART
        # =================================================
        data["redo_mockcatalog"] = \
            self.redo_mockcatalog_manual.get()

        data["redo_EAZY"] = \
            self.redo_EAZY.get()

        # ============================================
        # Templates
        # ============================================
        data["templates"] = [
            self.galaxy_template_var.get(),
            self.stellar_template_var.get()
        ]
        
        # ============================================
        # EAZY Percenties
        # ============================================
        pct_choice = self.eazy_percenties.get()
        if "0.8" in pct_choice:

            data["eazy_pct"] = 20

        elif "0.7" in pct_choice:
            
            data["eazy_pct"] = 30
        
        elif "0.65" in pct_choice:

            data["eazy_pct"] = 35

        else:

            data["eazy_pct"] = 20

        # ============================================
        # Color criteria
        # ============================================
        data["color_criteria"] = \
            self.color_criteria_var.get()

        data["criteria_set"] = \
            self.criteria_set_var.get()

        # Forced ON
        data["Calculate_numdens"] = True
        data["analysis_program"] = True


        # =================================================
        # SAVE YAML
        # =================================================
        file_path = os.path.join(
            yaml_folder,
            "synthesis_param.yaml"
        )

        try:

            with open(file_path, "w") as f:
                yaml.safe_dump(
                    data,
                    f,
                    sort_keys=False,
                    default_flow_style=False
                )

            messagebox.showinfo(
                "Saved",
                f"Settings saved to:\n{file_path}"
            )

            print(file_path, flush=True)

            self.selected_yaml_path = file_path

            self.root.destroy()

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Failed to save YAML:\n{e}"
            )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    root = tk.Tk()

    app = WizardApp(root)

    root.mainloop()
