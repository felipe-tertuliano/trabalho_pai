################################################################################
# TRABALHO PRÁTICO - PROCESSAMENTO E ANÁLISE DE IMAGENS (PUC MINAS)
#
# GRUPO: [INSIRA OS NOMES/MATRÍCULAS AQUI]
#
# ESPECIFICAÇÕES:
#   - Dataset: Coronal [cite: 51]
#   - Regressor Raso: Regressão Linear [cite: 54]
#   - Classificador Raso: XGBoost [cite: 55]
#   - Modelos Profundos: ResNet50 [cite: 57]
#
# AVISO: Este é um arquivo de template. A lógica principal (marcada com #TODO)
# deve ser implementada pelos alunos.
################################################################################

# --- 1. IMPORTAÇÕES ---
# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
from PIL import Image, ImageTk, ImageOps

# Manipulação de Dados e Imagens
import numpy as np
import pandas as pd
import cv2  # OpenCV
import nibabel as nib  # Para arquivos Nifti
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from skimage import measure  # Para descritores morfológicos

# Machine Learning
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, r2_score
)
from sklearn.linear_model import LinearRegression  # Regressor Raso
import xgboost as xgb  # Classificador Raso

# Deep Learning (TensorFlow com Keras)
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Para evitar problemas de exibição em algumas versões
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

################################################################################
# --- 2. CLASSE PRINCIPAL DA APLICAÇÃO (GUI) ---
################################################################################

class AlzheimerApp:

    def create_menu(self):
        """ Cria a barra de menu superior. """
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Menu Arquivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Carregar CSV", command=self.load_csv)
        file_menu.add_command(label="Carregar Imagem...", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.root.quit)

        # Menu Acessibilidade [cite: 75]
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Acessibilidade", menu=help_menu)
        help_menu.add_command(label="Aumentar Fonte", command=self.increase_font)
        help_menu.add_command(label="Diminuir Fonte", command=self.decrease_font)

        # Menu de Navegação
        nav_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Navegação", menu=nav_menu)
        nav_menu.add_command(label="Carregamento", command=lambda: self.show_tab(0))
        nav_menu.add_command(label="Filtros", command=lambda: self.show_tab(1))
        nav_menu.add_command(label="Segmentação", command=lambda: self.show_tab(2))
        nav_menu.add_command(label="Processamento em Lote", command=lambda: self.show_tab(3))

    def show_tab(self, tab_index):
        """Mostra a aba especificada"""
        self.notebook.select(tab_index)

    def on_tab_changed(self, event):
        """Atualiza o título quando a aba muda"""
        self.notebook.select()

    def setup_tab1(self):
        """Configura a aba de Carregamento"""
        # Frame principal com scroll
        tab1_canvas = tk.Canvas(self.tab1, bg="white")
        scrollbar = ttk.Scrollbar(self.tab1, orient="vertical", command=tab1_canvas.yview)
        
        tab1_frame = ttk.Frame(tab1_canvas)
        tab1_frame.bind(
            "<Configure>",
            lambda e: tab1_canvas.configure(scrollregion=tab1_canvas.bbox("all"))
        )
        
        tab1_canvas.create_window((0, 0), window=tab1_frame, anchor="nw")
        tab1_canvas.configure(yscrollcommand=scrollbar.set)
        
        tab1_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel para scroll
        def _on_mousewheel(event):
            tab1_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        tab1_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Conteúdo da aba 1
        ttk.Label(tab1_frame, text="📁 CARREGAMENTO DE DADOS", 
                font=("Arial", 12, "bold"), foreground="darkblue").pack(pady=10)
        
        # Status do CSV
        self.lbl_csv_status = ttk.Label(tab1_frame, text="CSV não carregado", foreground="red")
        self.lbl_csv_status.pack(pady=5, fill=tk.X)
        
        # Botões de carregamento
        btn_load_csv = ttk.Button(tab1_frame, text="Carregar CSV", command=self.load_csv)
        btn_load_csv.pack(pady=5, fill=tk.X)
        
        btn_load_image = ttk.Button(tab1_frame, text="Carregar Imagem", command=self.load_image)
        btn_load_image.pack(pady=5, fill=tk.X)
        
        # Controles de Zoom
        ttk.Separator(tab1_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        ttk.Label(tab1_frame, text="🔍 CONTROLES DE ZOOM", 
                font=("Arial", 10, "bold"), foreground="darkgreen").pack(pady=5)
        
        zoom_frame = ttk.Frame(tab1_frame)
        zoom_frame.pack(pady=5, fill=tk.X)
        
        btn_reset_original = ttk.Button(zoom_frame, text="↻ Original", 
                                    command=lambda: self.reset_zoom("original"))
        btn_reset_original.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        btn_reset_preprocessed = ttk.Button(zoom_frame, text="↻ Pré-processada", 
                                        command=lambda: self.reset_zoom("preprocessed"))
        btn_reset_preprocessed.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        btn_reset_segmented = ttk.Button(zoom_frame, text="↻ Segmentada", 
                                        command=lambda: self.reset_zoom("segmented"))
        btn_reset_segmented.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Coordenadas do mouse
        ttk.Separator(tab1_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        self.lbl_mouse_coords = ttk.Label(tab1_frame, text="🖱️ Mouse: X: -- | Y: --", 
                                        foreground="darkblue", font=("Courier", 9))
        self.lbl_mouse_coords.pack(pady=5)

    def setup_tab2(self):
        """Configura a aba de Filtros"""
        # Frame principal com scroll
        tab2_canvas = tk.Canvas(self.tab2, bg="white")
        scrollbar = ttk.Scrollbar(self.tab2, orient="vertical", command=tab2_canvas.yview)
        
        tab2_frame = ttk.Frame(tab2_canvas)
        tab2_frame.bind(
            "<Configure>",
            lambda e: tab2_canvas.configure(scrollregion=tab2_canvas.bbox("all"))
        )
        
        tab2_canvas.create_window((0, 0), window=tab2_frame, anchor="nw")
        tab2_canvas.configure(yscrollcommand=scrollbar.set)
        
        tab2_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel para scroll
        def _on_mousewheel(event):
            tab2_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        tab2_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Conteúdo da aba 2 (todo o conteúdo original da SEÇÃO 1: FILTROS)
        ttk.Label(tab2_frame, text="🔧 FILTROS DE PRÉ-PROCESSAMENTO", 
                font=("Arial", 12, "bold"), foreground="darkgreen").pack(pady=10)
        
        ttk.Label(tab2_frame, text="Escolha o filtro a aplicar:", 
                foreground="blue", wraplength=340).pack(pady=(2,0), padx=5)
        
        # Variável para armazenar o filtro selecionado
        self.filter_mode = tk.StringVar(value="otsu_clahe")
        
        # Opções de filtros
        rb_filter_frame = ttk.Frame(tab2_frame)
        rb_filter_frame.pack(pady=5, fill=tk.X)
        
        ttk.Radiobutton(rb_filter_frame, text="Otsu + CLAHE", 
                    variable=self.filter_mode, value="otsu_clahe").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="CLAHE (Equalização)", 
                    variable=self.filter_mode, value="clahe").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Otsu (Binarização)", 
                    variable=self.filter_mode, value="otsu").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Canny (Bordas)", 
                    variable=self.filter_mode, value="canny").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Gaussian Blur", 
                    variable=self.filter_mode, value="gaussian").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Median Filter", 
                    variable=self.filter_mode, value="median").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Bilateral Filter", 
                    variable=self.filter_mode, value="bilateral").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(rb_filter_frame, text="Erosão (Morphology)", 
                    variable=self.filter_mode, value="erosion").pack(anchor=tk.W, padx=20)
        
        # --- Parâmetros de Filtros ---
        ttk.Label(tab2_frame, text="⚙️ Parâmetros do Filtro:", foreground="blue").pack(pady=(10,2))
        
        # Frame com scroll para parâmetros
        params_canvas = tk.Canvas(tab2_frame, height=120, bg="white")
        params_canvas.pack(fill=tk.X, pady=5)
        
        params_frame = ttk.Frame(params_canvas)
        params_canvas.create_window((0, 0), window=params_frame, anchor=tk.NW)
        
        # CLAHE
        ttk.Label(params_frame, text="CLAHE Clip Limit:", font=("Arial", 8)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_clahe_clip = ttk.Label(params_frame, text="2.0", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_clahe_clip.grid(row=0, column=1, padx=5)
        self.slider_clahe_clip = ttk.Scale(params_frame, from_=0.5, to=10.0, orient=tk.HORIZONTAL, 
                                        command=self.update_clahe_clip, length=150)
        self.slider_clahe_clip.set(2.0)
        self.slider_clahe_clip.grid(row=0, column=2, padx=5)
        
        ttk.Label(params_frame, text="CLAHE Grid Size:", font=("Arial", 8)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_clahe_grid = ttk.Label(params_frame, text="8", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_clahe_grid.grid(row=1, column=1, padx=5)
        self.slider_clahe_grid = ttk.Scale(params_frame, from_=4, to=16, orient=tk.HORIZONTAL,
                                        command=self.update_clahe_grid, length=150)
        self.slider_clahe_grid.set(8)
        self.slider_clahe_grid.grid(row=1, column=2, padx=5)
        
        # Gaussian
        ttk.Label(params_frame, text="Gaussian Kernel:", font=("Arial", 8)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_gaussian = ttk.Label(params_frame, text="5", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_gaussian.grid(row=2, column=1, padx=5)
        self.slider_gaussian = ttk.Scale(params_frame, from_=3, to=15, orient=tk.HORIZONTAL,
                                        command=self.update_gaussian, length=150)
        self.slider_gaussian.set(5)
        self.slider_gaussian.grid(row=2, column=2, padx=5)
        
        # Median
        ttk.Label(params_frame, text="Median Kernel:", font=("Arial", 8)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_median = ttk.Label(params_frame, text="5", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_median.grid(row=3, column=1, padx=5)
        self.slider_median = ttk.Scale(params_frame, from_=3, to=15, orient=tk.HORIZONTAL,
                                    command=self.update_median, length=150)
        self.slider_median.set(5)
        self.slider_median.grid(row=3, column=2, padx=5)
        
        # Canny
        ttk.Label(params_frame, text="Canny Low:", font=("Arial", 8)).grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_canny_low = ttk.Label(params_frame, text="50", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_canny_low.grid(row=4, column=1, padx=5)
        self.slider_canny_low = ttk.Scale(params_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        command=self.update_canny_low, length=150)
        self.slider_canny_low.set(50)
        self.slider_canny_low.grid(row=4, column=2, padx=5)
        
        ttk.Label(params_frame, text="Canny High:", font=("Arial", 8)).grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_canny_high = ttk.Label(params_frame, text="150", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_canny_high.grid(row=5, column=1, padx=5)
        self.slider_canny_high = ttk.Scale(params_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                        command=self.update_canny_high, length=150)
        self.slider_canny_high.set(150)
        self.slider_canny_high.grid(row=5, column=2, padx=5)
        
        # Bilateral
        ttk.Label(params_frame, text="Bilateral d:", font=("Arial", 8)).grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_bilateral_d = ttk.Label(params_frame, text="9", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_bilateral_d.grid(row=6, column=1, padx=5)
        self.slider_bilateral_d = ttk.Scale(params_frame, from_=5, to=15, orient=tk.HORIZONTAL,
                                        command=self.update_bilateral_d, length=150)
        self.slider_bilateral_d.set(9)
        self.slider_bilateral_d.grid(row=6, column=2, padx=5)
        
        ttk.Label(params_frame, text="Bilateral Sigma:", font=("Arial", 8)).grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_bilateral_sigma = ttk.Label(params_frame, text="75", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_bilateral_sigma.grid(row=7, column=1, padx=5)
        self.slider_bilateral_sigma = ttk.Scale(params_frame, from_=10, to=150, orient=tk.HORIZONTAL,
                                            command=self.update_bilateral_sigma, length=150)
        self.slider_bilateral_sigma.set(75)
        self.slider_bilateral_sigma.grid(row=7, column=2, padx=5)
        
        # Erosão
        ttk.Label(params_frame, text="Erosão Kernel:", font=("Arial", 8)).grid(row=8, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_erosion_kernel = ttk.Label(params_frame, text="3", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_erosion_kernel.grid(row=8, column=1, padx=5)
        self.slider_erosion_kernel = ttk.Scale(params_frame, from_=3, to=9, orient=tk.HORIZONTAL,
                                            command=self.update_erosion_kernel, length=150)
        self.slider_erosion_kernel.set(3)
        self.slider_erosion_kernel.grid(row=8, column=2, padx=5)
        
        ttk.Label(params_frame, text="Erosão Iterações:", font=("Arial", 8)).grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
        self.lbl_erosion_iterations = ttk.Label(params_frame, text="1", foreground="blue", font=("Arial", 8, "bold"))
        self.lbl_erosion_iterations.grid(row=9, column=1, padx=5)
        self.slider_erosion_iterations = ttk.Scale(params_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                                command=self.update_erosion_iterations, length=150)
        self.slider_erosion_iterations.set(1)
        self.slider_erosion_iterations.grid(row=9, column=2, padx=5)
        
        # --- Botões de Filtros ---
        filter_buttons = ttk.Frame(tab2_frame)
        filter_buttons.pack(pady=5, fill=tk.X)
        
        btn_apply_filter = ttk.Button(filter_buttons, text="✅ Aplicar Filtro", 
                                    command=self.apply_selected_filter)
        btn_apply_filter.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        btn_reset_filters = ttk.Button(filter_buttons, text="↻ Reset", 
                                    command=self.reset_filters)
        btn_reset_filters.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Histórico de filtros
        ttk.Label(tab2_frame, text="📜 Filtros Aplicados:", foreground="blue", font=("Arial", 8)).pack(pady=(5,0))
        
        self.lbl_filter_history = ttk.Label(tab2_frame, text="Nenhum", 
                                            foreground="gray", font=("Arial", 7), wraplength=340)
        self.lbl_filter_history.pack(pady=2)
        
        self.lbl_current_filter = ttk.Label(tab2_frame, text="Status: Original", 
                                            foreground="green", font=("Arial", 8))
        self.lbl_current_filter.pack(pady=2)

    def setup_tab3(self):
        """Configura a aba de Segmentação"""
        # Frame principal com scroll
        tab3_canvas = tk.Canvas(self.tab3, bg="white")
        scrollbar = ttk.Scrollbar(self.tab3, orient="vertical", command=tab3_canvas.yview)
        
        tab3_frame = ttk.Frame(tab3_canvas)
        tab3_frame.bind(
            "<Configure>",
            lambda e: tab3_canvas.configure(scrollregion=tab3_canvas.bbox("all"))
        )
        
        tab3_canvas.create_window((0, 0), window=tab3_frame, anchor="nw")
        tab3_canvas.configure(yscrollcommand=scrollbar.set)
        
        tab3_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel para scroll
        def _on_mousewheel(event):
            tab3_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        tab3_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Conteúdo da aba 3 (todo o conteúdo original da SEÇÃO 2: SEGMENTAÇÃO)
        ttk.Label(tab3_frame, text="✂️ SEGMENTAÇÃO DE VENTRÍCULOS", 
                font=("Arial", 12, "bold"), foreground="darkblue").pack(pady=10)
        
        # Informação sobre qual imagem é usada
        info_frame = ttk.Frame(tab3_frame)
        info_frame.pack(pady=(5,10), fill=tk.X)
        ttk.Label(info_frame, text="ℹ️ A segmentação SEMPRE usa a Janela 2 (Pré-processada)", 
                foreground="darkblue", font=("Arial", 8, "italic"), wraplength=340).pack(padx=5)
        
        ttk.Label(tab3_frame, text="⚙️ Parâmetros de Segmentação:", foreground="blue", font=("Arial", 9, "bold")).pack(pady=(5,5))
        
        # Threshold do Region Growing
        threshold_frame = ttk.Frame(tab3_frame)
        threshold_frame.pack(pady=5, fill=tk.X)
        ttk.Label(threshold_frame, text="Threshold Region Growing:").pack(side=tk.LEFT)
        self.lbl_threshold = ttk.Label(threshold_frame, text="50", foreground="red", font=("Arial", 9, "bold"))
        self.lbl_threshold.pack(side=tk.LEFT, padx=5)
        
        self.slider_threshold = ttk.Scale(tab3_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                        command=self.update_threshold)
        self.slider_threshold.set(50)
        self.slider_threshold.pack(fill=tk.X, padx=10)
        
        # Conectividade do Region Growing
        connectivity_frame = ttk.Frame(tab3_frame)
        connectivity_frame.pack(pady=5, fill=tk.X)
        ttk.Label(connectivity_frame, text="Conectividade:", foreground="blue").pack(side=tk.LEFT, padx=5)
        
        self.connectivity_var = tk.IntVar(value=8)
        ttk.Radiobutton(connectivity_frame, text="4-vizinhos (↑↓←→)", 
                    variable=self.connectivity_var, value=4).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(connectivity_frame, text="8-vizinhos (↑↓←→↖↗↙↘)", 
                    variable=self.connectivity_var, value=8).pack(side=tk.LEFT, padx=5)
        
        # Kernel Morfológico
        kernel_frame = ttk.Frame(tab3_frame)
        kernel_frame.pack(pady=5, fill=tk.X)
        ttk.Label(kernel_frame, text="Kernel Morfológico:").pack(side=tk.LEFT)
        self.lbl_kernel = ttk.Label(kernel_frame, text="15x15", foreground="red", font=("Arial", 9, "bold"))
        self.lbl_kernel.pack(side=tk.LEFT, padx=5)
        
        self.slider_kernel = ttk.Scale(tab3_frame, from_=3, to=25, orient=tk.HORIZONTAL,
                                    command=self.update_kernel)
        self.slider_kernel.set(15)
        self.slider_kernel.pack(fill=tk.X, padx=10)
        
        # Operações Morfológicas
        ttk.Label(tab3_frame, text="Operações Morfológicas:", foreground="blue").pack(pady=(10,5))
        
        self.var_opening = tk.BooleanVar(value=True)
        chk_opening = ttk.Checkbutton(tab3_frame, text="🔹 Abertura (remover ruído)", 
                                    variable=self.var_opening, command=self.update_morphology_flags)
        chk_opening.pack(anchor=tk.W, padx=20)
        
        self.var_closing = tk.BooleanVar(value=False)
        chk_closing = ttk.Checkbutton(tab3_frame, text="🔹 Fechamento (fechar gaps)", 
                                    variable=self.var_closing, command=self.update_morphology_flags)
        chk_closing.pack(anchor=tk.W, padx=20)
        
        self.var_fill_holes = tk.BooleanVar(value=True)
        chk_fill_holes = ttk.Checkbutton(tab3_frame, text="🔹 Preencher buracos", 
                                        variable=self.var_fill_holes, command=self.update_morphology_flags)
        chk_fill_holes.pack(anchor=tk.W, padx=20)
        
        self.var_smooth = tk.BooleanVar(value=True)
        chk_smooth = ttk.Checkbutton(tab3_frame, text="🔹 Suavizar contornos", 
                                    variable=self.var_smooth, command=self.update_morphology_flags)
        chk_smooth.pack(anchor=tk.W, padx=20)
        
        ttk.Separator(tab3_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Opções de Segmentação
        ttk.Label(tab3_frame, text="Métodos de Segmentação:", foreground="blue", 
                font=("Arial", 9, "bold")).pack(pady=(5,5))
        
        # Método de segmentação
        self.segmentation_method = tk.StringVar(value="region_growing")
        
        ttk.Radiobutton(tab3_frame, text="🌱 Region Growing (clique para seed)", 
                    variable=self.segmentation_method, value="region_growing").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(tab3_frame, text="🎯 Watershed", 
                    variable=self.segmentation_method, value="watershed").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(tab3_frame, text="🔲 Thresholding Adaptativo", 
                    variable=self.segmentation_method, value="adaptive_threshold").pack(anchor=tk.W, padx=20)
        ttk.Radiobutton(tab3_frame, text="🧲 K-Means Clustering", 
                    variable=self.segmentation_method, value="kmeans").pack(anchor=tk.W, padx=20)
        
        # Edição de Seeds Fixos
        seeds_edit_frame = ttk.LabelFrame(tab3_frame, text="📍 Seeds Fixos (Editáveis)", padding="5")
        seeds_edit_frame.pack(pady=(10,5), fill=tk.X)
        
        # Lista de seeds
        seeds_list_frame = ttk.Frame(seeds_edit_frame)
        seeds_list_frame.pack(fill=tk.X, pady=2)
        
        self.auto_seeds_listbox = tk.Listbox(seeds_list_frame, height=3, font=("Courier", 9))
        self.auto_seeds_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.update_auto_seeds_display()
        
        # Botões de edição
        seeds_buttons_frame = ttk.Frame(seeds_edit_frame)
        seeds_buttons_frame.pack(fill=tk.X, pady=2)
        
        # Adicionar seed
        add_seed_frame = ttk.Frame(seeds_buttons_frame)
        add_seed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(add_seed_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.auto_seed_x_entry = ttk.Entry(add_seed_frame, width=6)
        self.auto_seed_x_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(add_seed_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.auto_seed_y_entry = ttk.Entry(add_seed_frame, width=6)
        self.auto_seed_y_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(add_seed_frame, text="➕ Adicionar", 
                command=self.add_auto_seed, width=12).pack(side=tk.LEFT, padx=5)
        
        # Remover seed selecionado
        ttk.Button(seeds_buttons_frame, text="➖ Remover Selecionado", 
                command=self.remove_auto_seed).pack(pady=2, fill=tk.X)
        
        # Botões de Segmentação
        ttk.Label(tab3_frame, text="Executar Segmentação:", foreground="blue").pack(pady=(10,2))
        
        btn_segment_auto = ttk.Button(tab3_frame, text="▶ Segmentação Automática (Seeds Fixos)", 
                                    command=self.segment_ventricles)
        btn_segment_auto.pack(pady=2, fill=tk.X)
        
        btn_multi_segment = ttk.Button(tab3_frame, text="🖱️ Modo Multi-Seed (Clique nas Janelas)", 
                                    command=self.toggle_multi_seed_mode)
        btn_multi_segment.pack(pady=2, fill=tk.X)
        
        self.lbl_multi_seed = ttk.Label(tab3_frame, text="Multi-Seed: Inativo", foreground="gray", 
                                        font=("Arial", 8))
        self.lbl_multi_seed.pack(pady=2)
        
        self.lbl_segment_status = ttk.Label(tab3_frame, text="Segmentação: Aguardando...", 
                                            foreground="gray", font=("Arial", 8))
        self.lbl_segment_status.pack(pady=2)

    def setup_tab4(self):
        """Configura a aba de Processamento em Lote"""
        # Frame principal
        tab4_frame = ttk.Frame(self.tab4)
        tab4_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(tab4_frame, text="📁 PROCESSAMENTO EM LOTE", 
                font=("Arial", 12, "bold"), foreground="darkorange").pack(pady=10)
        
        # Botão principal para abrir configuração
        btn_batch_config = ttk.Button(tab4_frame, text="⚙️ Configurar e Processar Pasta", 
                                    command=self.open_batch_config_window)
        btn_batch_config.pack(pady=20, fill=tk.X)
        
        ttk.Label(tab4_frame, text="Configure filtros, seeds e processe múltiplos arquivos", 
                foreground="gray", font=("Arial", 10)).pack(pady=10)
        
        # Informações adicionais
        info_frame = ttk.LabelFrame(tab4_frame, text="ℹ️ Informações", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = """O processamento em lote permite:
        • Processar todas as imagens de uma pasta
        • Aplicar os mesmos filtros e parâmetros
        • Usar seeds fixos ou automáticos
        • Gerar relatório consolidado
        • Exportar características extraídas"""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Arial", 9)).pack(anchor=tk.W)

    def __init__(self, root):
        self.root = root
        self.root.title("Trabalho Prático - Diagnóstico de Alzheimer")
        self.root.geometry("1000x800")
        
        # --- Variáveis de Estado ---
        self.current_font_size = 10
        self.dataframe = None  # Armazena o CSV
        self.image_path = None
        self.original_image = None # Imagem PIL original
        self.preprocessed_image = None # Imagem PIL pré-processada (com filtros)
        self.segmented_image = None # Imagem PIL segmentada (com contorno)
        self.image_mask = None # Máscara (numpy) da segmentação
        self.features_df = None # DataFrame com características extraídas
        self.current_filter = "none" # Filtro atual aplicado
        
        # --- Variáveis de Zoom ---
        self.zoom_level_original = 1.0
        self.zoom_level_preprocessed = 1.0
        self.zoom_level_segmented = 1.0
        self.pan_start_x_original = 0
        self.pan_start_y_original = 0
        self.pan_start_x_preprocessed = 0
        self.pan_start_y_preprocessed = 0
        self.pan_start_x_segmented = 0
        self.pan_start_y_segmented = 0
        self.is_panning_original = False
        self.is_panning_preprocessed = False
        self.is_panning_segmented = False
        
        # --- Variáveis de Region Growing ---
        self.click_moved_original = False
        self.click_moved_preprocessed = False
        self.region_growing_threshold = 50  # Threshold padrão: 50
        self.region_growing_connectivity = 8  # Conectividade: 4 ou 8 (padrão 8)
        self.use_histogram_equalization = True  # FIXO: CLAHE sempre ativado
        self.use_otsu_for_segmentation = False  # Usar Otsu (binarizada) ou CLAHE (escala de cinza)
        self.multi_seed_mode = False  # Modo de múltiplos seeds manual
        self.accumulated_mask = None  # Máscara acumulada de múltiplos cliques
        self.multi_seed_points = []  # Pontos coletados no modo multi-seed
        
        # --- Pós-processamento Morfológico (AJUSTÁVEL) ---
        self.morphology_kernel_size = 15  # Ajustável via slider (3-25)
        self.apply_closing = False  # Ajustável via checkbox
        self.apply_fill_holes = True  # Ajustável via checkbox
        self.apply_opening = True  # Ajustável via checkbox
        self.apply_smooth_contours = True  # Ajustável via checkbox
        
        # --- Variáveis de medição de coordenadas ---
        self.show_coordinates = True  # Mostrar coordenadas do mouse
        self.current_mouse_x = 0
        self.current_mouse_y = 0
        
        # --- Seed points FIXOS para segmentação automática (EDITÁVEIS) ---
        self.auto_seed_points = [
            (158,98),
            (109,124),
            (109,81),
        ]
        
        # Limite de pixels para validação de segmentação (se exceder, considera erro)
        self.max_segmentation_pixels = 50000  # Ajustável - se máscara tiver mais pixels, considera erro
        
        # Método alternativo de segmentação quando Region Growing falha
        self.alternative_segmentation_method = 'roi_fixed'  # Padrão: ROI fixa
        # Opções: 'roi_fixed', 'spatial_mask', 'connected_components', 'centroid_based',
        #         'hole_filling', 'flood_fill', 'distance_transform', 'watershed_markers', 'active_contours'
        
        # --- Sistema de Pipeline de Filtros ---
        self.filter_history = []  # Lista de filtros aplicados
        self.original_image_backup = None  # Backup da imagem original
        self.current_filtered_image = None  # Imagem com filtros aplicados
        
        # --- Sistema de Descritores Morfológicos ---
        self.descriptors_list = []  # Lista para acumular descritores de todas as imagens
        
        # Parâmetros ajustáveis para cada filtro
        self.filter_params = {
            'clahe_clip_limit': 2.0,  # CLAHE: 0.5 - 10.0
            'clahe_grid_size': 8,     # CLAHE: 4 - 16
            'gaussian_kernel': 5,      # Gaussian: 3, 5, 7, 9, 11, 13, 15
            'median_kernel': 5,        # Median: 3, 5, 7, 9, 11
            'bilateral_d': 9,          # Bilateral: 5 - 15
            'bilateral_sigma': 75,     # Bilateral: 10 - 150
            'canny_low': 50,           # Canny: 0 - 255
            'canny_high': 150,         # Canny: 0 - 255
            'erosion_kernel': 3,       # Erosão: 3, 5, 7, 9
            'erosion_iterations': 1,   # Erosão: 1 - 10
        }
        
        # Configura a fonte padrão
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(size=self.current_font_size)
        
        # --- Menu Superior com Abas ---
        self.create_menu()
        
        # --- Layout Principal ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # --- Grid de Imagens ---
        images_container = ttk.Frame(main_frame)
        images_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # LINHA 1: Original e Com Filtro lado a lado
        top_row_frame = ttk.Frame(images_container)
        top_row_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0,3))
        
        # Canvas 1: Original (sem filtro)
        original_frame = ttk.Frame(top_row_frame, relief=tk.RIDGE, padding="2")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,2))
        self.lbl_image_original = ttk.Label(original_frame, text="📷 Original (Sem Filtro)", 
                                            font=("Arial", 9, "bold"), foreground="blue")
        self.lbl_image_original.pack(pady=1)
        self.canvas_original = tk.Canvas(original_frame, bg="gray", width=280, height=250)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Canvas 2: Com Filtro
        preprocessed_frame = ttk.Frame(top_row_frame, relief=tk.RIDGE, padding="2")
        preprocessed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2,0))
        self.lbl_image_preprocessed = ttk.Label(preprocessed_frame, text="🔧 Com Filtro", 
                                                font=("Arial", 9, "bold"), foreground="green")
        self.lbl_image_preprocessed.pack(pady=1)
        self.canvas_preprocessed = tk.Canvas(preprocessed_frame, bg="gray", width=280, height=250)
        self.canvas_preprocessed.pack(fill=tk.BOTH, expand=True)
        
        # LINHA 2: Segmentada (largura completa)
        bottom_row_frame = ttk.Frame(images_container)
        bottom_row_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(3,0))
        
        # Canvas 3: Com Filtro + Segmentada
        segmented_frame = ttk.Frame(bottom_row_frame, relief=tk.RIDGE, padding="2")
        segmented_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.lbl_image_segmented = ttk.Label(segmented_frame, text="✂️ Segmentada (Contorno Vermelho)", 
                                            font=("Arial", 9, "bold"), foreground="red")
        self.lbl_image_segmented.pack(pady=1)
        self.canvas_segmented = tk.Canvas(segmented_frame, bg="gray", height=250)
        self.canvas_segmented.pack(fill=tk.BOTH, expand=True)
        
        # --- Notebook com Abas + Log  ---
        notebook_container = ttk.Frame(main_frame)
        notebook_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Configure grid weights for equal splitting
        notebook_container.grid_rowconfigure(0, weight=1)  # Top row (notebook)
        notebook_container.grid_rowconfigure(1, weight=1)  # Bottom row (log)
        notebook_container.grid_columnconfigure(0, weight=1)

        # --- Notebook (Abas) ---
        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=(5, 2))
        # Bind para detectar mudança de aba
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # --- Aba 1: Carregamento e Visualização ---
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Carregamento")
        self.setup_tab1()

        # --- Aba 2: Filtros ---
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Filtros")
        self.setup_tab2()

        # --- Aba 3: Segmentação ---
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Segmentação")
        self.setup_tab3()

        # --- Aba 4: Processamento em Lote ---
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Processamento em Lote")
        self.setup_tab4()

        # --- Área de Log ---
        log_frame = ttk.LabelFrame(notebook_container, text="Log do Sistema", padding="5")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(2, 5))

        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Barra de scroll para o log
        log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Configura os bindings do sistema depois da criação da interface
        self.root.after(100, self.setup_bindings)

    def log(self, message):
        """ Adiciona uma mensagem ao log na GUI. """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_font_size(self):
        """ Atualiza o tamanho da fonte de todos os widgets. """
        self.default_font.configure(size=self.current_font_size)
        # Você pode precisar iterar sobre widgets específicos se eles não atualizarem
        self.lbl_csv_status.config(font=self.default_font)
        # ... adicionar outros widgets ...

    def increase_font(self):
        self.current_font_size += 2
        self.update_font_size()
        self.log(f"Tamanho da fonte aumentado para {self.current_font_size}")

    def decrease_font(self):
        if self.current_font_size > 6:
            self.current_font_size -= 2
            self.update_font_size()
            self.log(f"Tamanho da fonte diminuído para {self.current_font_size}")


    def track_mouse_position(self, event):
        """Rastreia a posição do mouse na imagem original."""
        if self.original_image is None:
            return
        
        # Converte coordenadas do canvas para coordenadas da imagem
        canvas_w = self.canvas_original.winfo_width()
        canvas_h = self.canvas_original.winfo_height()
        
        img_w = self.original_image.width
        img_h = self.original_image.height
        
        # Calcula o tamanho da imagem exibida considerando o zoom
        display_w = int(img_w * self.zoom_level_original)
        display_h = int(img_h * self.zoom_level_original)
        
        # Ajuste: centralização no canvas
        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2
        
        # Converte para coordenadas da imagem original
        img_x = int((event.x - offset_x) / self.zoom_level_original)
        img_y = int((event.y - offset_y) / self.zoom_level_original)
        
        # Verifica se está dentro da imagem
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            self.current_mouse_x = img_x
            self.current_mouse_y = img_y
            self.lbl_mouse_coords.config(text=f"🖱️ Mouse: X: {img_x:3d} | Y: {img_y:3d}")
        else:
            self.lbl_mouse_coords.config(text="🖱️ Mouse: X: -- | Y: --")

    def track_mouse_position_preprocessed(self, event):
        """Rastreia a posição do mouse na imagem pré-processada."""
        if self.preprocessed_image is None:
            return
        
        # Converte coordenadas do canvas para coordenadas da imagem
        canvas_w = self.canvas_preprocessed.winfo_width()
        canvas_h = self.canvas_preprocessed.winfo_height()
        
        img_w = self.preprocessed_image.width
        img_h = self.preprocessed_image.height
        
        # Calcula o tamanho da imagem exibida considerando o zoom
        display_w = int(img_w * self.zoom_level_preprocessed)
        display_h = int(img_h * self.zoom_level_preprocessed)
        
        # Ajuste: centralização no canvas
        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2
        
        # Converte para coordenadas da imagem
        img_x = int((event.x - offset_x) / self.zoom_level_preprocessed)
        img_y = int((event.y - offset_y) / self.zoom_level_preprocessed)
        
        # Verifica se está dentro da imagem
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            self.lbl_mouse_coords.config(text=f"🖱️ Mouse: X: {img_x:3d} | Y: {img_y:3d} [PRÉ-PROC]")
        else:
            self.lbl_mouse_coords.config(text="🖱️ Mouse: X: -- | Y: --")

    def clear_registered_points(self):
        """Limpa a lista de pontos registrados."""
        self.registered_points = []
        self.lbl_clicked_coords.config(text="Nenhum ponto registrado")
        self.log("📍 Pontos registrados limpos.")

    def export_registered_points(self):
        """Exporta os pontos registrados em formato Python para o log."""
        if not self.registered_points:
            self.log("⚠ Nenhum ponto registrado para exportar.")
            return
        
        self.log("\n" + "="*50)
        self.log("📍 PONTOS REGISTRADOS (formato Python):")
        self.log("="*50)
        self.log(f"# Total de pontos: {len(self.registered_points)}")
        self.log(f"seed_points = [")
        for i, (x, y) in enumerate(self.registered_points):
            self.log(f"    ({x}, {y}),  # Ponto {i+1}")
        self.log("]")
        self.log("="*50 + "\n")
        
        messagebox.showinfo("Pontos Exportados", 
                           f"{len(self.registered_points)} pontos exportados para o log!\n\n"
                           "Copie do log para usar na segmentação automática.")

    def start_multi_seed(self):
        """Inicia o modo de múltiplos seeds manual."""
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        
        self.multi_seed_mode = True
        self.multi_seed_points = []
        self.accumulated_mask = None
        self.lbl_multi_seed.config(text="🎯 Multi-Seed ATIVO - Clique nos ventrículos", foreground="purple")
        self.log("🎯 Modo Multi-Seed ativado. Clique em múltiplos pontos dos ventrículos.")

    def finish_multi_seed(self):
        """Finaliza o modo multi-seed e executa segmentação com os pontos coletados."""
        if not self.multi_seed_mode:
            messagebox.showwarning("Aviso", "Modo Multi-Seed não está ativo.")
            return
        
        if not self.multi_seed_points:
            messagebox.showwarning("Aviso", "Nenhum ponto coletado no modo Multi-Seed.")
            self.multi_seed_mode = False
            self.lbl_multi_seed.config(text="")
            return
        
        self.multi_seed_mode = False
        self.lbl_multi_seed.config(text=f"✓ Multi-Seed finalizado: {len(self.multi_seed_points)} pontos", 
                                   foreground="green")
        
        # A máscara já foi acumulada durante os cliques
        self.log(f"✓ Modo Multi-Seed finalizado com {len(self.multi_seed_points)} pontos.")
        
        # Atualiza status
        if self.accumulated_mask is not None:
            num_pixels = np.sum(self.accumulated_mask == 255)
            self.lbl_segment_status.config(
                text=f"✓ Multi-Seed: {len(self.multi_seed_points)} seeds | {num_pixels} pixels",
                foreground="green"
            )

    def export_multi_seed_points(self):
        """Exporta os pontos coletados no modo Multi-Seed."""
        if not self.multi_seed_points:
            self.log("⚠ Nenhum ponto Multi-Seed para exportar.")
            messagebox.showwarning("Aviso", "Nenhum ponto Multi-Seed coletado.")
            return
        
        self.log("\n" + "="*60)
        self.log("🎯 PONTOS MULTI-SEED (formato Python):")
        self.log("="*60)
        self.log(f"# Total de pontos: {len(self.multi_seed_points)}")
        self.log(f"auto_seed_points = [")
        for i, (x, y) in enumerate(self.multi_seed_points):
            self.log(f"    ({x}, {y}),  # Ponto {i+1}")
        self.log("]")
        self.log("="*60 + "\n")
        
        messagebox.showinfo("Pontos Multi-Seed Exportados", 
                           f"{len(self.multi_seed_points)} pontos exportados para o log!\n\n"
                           "Copie do log e cole em self.auto_seed_points no código.")

    # --- 3. FUNÇÕES DE CARREGAMENTO E EXIBIÇÃO ---

    def setup_bindings(self):
        """Configura os bindings do sistema."""
        # Imagem original
        self.canvas_original.bind("<MouseWheel>", lambda e: self.zoom_image(e, "original"))
        self.canvas_original.bind("<Button-4>", lambda e: self.zoom_image(e, "original"))  # Linux zoom in
        self.canvas_original.bind("<Button-5>", lambda e: self.zoom_image(e, "original"))  # Linux zoom out
        self.canvas_original.bind("<ButtonPress-1>", lambda e: self.start_pan(e, "original"))
        self.canvas_original.bind("<B1-Motion>", lambda e: self.pan_image(e, "original"))
        self.canvas_original.bind("<ButtonRelease-1>", lambda e: self.stop_pan(e, "original"))
        self.canvas_original.bind("<Motion>", self.track_mouse_position)  # Rastrear posição do mouse
        
        # Imagem preprocessed
        self.canvas_preprocessed.bind("<MouseWheel>", lambda e: self.zoom_image(e, "preprocessed"))
        self.canvas_preprocessed.bind("<Button-4>", lambda e: self.zoom_image(e, "preprocessed"))
        self.canvas_preprocessed.bind("<Button-5>", lambda e: self.zoom_image(e, "preprocessed"))
        self.canvas_preprocessed.bind("<ButtonPress-1>", lambda e: self.start_pan(e, "preprocessed"))
        self.canvas_preprocessed.bind("<B1-Motion>", lambda e: self.pan_image(e, "preprocessed"))
        self.canvas_preprocessed.bind("<ButtonRelease-1>", lambda e: self.stop_pan(e, "preprocessed"))
        self.canvas_preprocessed.bind("<Motion>", self.track_mouse_position_preprocessed)  # Rastrear posição
        
        # Imagem segmented
        self.canvas_segmented.bind("<MouseWheel>", lambda e: self.zoom_image(e, "segmented"))
        self.canvas_segmented.bind("<Button-4>", lambda e: self.zoom_image(e, "segmented"))
        self.canvas_segmented.bind("<Button-5>", lambda e: self.zoom_image(e, "segmented"))
        self.canvas_segmented.bind("<ButtonPress-1>", lambda e: self.start_pan(e, "segmented"))
        self.canvas_segmented.bind("<B1-Motion>", lambda e: self.pan_image(e, "segmented"))
        self.canvas_segmented.bind("<ButtonRelease-1>", lambda e: self.stop_pan(e, "segmented"))
    
    def update_threshold(self, value):
        """Atualiza o threshold do region growing"""
        self.region_growing_threshold = int(float(value))
        self.lbl_threshold.config(text=str(self.region_growing_threshold))
    
    def update_kernel(self, value):
        """Atualiza o tamanho do kernel morfológico"""
        kernel_size = int(float(value))
        if kernel_size % 2 == 0:  # Garantir que seja ímpar
            kernel_size += 1
        self.morphology_kernel_size = kernel_size
        self.lbl_kernel.config(text=f"{kernel_size}x{kernel_size}")
    
    def update_morphology_flags(self):
        """Atualiza as flags de operações morfológicas"""
        self.apply_opening = self.var_opening.get()
        self.apply_closing = self.var_closing.get()
        self.apply_fill_holes = self.var_fill_holes.get()
        self.apply_smooth_contours = self.var_smooth.get()
    
    def update_auto_seeds_display(self):
        """Atualiza a lista visual de seeds fixos."""
        self.auto_seeds_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.auto_seed_points, 1):
            self.auto_seeds_listbox.insert(tk.END, f"{i}. ({x}, {y})")
    
    def add_auto_seed(self):
        """Adiciona um novo seed fixo."""
        try:
            x = int(self.auto_seed_x_entry.get())
            y = int(self.auto_seed_y_entry.get())
            self.auto_seed_points.append((x, y))
            self.update_auto_seeds_display()
            self.auto_seed_x_entry.delete(0, tk.END)
            self.auto_seed_y_entry.delete(0, tk.END)
            self.log(f"✅ Seed fixo adicionado: ({x}, {y}). Total: {len(self.auto_seed_points)}")
        except ValueError:
            messagebox.showerror("Erro", "Digite valores numéricos válidos para X e Y!")
    
    def remove_auto_seed(self):
        """Remove o seed fixo selecionado."""
        selection = self.auto_seeds_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione um seed para remover!")
            return
        
        index = selection[0]
        removed = self.auto_seed_points.pop(index)
        self.update_auto_seeds_display()
        self.log(f"✅ Seed fixo removido: {removed}. Total: {len(self.auto_seed_points)}")
    
    # --- Métodos de Atualização de Parâmetros de Filtros ---
    
    def update_clahe_clip(self, value):
        self.filter_params['clahe_clip_limit'] = float(value)
        self.lbl_clahe_clip.config(text=f"{float(value):.1f}")
    
    def update_clahe_grid(self, value):
        grid_size = int(float(value))
        self.filter_params['clahe_grid_size'] = grid_size
        self.lbl_clahe_grid.config(text=str(grid_size))
    
    def update_gaussian(self, value):
        kernel = int(float(value))
        if kernel % 2 == 0:
            kernel += 1
        self.filter_params['gaussian_kernel'] = kernel
        self.lbl_gaussian.config(text=str(kernel))
    
    def update_median(self, value):
        kernel = int(float(value))
        if kernel % 2 == 0:
            kernel += 1
        self.filter_params['median_kernel'] = kernel
        self.lbl_median.config(text=str(kernel))
    
    def update_canny_low(self, value):
        self.filter_params['canny_low'] = int(float(value))
        self.lbl_canny_low.config(text=str(int(float(value))))
    
    def update_canny_high(self, value):
        self.filter_params['canny_high'] = int(float(value))
        self.lbl_canny_high.config(text=str(int(float(value))))
    
    def update_bilateral_d(self, value):
        d = int(float(value))
        self.filter_params['bilateral_d'] = d
        self.lbl_bilateral_d.config(text=str(d))
    
    def update_bilateral_sigma(self, value):
        sigma = int(float(value))
        self.filter_params['bilateral_sigma'] = sigma
        self.lbl_bilateral_sigma.config(text=str(sigma))
    
    def update_erosion_kernel(self, value):
        kernel = int(float(value))
        if kernel % 2 == 0:
            kernel += 1
        self.filter_params['erosion_kernel'] = kernel
        self.lbl_erosion_kernel.config(text=str(kernel))
    
    def update_erosion_iterations(self, value):
        iterations = int(float(value))
        self.filter_params['erosion_iterations'] = iterations
        self.lbl_erosion_iterations.config(text=str(iterations))

    def zoom_image(self, event, canvas_type):
        """Controle de zoom"""
        if canvas_type == "original" and self.original_image is None:
            return
        if canvas_type == "preprocessed" and self.preprocessed_image is None:
            return
        if canvas_type == "segmented" and self.segmented_image is None:
            return
        
        # Determinar a direção do zoom
        if event.delta > 0 or event.num == 4:
            # Aumentar o zoom
            zoom_factor = 1.2
        else:
            # Diminuir o zoom
            zoom_factor = 0.8
        
        # Atualizar nível do zoom
        if canvas_type == "original":
            self.zoom_level_original *= zoom_factor
            self.zoom_level_original = max(0.1, min(5.0, self.zoom_level_original))
            self.display_image_zoomed(
                self.original_image, self.canvas_original, 
                self.zoom_level_original,
                "original",
            )
        elif canvas_type == "preprocessed":
            self.zoom_level_preprocessed *= zoom_factor
            self.zoom_level_preprocessed = max(0.1, min(5.0, self.zoom_level_preprocessed))
            self.display_image_zoomed(
                self.preprocessed_image, self.canvas_preprocessed,
                self.zoom_level_preprocessed,
                "preprocessed",
            )
        else:  # segmented
            self.zoom_level_segmented *= zoom_factor
            self.zoom_level_segmented = max(0.1, min(5.0, self.zoom_level_segmented))
            self.display_image_zoomed(
                self.segmented_image, self.canvas_segmented,
                self.zoom_level_segmented,
                "segmented",
            )

    def display_image_zoomed(self, pil_image, canvas, zoom_level, canvas_type):
        """Exibe imagem com zoom aplicado."""
        if pil_image is None:
            return
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2:
            canvas_width, canvas_height = 300, 300
        
        # Calcular novas dimensões com base no zoom
        new_width = int(pil_image.width * zoom_level)
        new_height = int(pil_image.height * zoom_level)
        
        # Redimensionar imagem
        img_resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Criar imagem
        if canvas_type == "original":
            self.tk_image_original = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_original
        elif canvas_type == "preprocessed":
            self.tk_image_preprocessed = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_preprocessed
        else:  # segmented
            self.tk_image_segmented = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_segmented
        
        # Limpar e redesenhar
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            anchor=tk.CENTER, image=img_tk
        )

    def start_pan(self, event, canvas_type):
        """Iniciar operação de exibição"""
        if canvas_type == "original":
            self.is_panning_original = True
            self.pan_start_x_original = event.x
            self.pan_start_y_original = event.y
            self.click_moved_original = False  # Reset flag de movimento
            self.canvas_original.config(cursor="fleur")
        elif canvas_type == "preprocessed":
            self.is_panning_preprocessed = True
            self.pan_start_x_preprocessed = event.x
            self.pan_start_y_preprocessed = event.y
            self.click_moved_preprocessed = False  # Reset flag de movimento
            self.canvas_preprocessed.config(cursor="fleur")
        else:  # segmented
            self.is_panning_segmented = True
            self.pan_start_x_segmented = event.x
            self.pan_start_y_segmented = event.y
            self.canvas_segmented.config(cursor="fleur")

    def pan_image(self, event, canvas_type):
        """Exibe imagem por enquanto é arrastado."""
        if canvas_type == "original" and self.is_panning_original:
            # Calcula a distância
            dx = event.x - self.pan_start_x_original
            dy = event.y - self.pan_start_y_original
            
            # Marca que houve movimento
            if abs(dx) > 3 or abs(dy) > 3:
                self.click_moved_original = True
            
            # Move elementos do canvas
            self.canvas_original.move("all", dx, dy)
            
            # Atualiza posições iniciais
            self.pan_start_x_original = event.x
            self.pan_start_y_original = event.y
            
        elif canvas_type == "preprocessed" and self.is_panning_preprocessed:
            dx = event.x - self.pan_start_x_preprocessed
            dy = event.y - self.pan_start_y_preprocessed
            
            # Marca que houve movimento
            if abs(dx) > 3 or abs(dy) > 3:
                self.click_moved_preprocessed = True
            
            self.canvas_preprocessed.move("all", dx, dy)
            self.pan_start_x_preprocessed = event.x
            self.pan_start_y_preprocessed = event.y
            
        elif canvas_type == "segmented" and self.is_panning_segmented:
            dx = event.x - self.pan_start_x_segmented
            dy = event.y - self.pan_start_y_segmented
            
            self.canvas_segmented.move("all", dx, dy)
            self.pan_start_x_segmented = event.x
            self.pan_start_y_segmented = event.y

    def stop_pan(self, event, canvas_type):
        """Parar operação de exibição"""
        if canvas_type == "original":
            self.is_panning_original = False
            self.canvas_original.config(cursor="")
            
            # Se não houve movimento, trata como clique de seed
            if not self.click_moved_original:
                self.on_click_seed(event)
        elif canvas_type == "preprocessed":
            self.is_panning_preprocessed = False
            self.canvas_preprocessed.config(cursor="")
            
            # Se não houve movimento, trata como clique na pré-processada
            if not self.click_moved_preprocessed:
                self.on_click_seed_preprocessed(event)
        else:  # segmented
            self.is_panning_segmented = False
            self.canvas_segmented.config(cursor="")

    def reset_zoom(self, canvas_type):
        """Reinicia zoom e display para default"""
        if canvas_type == "original":
            self.zoom_level_original = 1.0
            if self.original_image is not None:
                self.display_image(self.original_image, self.canvas_original, "original")
        elif canvas_type == "preprocessed":
            self.zoom_level_preprocessed = 1.0
            if self.preprocessed_image is not None:
                self.display_image(self.preprocessed_image, self.canvas_preprocessed, "preprocessed")
        else:  # segmented
            self.zoom_level_segmented = 1.0
            if self.segmented_image is not None:
                self.display_image(self.segmented_image, self.canvas_segmented, "segmented")

    def load_csv(self):
        """ Carrega o arquivo CSV de demografia. """
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not file_path:
            return

        try:
            self.dataframe = pd.read_csv(file_path, sep=';')
            
            # TODO: Pré-processar o CSV conforme item 9 [cite: 90]
            # Exemplo:
            # self.dataframe['Class'] = self.dataframe['Group'].apply(
            #    lambda x: 'NonDemented' if x == 'Nondemented' or (x == 'Converted' and row['CDR'] == 0) else 'Demented'
            # )
            # Nota: O 'apply' com 'row' é complexo. Pode ser melhor fazer em etapas.
            
            self.lbl_csv_status.config(text=f"CSV Carregado: {os.path.basename(file_path)}", foreground="green")
            self.log(f"CSV '{file_path}' carregado com {len(self.dataframe)} linhas.")
            self.log(f"Colunas: {list(self.dataframe.columns)}")
        except Exception as e:
            messagebox.showerror("Erro ao Carregar CSV", f"Erro: {e}")
            self.log(f"Falha ao carregar CSV: {e}")

    def load_image(self):
        """ Carrega uma imagem (JPG, PNG, Nifti) e a exibe. [cite: 76, 77] """
        file_path = filedialog.askopenfilename(
            title="Selecione a imagem",
            filetypes=(
                ("Nifti files", "*.nii *.nii.gz"),
                ("Imagens", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            )
        )
        if not file_path:
            return
        
        self.image_path = file_path
        self.log(f"Carregando imagem: {file_path}")

        try:
            if file_path.endswith((".nii", ".nii.gz")):
                # Carregamento de Nifti [cite: 77]
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                
                # TODO: O PDF diz "coronal" [cite: 51] e "coronal 134"[cite: 29].
                # Se o Nifti for 3D, você precisa pegar o slice correto.
                # Ex: if len(img_data.shape) == 3: img_data = img_data[:, :, 134]
                # Ou, se for um arquivo 2D salvo como Nifti:
                if len(img_data.shape) == 3:
                    # Assumindo que o slice coronal é o 2º eixo (Y)
                    slice_idx = img_data.shape[1] // 2
                    img_data = img_data[:, slice_idx, :]

                # Normalizar para 8 bits (0-255)
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                img_data = (img_data * 255).astype(np.uint8)
                self.original_image = Image.fromarray(img_data).convert("L") # Converte para Grayscale
                
            else:
                # Carregamento de PNG/JPG [cite: 77]
                self.original_image = Image.open(file_path).convert("L") # Converte para Grayscale

            # Exibir imagem original
            self.display_image(self.original_image, self.canvas_original, "original")
            
            # Reseta sistema de filtros
            self.preprocessed_image = None
            self.segmented_image = None
            self.image_mask = None
            self.filter_history = []
            self.original_image_backup = None
            self.current_filtered_image = None
            
            # Limpa canvas
            self.canvas_preprocessed.delete("all")
            self.canvas_segmented.delete("all")
            
            # Atualiza labels
            self.lbl_current_filter.config(text="Status: Nova imagem carregada", foreground="green")
            self.lbl_filter_history.config(text="Nenhum", foreground="gray")

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Imagem", f"Erro: {e}")
            self.log(f"Falha ao carregar imagem: {e}")

    def display_image(self, pil_image, canvas, canvas_type):
        """ Exibe uma imagem PIL em um canvas Tkinter, com zoom/redimensionamento. [cite: 76]"""
        # Reinicia zoom ao exibir uma nova imagem
        if canvas_type == "original":
            self.zoom_level_original = 1.0
        elif canvas_type == "preprocessed":
            self.zoom_level_preprocessed = 1.0
        else:  # segmented
            self.zoom_level_segmented = 1.0
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2: # Canvas não está pronto
            canvas_width, canvas_height = 300, 300 # Valor padrão

        # Redimensiona a imagem para caber no canvas mantendo a proporção
        img_resized = ImageOps.contain(pil_image, (canvas_width, canvas_height))

        # Criar PhotoImage
        if canvas_type == "original":
            self.tk_image_original = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_original
        elif canvas_type == "preprocessed":
            self.tk_image_preprocessed = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_preprocessed
        else:  # segmented
            self.tk_image_segmented = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_segmented
        
        canvas.delete("all")
        canvas.create_image(
            canvas_width / 2, canvas_height / 2,
            anchor=tk.CENTER, image=img_tk
        )

    # --- 4. FUNÇÕES DE PROCESSAMENTO DE IMAGEM ---

    def apply_selected_filter(self):
        """
        Aplica o filtro selecionado SOBRE a imagem atual (empilha filtros).
        Se for o primeiro filtro, usa a imagem original.
        """
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro.")
            return
        
        # Backup da imagem original na primeira aplicação
        if self.original_image_backup is None:
            self.original_image_backup = self.original_image.copy()
        
        # Define a imagem base: se já tem filtros, usa a filtrada; senão, usa a original
        if self.current_filtered_image is not None:
            base_image = self.current_filtered_image
        else:
            base_image = self.original_image
        
        filter_type = self.filter_mode.get()
        params = self.filter_params
        
        self.log(f"\n🔧 Aplicando filtro: {filter_type.upper()}")
        self.log(f"   Base: {'Imagem filtrada anterior' if self.current_filtered_image else 'Imagem original'}")
        
        # Converte para numpy
        img_np = np.array(base_image.convert('L'))
        
        # Aplica o filtro selecionado COM PARÂMETROS
        if filter_type == "otsu_clahe":
            # CLAHE primeiro
            clahe = cv2.createCLAHE(
                clipLimit=params['clahe_clip_limit'], 
                tileGridSize=(params['clahe_grid_size'], params['clahe_grid_size'])
            )
            img_filtered = clahe.apply(img_np)
            # Depois Otsu
            _, img_filtered = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            filter_name = f"Otsu+CLAHE(clip={params['clahe_clip_limit']:.1f}, grid={params['clahe_grid_size']})"
            self.log(f"   ✓ Params: clip={params['clahe_clip_limit']:.1f}, grid={params['clahe_grid_size']}")
            
        elif filter_type == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=params['clahe_clip_limit'], 
                tileGridSize=(params['clahe_grid_size'], params['clahe_grid_size'])
            )
            img_filtered = clahe.apply(img_np)
            filter_name = f"CLAHE(clip={params['clahe_clip_limit']:.1f}, grid={params['clahe_grid_size']})"
            self.log(f"   ✓ Params: clip={params['clahe_clip_limit']:.1f}, grid={params['clahe_grid_size']}")
            
        elif filter_type == "otsu":
            _, img_filtered = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            filter_name = "Otsu"
            self.log("   ✓ Binarização automática")
            
        elif filter_type == "canny":
            img_filtered = cv2.Canny(img_np, params['canny_low'], params['canny_high'])
            filter_name = f"Canny(low={params['canny_low']}, high={params['canny_high']})"
            self.log(f"   ✓ Params: low={params['canny_low']}, high={params['canny_high']}")
            
        elif filter_type == "gaussian":
            k = params['gaussian_kernel']
            img_filtered = cv2.GaussianBlur(img_np, (k, k), 0)
            filter_name = f"Gaussian(k={k})"
            self.log(f"   ✓ Params: kernel={k}x{k}")
            
        elif filter_type == "median":
            k = params['median_kernel']
            img_filtered = cv2.medianBlur(img_np, k)
            filter_name = f"Median(k={k})"
            self.log(f"   ✓ Params: kernel={k}x{k}")
            
        elif filter_type == "bilateral":
            d = params['bilateral_d']
            sigma = params['bilateral_sigma']
            img_filtered = cv2.bilateralFilter(img_np, d, sigma, sigma)
            filter_name = f"Bilateral(d={d}, σ={sigma})"
            self.log(f"   ✓ Params: d={d}, sigma={sigma}")
        
        elif filter_type == "erosion":
            kernel_size = params['erosion_kernel']
            iterations = params['erosion_iterations']
            # Cria kernel morfológico
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Aplica erosão
            img_filtered = cv2.erode(img_np, kernel, iterations=iterations)
            filter_name = f"Erosão(k={kernel_size}, iter={iterations})"
            self.log(f"   ✓ Params: kernel={kernel_size}x{kernel_size}, iterations={iterations}")
        
        else:
            self.log(f"   ⚠ Filtro desconhecido: {filter_type}")
            return
        
        # Adiciona ao histórico
        self.filter_history.append(filter_name)
        
        # Armazena como imagem filtrada atual
        self.current_filtered_image = Image.fromarray(img_filtered)
        self.preprocessed_image = self.current_filtered_image
        
        # Exibe na janela 2
        self.display_image(self.preprocessed_image, self.canvas_preprocessed, "preprocessed")
        
        # Atualiza labels
        history_text = " → ".join(self.filter_history)
        self.lbl_filter_history.config(text=history_text, foreground="purple")
        
        self.lbl_current_filter.config(
            text=f"Status: {len(self.filter_history)} filtro(s) aplicado(s)",
            foreground="green"
        )
        
        self.log(f"✅ Filtro aplicado! Total de filtros: {len(self.filter_history)}")
        self.log(f"   Pipeline: {history_text}\n")
    
    def reset_filters(self):
        """Reseta todos os filtros e volta para a imagem original."""
        if self.original_image_backup is None:
            messagebox.showinfo("Info", "Nenhum filtro foi aplicado ainda.")
            return
        
        self.log("\n↻ RESET DE FILTROS")
        self.log(f"   Removendo {len(self.filter_history)} filtro(s)")
        
        # Limpa histórico e estado
        self.filter_history = []
        self.current_filtered_image = None
        
        # Restaura imagem original na janela 2
        self.preprocessed_image = self.original_image_backup.copy()
        self.display_image(self.preprocessed_image, self.canvas_preprocessed, "preprocessed")
        
        # Atualiza labels
        self.lbl_filter_history.config(text="Nenhum (resetado)", foreground="gray")
        self.lbl_current_filter.config(text="Status: Original (sem filtros)", foreground="green")
        
        self.log("✅ Filtros resetados! Imagem original restaurada.\n")
        messagebox.showinfo("Reset", "Todos os filtros foram removidos!\nImagem original restaurada.")

    def toggle_multi_seed_mode(self):
        """Alterna o modo multi-seed ligado/desligado."""
        self.multi_seed_mode = not self.multi_seed_mode
        
        if self.multi_seed_mode:
            self.accumulated_mask = None
            self.multi_seed_points = []
            self.lbl_multi_seed.config(
                text="Multi-Seed: 🟢 ATIVO (Clique nas janelas para adicionar seeds)",
                foreground="green"
            )
            self.log("\n🟢 Modo Multi-Seed ATIVADO - Clique nas janelas para adicionar seeds")
        else:
            self.lbl_multi_seed.config(
                text="Multi-Seed: 🔴 INATIVO",
                foreground="gray"
            )
            self.log("🔴 Modo Multi-Seed DESATIVADO\n")
    
    def open_batch_config_window(self):
        """Abre janela de configuração avançada para processamento em lote."""
        config_window = tk.Toplevel(self.root)
        config_window.title("⚙️ Configuração de Processamento em Lote")
        config_window.geometry("700x800")
        config_window.resizable(True, True)
        config_window.transient(self.root)  # Mantém a janela no topo
        config_window.grab_set()  # Torna a janela modal
        
        # Container principal com scroll
        container = ttk.Frame(config_window)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para scroll
        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        
        # Frame principal (vai dentro do canvas)
        main_frame = ttk.Frame(canvas, padding="10")
        
        # Configuração do scroll
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        main_frame.bind("<Configure>", configure_scroll_region)
        
        # Cria janela no canvas
        canvas_window = canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configura scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas e scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Ajusta largura do frame quando o canvas é redimensionado
        def on_canvas_configure(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
            configure_scroll_region()
        
        canvas.bind('<Configure>', on_canvas_configure)
        
        # Bind mouse wheel para scroll (Windows e Linux)
        def _on_mousewheel(event):
            if event.delta:
                # Windows
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                # Linux
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
        
        # Bind para diferentes sistemas
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)
        
        # Foca no canvas para receber eventos de scroll
        canvas.focus_set()
        
        # Armazena referência do canvas na janela para atualizações
        config_window.canvas = canvas
        config_window.main_frame = main_frame
        
        # Função para atualizar scroll (pode ser chamada de outras funções)
        def update_scroll():
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        config_window.update_scroll = update_scroll
        
        # Título
        title_label = ttk.Label(main_frame, text="🔧 Configuração Avançada de Lote", 
                                font=("Arial", 14, "bold"), foreground="darkblue")
        title_label.pack(pady=10)
        
        # ═══════════════════════════════════════════════════════════
        # SEÇÃO 1: FILTROS PARA APLICAR
        # ═══════════════════════════════════════════════════════════
        filter_frame = ttk.LabelFrame(main_frame, text="🎨 1. FILTROS A APLICAR (Pipeline)", padding="10")
        filter_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(filter_frame, text="📋 Adicione filtros ao pipeline (opcional):", 
                  foreground="blue", font=("Arial", 9)).pack(anchor=tk.W, pady=(0,2))
        ttk.Label(filter_frame, text="Os filtros serão aplicados em sequência antes da segmentação.", 
                  foreground="gray", font=("Arial", 8)).pack(anchor=tk.W, pady=(0,5))
        
        # Lista de filtros selecionados
        self.batch_filters = []  # [(tipo, params), ...]
        
        # Frame de seleção de filtros
        filter_select_frame = ttk.Frame(filter_frame)
        filter_select_frame.pack(fill=tk.X, pady=5)
        
        self.batch_filter_var = tk.StringVar(value="CLAHE")
        
        # Mapeamento de nome de exibição para valor interno
        self.filter_options_map = {
            "CLAHE": "clahe",
            "Gaussian Blur": "gaussian",
            "Median Filter": "median",
            "Bilateral Filter": "bilateral",
            "Canny": "canny",
            "Otsu": "otsu",
            "Otsu + CLAHE": "otsu_clahe",
            "Erosão": "erosion"
        }
        
        ttk.Label(filter_select_frame, text="Filtro:").pack(side=tk.LEFT, padx=5)
        filter_combo = ttk.Combobox(filter_select_frame, textvariable=self.batch_filter_var, 
                                    values=list(self.filter_options_map.keys()), width=20, state="readonly")
        filter_combo.pack(side=tk.LEFT, padx=5)
        
        btn_add_filter = ttk.Button(filter_select_frame, text="➕ Adicionar Filtro", 
                                    command=lambda: self.add_filter_to_batch(config_window))
        btn_add_filter.pack(side=tk.LEFT, padx=5)
        
        # Lista de filtros adicionados
        filter_list_header = ttk.Frame(filter_frame)
        filter_list_header.pack(fill=tk.X, pady=(10,2))
        ttk.Label(filter_list_header, text="Pipeline de Filtros:", 
                  foreground="blue").pack(side=tk.LEFT)
        self.lbl_filter_count = ttk.Label(filter_list_header, text="(0 filtros)", 
                                          foreground="gray", font=("Arial", 8))
        self.lbl_filter_count.pack(side=tk.LEFT, padx=5)
        
        self.batch_filter_list = tk.Listbox(filter_frame, height=4, font=("Courier", 9))
        self.batch_filter_list.pack(fill=tk.X, pady=5)
        # Mensagem inicial
        self.batch_filter_list.insert(tk.END, "   (Nenhum filtro adicionado - OPCIONAL)")
        self.batch_filter_list.config(foreground="gray")
        
        btn_clear_filters = ttk.Button(filter_frame, text="🗑️ Limpar Filtros", 
                                       command=self.clear_batch_filters)
        btn_clear_filters.pack(pady=2)
        
        # ═══════════════════════════════════════════════════════════
        # SEÇÃO 2: SEED POINTS
        # ═══════════════════════════════════════════════════════════
        seed_frame = ttk.LabelFrame(main_frame, text="📍 2. SEED POINTS", padding="10")
        seed_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(seed_frame, text="Configure os pontos de seed para region growing:", 
                  foreground="blue").pack(anchor=tk.W, pady=(0,5))
        
        # Entry para adicionar seeds
        seed_entry_frame = ttk.Frame(seed_frame)
        seed_entry_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(seed_entry_frame, text="X:").pack(side=tk.LEFT, padx=2)
        self.batch_seed_x = ttk.Entry(seed_entry_frame, width=8)
        self.batch_seed_x.pack(side=tk.LEFT, padx=2)
        self.batch_seed_x.insert(0, "164")
        
        ttk.Label(seed_entry_frame, text="Y:").pack(side=tk.LEFT, padx=2)
        self.batch_seed_y = ttk.Entry(seed_entry_frame, width=8)
        self.batch_seed_y.pack(side=tk.LEFT, padx=2)
        self.batch_seed_y.insert(0, "91")
        
        btn_add_seed = ttk.Button(seed_entry_frame, text="➕ Adicionar Seed", 
                                  command=self.add_seed_to_batch)
        btn_add_seed.pack(side=tk.LEFT, padx=5)
        
        # Lista de seeds
        self.batch_seed_list = tk.Listbox(seed_frame, height=3, font=("Courier", 9))
        self.batch_seed_list.pack(fill=tk.X, pady=5)
        
        # Seeds padrão
        self.batch_seeds = [(158,98),(109,124),(109,81)]
        self.update_seed_list()
        
        btn_clear_seeds = ttk.Button(seed_frame, text="🗑️ Limpar Seeds", 
                                     command=self.clear_batch_seeds)
        btn_clear_seeds.pack(pady=2)
        
        # ═══════════════════════════════════════════════════════════
        # SEÇÃO 3: PARÂMETROS DE SEGMENTAÇÃO
        # ═══════════════════════════════════════════════════════════
        seg_frame = ttk.LabelFrame(main_frame, text="⚙️ 3. PARÂMETROS DE SEGMENTAÇÃO", padding="10")
        seg_frame.pack(fill=tk.X, pady=10)
        
        # Threshold
        threshold_frame = ttk.Frame(seg_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Threshold:").pack(side=tk.LEFT, padx=5)
        self.batch_threshold_var = tk.IntVar(value=self.region_growing_threshold)
        self.batch_lbl_threshold = ttk.Label(threshold_frame, text=str(self.region_growing_threshold), 
                                             foreground="red", font=("Arial", 10, "bold"))
        self.batch_lbl_threshold.pack(side=tk.LEFT, padx=5)
        
        batch_threshold_slider = ttk.Scale(seg_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                          command=lambda v: self.update_batch_threshold(v))
        batch_threshold_slider.set(self.region_growing_threshold)
        batch_threshold_slider.pack(fill=tk.X, padx=5)
        
        # Conectividade
        connectivity_batch_frame = ttk.Frame(seg_frame)
        connectivity_batch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(connectivity_batch_frame, text="Conectividade:", foreground="blue").pack(side=tk.LEFT, padx=5)
        
        self.batch_connectivity_var = tk.IntVar(value=8)
        ttk.Radiobutton(connectivity_batch_frame, text="4-vizinhos", 
                       variable=self.batch_connectivity_var, value=4).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(connectivity_batch_frame, text="8-vizinhos", 
                       variable=self.batch_connectivity_var, value=8).pack(side=tk.LEFT, padx=5)
        
        # Kernel
        kernel_frame = ttk.Frame(seg_frame)
        kernel_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(kernel_frame, text="Kernel Morfológico:").pack(side=tk.LEFT, padx=5)
        self.batch_kernel_var = tk.IntVar(value=self.morphology_kernel_size)
        self.batch_lbl_kernel = ttk.Label(kernel_frame, text=f"{self.morphology_kernel_size}x{self.morphology_kernel_size}", 
                                          foreground="red", font=("Arial", 10, "bold"))
        self.batch_lbl_kernel.pack(side=tk.LEFT, padx=5)
        
        batch_kernel_slider = ttk.Scale(seg_frame, from_=3, to=25, orient=tk.HORIZONTAL,
                                       command=lambda v: self.update_batch_kernel(v))
        batch_kernel_slider.set(self.morphology_kernel_size)
        batch_kernel_slider.pack(fill=tk.X, padx=5)
        
        # Morfologia
        ttk.Label(seg_frame, text="Operações Morfológicas:", foreground="blue").pack(anchor=tk.W, pady=(10,5))
        
        self.batch_var_opening = tk.BooleanVar(value=self.apply_opening)
        ttk.Checkbutton(seg_frame, text="Abertura (remover ruído)", 
                       variable=self.batch_var_opening).pack(anchor=tk.W, padx=20)
        
        self.batch_var_closing = tk.BooleanVar(value=self.apply_closing)
        ttk.Checkbutton(seg_frame, text="Fechamento (fechar gaps)", 
                       variable=self.batch_var_closing).pack(anchor=tk.W, padx=20)
        
        self.batch_var_fill_holes = tk.BooleanVar(value=self.apply_fill_holes)
        ttk.Checkbutton(seg_frame, text="Preencher buracos", 
                       variable=self.batch_var_fill_holes).pack(anchor=tk.W, padx=20)
        
        self.batch_var_smooth = tk.BooleanVar(value=self.apply_smooth_contours)
        ttk.Checkbutton(seg_frame, text="Suavizar contornos", 
                       variable=self.batch_var_smooth).pack(anchor=tk.W, padx=20)
        
        # ═══════════════════════════════════════════════════════════
        # BOTÃO FINAL
        # ═══════════════════════════════════════════════════════════
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=20)
        
        btn_execute = ttk.Button(btn_frame, text="▶️ EXECUTAR PROCESSAMENTO EM LOTE", 
                                command=lambda: self.execute_batch_with_config(config_window))
        btn_execute.pack(side=tk.LEFT, padx=5)
        
        btn_cancel = ttk.Button(btn_frame, text="❌ Cancelar", 
                               command=config_window.destroy)
        btn_cancel.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.batch_config_status = ttk.Label(main_frame, text="", foreground="blue", font=("Arial", 9))
        self.batch_config_status.pack(pady=5)
    
    def add_filter_to_batch(self, window):
        """Adiciona um filtro à lista de filtros do lote."""
        # Pega o nome de exibição e converte para valor interno
        display_name = self.batch_filter_var.get()
        filter_type = self.filter_options_map.get(display_name, "clahe")
        
        # Captura parâmetros atuais do filtro
        params = self.filter_params.copy()
        
        filter_info = {
            'type': filter_type,
            'params': params
        }
        
        # Remove mensagem inicial se for o primeiro filtro
        if len(self.batch_filters) == 0:
            self.batch_filter_list.delete(0, tk.END)
            self.batch_filter_list.config(foreground="black")
        
        self.batch_filters.append(filter_info)
        
        # Atualiza lista visual com parâmetros
        filter_display = {
            "clahe": f"CLAHE(clip={params['clahe_clip_limit']:.1f}, grid={params['clahe_grid_size']})",
            "gaussian": f"Gaussian(k={params['gaussian_kernel']})",
            "median": f"Median(k={params['median_kernel']})",
            "bilateral": f"Bilateral(d={params['bilateral_d']}, σ={params['bilateral_sigma']})",
            "canny": f"Canny(low={params['canny_low']}, high={params['canny_high']})",
            "otsu": "Otsu",
            "otsu_clahe": f"Otsu+CLAHE(clip={params['clahe_clip_limit']:.1f})"
        }
        
        detailed_name = filter_display.get(filter_type, display_name)
        self.batch_filter_list.insert(tk.END, f"{len(self.batch_filters)}. {detailed_name}")
        
        # Atualiza contador
        self.lbl_filter_count.config(text=f"({len(self.batch_filters)} filtro{'s' if len(self.batch_filters) > 1 else ''})", 
                                     foreground="darkgreen")
        
        # Atualiza status visual na própria janela (sem popup)
        if hasattr(self, 'batch_config_status'):
            self.batch_config_status.config(
                text=f"✅ Filtro '{detailed_name}' adicionado! Total: {len(self.batch_filters)}",
                foreground="green"
            )
            
            # Limpa o status após 3 segundos
            if hasattr(self, 'batch_status_timer'):
                try:
                    window.after_cancel(self.batch_status_timer)
                except:
                    pass
            try:
                self.batch_status_timer = window.after(3000, lambda: self.batch_config_status.config(text="", foreground="blue") if hasattr(self, 'batch_config_status') else None)
            except:
                pass
        
        # Atualiza scroll se disponível
        if hasattr(window, 'update_scroll'):
            window.update_scroll()
    
    def clear_batch_filters(self):
        """Limpa todos os filtros do lote."""
        self.batch_filters = []
        self.batch_filter_list.delete(0, tk.END)
        # Restaura mensagem inicial
        self.batch_filter_list.insert(tk.END, "   (Nenhum filtro adicionado - OPCIONAL)")
        self.batch_filter_list.config(foreground="gray")
        # Atualiza contador
        self.lbl_filter_count.config(text="(0 filtros)", foreground="gray")
        
        # Atualiza status visual (sem popup)
        # Nota: batch_config_status pode não existir se a janela não estiver aberta
        if hasattr(self, 'batch_config_status'):
            self.batch_config_status.config(
                text="✅ Todos os filtros foram removidos!",
                foreground="orange"
            )
            # Limpa o status após 3 segundos
            if hasattr(self, 'batch_status_timer'):
                self.batch_config_status.master.after_cancel(self.batch_status_timer)
            self.batch_status_timer = self.batch_config_status.master.after(
                3000, 
                lambda: self.batch_config_status.config(text="", foreground="blue") if hasattr(self, 'batch_config_status') else None
            )
    
    def add_seed_to_batch(self):
        """Adiciona um seed point à lista."""
        try:
            x = int(self.batch_seed_x.get())
            y = int(self.batch_seed_y.get())
            self.batch_seeds.append((x, y))
            self.update_seed_list()
            
            # Atualiza status visual (sem popup)
            if hasattr(self, 'batch_config_status'):
                self.batch_config_status.config(
                    text=f"✅ Seed ({x}, {y}) adicionado! Total: {len(self.batch_seeds)}",
                    foreground="green"
                )
                # Limpa o status após 3 segundos
                if hasattr(self, 'batch_status_timer'):
                    try:
                        self.batch_config_status.master.after_cancel(self.batch_status_timer)
                    except:
                        pass
                try:
                    self.batch_status_timer = self.batch_config_status.master.after(
                        3000, 
                        lambda: self.batch_config_status.config(text="", foreground="blue") if hasattr(self, 'batch_config_status') else None
                    )
                except:
                    pass
        except ValueError:
            messagebox.showerror("Erro", "Digite valores numéricos válidos para X e Y!")
    
    def clear_batch_seeds(self):
        """Limpa todos os seeds."""
        self.batch_seeds = []
        self.update_seed_list()
        
        # Atualiza status visual (sem popup)
        if hasattr(self, 'batch_config_status'):
            self.batch_config_status.config(
                text="✅ Todos os seeds foram removidos!",
                foreground="orange"
            )
            # Limpa o status após 3 segundos
            if hasattr(self, 'batch_status_timer'):
                try:
                    self.batch_config_status.master.after_cancel(self.batch_status_timer)
                except:
                    pass
            try:
                self.batch_status_timer = self.batch_config_status.master.after(
                    3000, 
                    lambda: self.batch_config_status.config(text="", foreground="blue") if hasattr(self, 'batch_config_status') else None
                )
            except:
                pass
    
    def update_seed_list(self):
        """Atualiza a lista visual de seeds."""
        self.batch_seed_list.delete(0, tk.END)
        for i, (x, y) in enumerate(self.batch_seeds, 1):
            self.batch_seed_list.insert(tk.END, f"{i}. Seed ({x}, {y})")
    
    def update_batch_threshold(self, value):
        """Atualiza threshold do lote."""
        threshold = int(float(value))
        self.batch_threshold_var.set(threshold)
        self.batch_lbl_threshold.config(text=str(threshold))
    
    def update_batch_kernel(self, value):
        """Atualiza kernel do lote."""
        kernel = int(float(value))
        if kernel % 2 == 0:
            kernel += 1
        self.batch_kernel_var.set(kernel)
        self.batch_lbl_kernel.config(text=f"{kernel}x{kernel}")
    
    def execute_batch_with_config(self, config_window):
        """Executa o processamento em lote com as configurações personalizadas."""
        # Validações
        if not self.batch_seeds:
            messagebox.showerror("Erro", "Adicione pelo menos um seed point!")
            return
        
        # Seleciona pastas
        input_folder = filedialog.askdirectory(title="Selecione a pasta com arquivos .nii (ENTRADA)")
        if not input_folder:
            return
        
        output_folder = filedialog.askdirectory(title="Selecione a pasta para salvar resultados (SAÍDA)")
        if not output_folder:
            return
        
        # Fecha janela de configuração
        config_window.destroy()
        
        # Lista arquivos
        nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        if not nii_files:
            messagebox.showwarning("Aviso", "Nenhum arquivo .nii encontrado na pasta!")
            return
        
        # Confirmação
        filter_pipeline = " → ".join([f['type'] for f in self.batch_filters]) if self.batch_filters else "Nenhum"
        
        result = messagebox.askyesno(
            "Confirmar Processamento em Lote",
            f"Processar {len(nii_files)} arquivos .nii?\n\n"
            f"📂 Entrada: {input_folder}\n"
            f"📂 Saída: {output_folder}\n\n"
            f"🎨 Filtros: {len(self.batch_filters)}\n"
            f"   {filter_pipeline}\n\n"
            f"📍 Seeds: {len(self.batch_seeds)}\n"
            f"   {self.batch_seeds}\n\n"
            f"⚙️ Threshold: {self.batch_threshold_var.get()}\n"
            f"⚙️ Kernel: {self.batch_kernel_var.get()}x{self.batch_kernel_var.get()}"
        )
        
        if not result:
            return
        
        # Executa processamento
        self.run_custom_batch_processing(input_folder, output_folder, nii_files)

    def run_custom_batch_processing(self, input_folder, output_folder, nii_files):
        """Executa o processamento em lote com configurações personalizadas."""
        # Limpa lista de descritores para novo processamento
        self.descriptors_list = []
        
        self.log("\n" + "="*80)
        self.log("📁 PROCESSAMENTO EM LOTE - CONFIGURAÇÃO PERSONALIZADA")
        self.log("="*80)
        self.log(f"Pasta de entrada: {input_folder}")
        self.log(f"Pasta de saída: {output_folder}")
        self.log(f"Total de arquivos: {len(nii_files)}")
        self.log(f"Pipeline de filtros: {len(self.batch_filters)}")
        for i, f in enumerate(self.batch_filters, 1):
            self.log(f"   {i}. {f['type']}")
        self.log(f"Seeds: {self.batch_seeds}")
        self.log(f"Threshold: {self.batch_threshold_var.get()}")
        self.log(f"Kernel: {self.batch_kernel_var.get()}x{self.batch_kernel_var.get()}")
        self.log("-"*80)
        
        success_count = 0
        error_count = 0
        
        for idx, filename in enumerate(nii_files, 1):
            try:
                self.log(f"\n[{idx}/{len(nii_files)}] Processando: {filename}")
                # self.lbl_batch_status.config(
                #     text=f"Processando {idx}/{len(nii_files)}: {filename[:30]}...",
                #     foreground="blue"
                # )
                self.root.update()
                
                # Carrega arquivo .nii
                file_path = os.path.join(input_folder, filename)
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                
                # Se for 3D, pega slice central
                if len(img_data.shape) == 3:
                    slice_idx = img_data.shape[1] // 2
                    img_data = img_data[:, slice_idx, :]
                
                # Normaliza para 8 bits
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                img_data = (img_data * 255).astype(np.uint8)
                
                # APLICA PIPELINE DE FILTROS
                processed_img = img_data.copy()
                
                for filter_info in self.batch_filters:
                    filter_type = filter_info['type']
                    params = filter_info['params']
                    
                    self.log(f"   🔧 Aplicando filtro: {filter_type}")
                    
                    if filter_type == "clahe":
                        clahe = cv2.createCLAHE(
                            clipLimit=params['clahe_clip_limit'],
                            tileGridSize=(params['clahe_grid_size'], params['clahe_grid_size'])
                        )
                        processed_img = clahe.apply(processed_img)
                    
                    elif filter_type == "gaussian_blur" or filter_type == "gaussian":
                        k = params['gaussian_kernel']
                        processed_img = cv2.GaussianBlur(processed_img, (k, k), 0)
                    
                    elif filter_type == "median_filter" or filter_type == "median":
                        k = params['median_kernel']
                        processed_img = cv2.medianBlur(processed_img, k)
                    
                    elif filter_type == "bilateral_filter" or filter_type == "bilateral":
                        d = params['bilateral_d']
                        sigma = params['bilateral_sigma']
                        processed_img = cv2.bilateralFilter(processed_img, d, sigma, sigma)
                    
                    elif filter_type == "canny":
                        processed_img = cv2.Canny(processed_img, params['canny_low'], params['canny_high'])
                    
                    elif filter_type == "otsu":
                        _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    elif filter_type == "otsu_clahe":
                        clahe = cv2.createCLAHE(
                            clipLimit=params['clahe_clip_limit'],
                            tileGridSize=(params['clahe_grid_size'], params['clahe_grid_size'])
                        )
                        processed_img = clahe.apply(processed_img)
                        _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    elif filter_type == "erosion":
                        kernel_size = params['erosion_kernel']
                        iterations = params['erosion_iterations']
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        processed_img = cv2.erode(processed_img, kernel, iterations=iterations)
                
                # SEGMENTAÇÃO MULTI-SEED
                self.log(f"   🎯 Segmentação com {len(self.batch_seeds)} seeds")
                combined_mask = None
                
                for seed_x, seed_y in self.batch_seeds:
                    # Verifica limites
                    if seed_x < 0 or seed_y < 0 or seed_x >= processed_img.shape[1] or seed_y >= processed_img.shape[0]:
                        self.log(f"      ⚠ Seed ({seed_x}, {seed_y}) fora dos limites")
                        continue
                    
                    mask = self.region_growing(processed_img, (seed_x, seed_y), 
                                              threshold=self.batch_threshold_var.get(),
                                              connectivity=self.batch_connectivity_var.get())
                    
                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                if combined_mask is None:
                    self.log(f"   ⚠ Nenhum seed válido! Pulando...")
                    error_count += 1
                    continue
                
                # PÓS-PROCESSAMENTO MORFOLÓGICO
                self.log(f"   🔬 Morfologia: kernel={self.batch_kernel_var.get()}x{self.batch_kernel_var.get()}")
                
                kernel_size = self.batch_kernel_var.get()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                final_mask = combined_mask.copy()
                
                if self.batch_var_opening.get():
                    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
                    self.log(f"      ✓ Abertura")
                
                if self.batch_var_closing.get():
                    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
                    self.log(f"      ✓ Fechamento")
                
                if self.batch_var_fill_holes.get():
                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filled_mask = np.zeros_like(final_mask)
                    for cnt in contours:
                        cv2.drawContours(filled_mask, [cnt], 0, 255, -1)
                    final_mask = filled_mask
                    self.log(f"      ✓ Preencher buracos")
                
                if self.batch_var_smooth.get():
                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    smoothed_mask = np.zeros_like(final_mask)
                    for cnt in contours:
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        smoothed_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                        cv2.drawContours(smoothed_mask, [smoothed_cnt], 0, 255, -1)
                    final_mask = smoothed_mask
                    self.log(f"      ✓ Suavizar contornos")
                
                # VALIDAÇÃO: Verifica se a segmentação não excedeu o limite
                is_valid, num_pixels = self.validate_segmentation_mask(final_mask, f"(LOTE: {filename})")
                if not is_valid:
                    self.log(f"      ⚠️ Region Growing falhou para {filename} ({num_pixels} pixels)")
                    self.log(f"      🔄 Aplicando método alternativo...")
                    # Usa método alternativo
                    final_mask = self.apply_alternative_segmentation(
                        processed_img, 
                        method=self.alternative_segmentation_method,
                        threshold=30
                    )
                    # Valida novamente
                    is_valid_alt, num_pixels_alt = self.validate_segmentation_mask(final_mask, f"(ALTERNATIVO: {filename})")
                    if is_valid_alt:
                        self.log(f"      ✅ Método alternativo funcionou! {num_pixels_alt} pixels")
                    else:
                        self.log(f"      ⚠️ Método alternativo também excedeu limite ({num_pixels_alt} pixels)")
                
                # Extrai descritores morfológicos
                base_name = os.path.splitext(filename)[0]
                if filename.endswith('.nii.gz'):
                    base_name = base_name.replace('.nii', '')
                
                self.log(f"   📊 Extraindo descritores morfológicos para: {base_name}")
                descriptors = self.extrair_descritores_ventriculo(final_mask, image_id=base_name)
                if descriptors:
                    self.descriptors_list.append(descriptors)
                    self.log(f"      ✅ Descritores adicionados (total: {len(self.descriptors_list)})")
                
                # CRIA IMAGEM SEGMENTADA COM CONTORNO VERMELHO
                img_with_contour = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
                cv2.drawContours(img_with_contour, large_contours, -1, (0, 0, 255), 3)  # Vermelho
                
                # SALVA RESULTADOS
                
                # Salva máscara binária
                mask_filename = f"{base_name}_mask.png"
                mask_path = os.path.join(output_folder, mask_filename)
                cv2.imwrite(mask_path, final_mask)
                
                # Salva imagem segmentada com contorno
                segmented_filename = f"{base_name}_segmented.png"
                segmented_path = os.path.join(output_folder, segmented_filename)
                cv2.imwrite(segmented_path, img_with_contour)
                
                # Salva imagem filtrada (intermediária)
                filtered_filename = f"{base_name}_filtered.png"
                filtered_path = os.path.join(output_folder, filtered_filename)
                cv2.imwrite(filtered_path, processed_img)
                
                num_pixels = np.sum(final_mask == 255)
                self.log(f"   ✅ Sucesso: {len(large_contours)} região(ões), {num_pixels} pixels")
                self.log(f"   📁 Salvos: {mask_filename}, {segmented_filename}, {filtered_filename}")
                
                success_count += 1
                
            except Exception as e:
                self.log(f"   ❌ ERRO: {str(e)}")
                error_count += 1
        
        # Salva CSV de descritores
        if len(self.descriptors_list) > 0:
            csv_path = self.salvar_descritores_csv(output_dir=output_folder)
            if csv_path:
                self.log(f"📄 CSV de descritores salvo: {csv_path}")
                
                # Faz merge automático dos CSVs
                self.log("\n🔄 Executando merge automático dos CSVs...")
                merged_path = self.merge_csv_files(
                    demographic_csv_path="oasis_longitudinal_demographic.csv",
                    descriptors_csv_path=csv_path,
                    output_path="merged_data.csv",
                    show_messagebox=False  # Não mostra messagebox quando executado automaticamente
                )
                if merged_path:
                    self.log(f"✅ Merge concluído: {merged_path}")
        
        # RELATÓRIO FINAL
        self.log("\n" + "="*80)
        self.log("📊 RELATÓRIO FINAL")
        self.log("="*80)
        self.log(f"✅ Sucesso: {success_count}/{len(nii_files)}")
        self.log(f"❌ Erros: {error_count}/{len(nii_files)}")
        self.log(f"📂 Resultados salvos em: {output_folder}")
        if len(self.descriptors_list) > 0:
            self.log(f"📊 Descritores extraídos: {len(self.descriptors_list)}")
        self.log("="*80 + "\n")
        
        # self.lbl_batch_status.config(
        #     text=f"✅ Lote concluído! {success_count} OK, {error_count} erros",
        #     foreground="green"
        # )
        
        messagebox.showinfo(
            "Processamento Concluído",
            f"Processamento em lote finalizado!\n\n"
            f"✅ Sucesso: {success_count}\n"
            f"❌ Erros: {error_count}\n\n"
            f"Resultados salvos em:\n{output_folder}"
        )

    def batch_segment_folder(self):
        """Processa em lote todos os arquivos .nii de uma pasta."""
        # Seleciona pasta de entrada
        input_folder = filedialog.askdirectory(
            title="Selecione a pasta com arquivos .nii (ENTRADA)"
        )
        if not input_folder:
            return
        
        # Seleciona/cria pasta de saída
        output_folder = filedialog.askdirectory(
            title="Selecione a pasta para salvar resultados (SAÍDA)"
        )
        if not output_folder:
            return
        
        # Lista todos os arquivos .nii
        nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
        
        if not nii_files:
            messagebox.showwarning("Aviso", "Nenhum arquivo .nii encontrado na pasta!")
            self.log("⚠ Nenhum arquivo .nii encontrado.")
            return
        
        self.log("\n" + "="*70)
        self.log("📁 PROCESSAMENTO EM LOTE - SEGMENTAÇÃO AUTOMÁTICA")
        self.log("="*70)
        self.log(f"Pasta de entrada: {input_folder}")
        self.log(f"Pasta de saída: {output_folder}")
        self.log(f"Total de arquivos .nii: {len(nii_files)}")
        self.log("-"*70)
        
        # Confirmação
        result = messagebox.askyesno(
            "Confirmar Processamento em Lote",
            f"Processar {len(nii_files)} arquivos .nii?\n\n"
            f"Entrada: {input_folder}\n"
            f"Saída: {output_folder}\n\n"
            f"Parâmetros:\n"
            f"• Seeds: {self.auto_seed_points}\n"
            f"• Threshold: {self.region_growing_threshold}\n"
            f"• Kernel: {self.morphology_kernel_size}x{self.morphology_kernel_size}"
        )
        
        if not result:
            self.log("❌ Processamento cancelado pelo usuário.\n")
            return
        
        # Limpa lista de descritores para novo processamento
        self.descriptors_list = []
        
        # Processa cada arquivo
        success_count = 0
        error_count = 0
        
        for idx, filename in enumerate(nii_files, 1):
            try:
                self.log(f"\n[{idx}/{len(nii_files)}] Processando: {filename}")
                # self.lbl_batch_status.config(
                #     text=f"Processando {idx}/{len(nii_files)}: {filename[:30]}...",
                #     foreground="blue"
                # )
                self.root.update()  # Atualiza interface
                
                # Carrega arquivo .nii
                file_path = os.path.join(input_folder, filename)
                nii_img = nib.load(file_path)
                img_data = nii_img.get_fdata()
                
                # Se for 3D, pega slice central (ou o método que você usa)
                if len(img_data.shape) == 3:
                    slice_idx = img_data.shape[1] // 2
                    img_data = img_data[:, slice_idx, :]
                
                # Normaliza para 8 bits
                img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
                img_data = (img_data * 255).astype(np.uint8)
                
                # Prepara imagem para segmentação (CLAHE ou Otsu)
                img_for_segmentation = self.prepare_image_for_segmentation(img_data)
                
                # Segmentação Multi-Seed
                combined_mask = None
                for seed_x, seed_y in self.auto_seed_points:
                    # Verifica limites
                    if seed_x < 0 or seed_y < 0 or seed_x >= img_data.shape[1] or seed_y >= img_data.shape[0]:
                        continue
                    
                    # Usa parâmetros atuais da interface (não do lote configurado)
                    mask = self.region_growing(img_for_segmentation, (seed_x, seed_y), 
                                              threshold=self.region_growing_threshold,
                                              connectivity=self.connectivity_var.get())
                    
                    if combined_mask is None:
                        combined_mask = mask.copy()
                    else:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                if combined_mask is None:
                    self.log(f"   ⚠ Seeds fora dos limites! Pulando...")
                    error_count += 1
                    continue
                
                # Pós-processamento morfológico
                final_mask = self.apply_morphological_postprocessing(combined_mask)
                
                # Extrai descritores morfológicos
                base_name = os.path.splitext(filename)[0]
                if filename.endswith('.nii.gz'):
                    base_name = base_name.replace('.nii', '')
                
                self.log(f"   📊 Extraindo descritores morfológicos para: {base_name}")
                descriptors = self.extrair_descritores_ventriculo(final_mask, image_id=base_name)
                if descriptors:
                    self.descriptors_list.append(descriptors)
                    self.log(f"      ✅ Descritores adicionados (total: {len(self.descriptors_list)})")
                
                # Cria imagem segmentada com contorno amarelo
                img_with_contour = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
                cv2.drawContours(img_with_contour, large_contours, -1, (0, 0, 255), 3)  # Vermelho vivo
                
                # Salva os resultados
                
                # Salva máscara binária
                mask_filename = f"{base_name}_mask.png"
                mask_path = os.path.join(output_folder, mask_filename)
                cv2.imwrite(mask_path, final_mask)
                
                # Salva imagem segmentada com contorno
                segmented_filename = f"{base_name}_segmented.png"
                segmented_path = os.path.join(output_folder, segmented_filename)
                cv2.imwrite(segmented_path, img_with_contour)
                
                num_pixels = np.sum(final_mask == 255)
                self.log(f"   ✓ Segmentado: {len(large_contours)} região(ões), {num_pixels} pixels")
                self.log(f"   ✓ Salvo: {mask_filename}")
                self.log(f"   ✓ Salvo: {segmented_filename}")
                
                success_count += 1
                
            except Exception as e:
                self.log(f"   ❌ ERRO: {str(e)}")
                error_count += 1
        
        # Salva CSV de descritores
        if len(self.descriptors_list) > 0:
            csv_path = self.salvar_descritores_csv(output_dir=output_folder)
            if csv_path:
                self.log(f"📄 CSV de descritores salvo: {csv_path}")
                
                # Faz merge automático dos CSVs
                self.log("\n🔄 Executando merge automático dos CSVs...")
                merged_path = self.merge_csv_files(
                    demographic_csv_path="oasis_longitudinal_demographic.csv",
                    descriptors_csv_path=csv_path,
                    output_path="merged_data.csv",
                    show_messagebox=False  # Não mostra messagebox quando executado automaticamente
                )
                if merged_path:
                    self.log(f"✅ Merge concluído: {merged_path}")
        
        # Relatório final
        self.log("-"*70)
        self.log(f"✅ PROCESSAMENTO EM LOTE CONCLUÍDO!")
        self.log(f"   • Total processado: {len(nii_files)}")
        self.log(f"   • Sucessos: {success_count}")
        self.log(f"   • Erros: {error_count}")
        self.log(f"   • Pasta de saída: {output_folder}")
        if len(self.descriptors_list) > 0:
            self.log(f"   • Descritores extraídos: {len(self.descriptors_list)}")
        self.log("="*70 + "\n")
        
        # self.lbl_batch_status.config(
        #     text=f"✓ Concluído: {success_count}/{len(nii_files)} arquivos",
        #     foreground="green"
        # )
        
        messagebox.showinfo(
            "Processamento Concluído",
            f"Processamento em lote finalizado!\n\n"
            f"Total: {len(nii_files)} arquivos\n"
            f"Sucessos: {success_count}\n"
            f"Erros: {error_count}\n\n"
            f"Resultados salvos em:\n{output_folder}"
        )

    def apply_filter(self, filter_name):
        """Aplica filtro de pré-processamento na imagem original."""
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return
        
        self.current_filter = filter_name
        img_np = np.array(self.original_image)
        
        if filter_name == "none":
            # Sem filtro: copia a original
            filtered_img = img_np.copy()
            self.lbl_current_filter.config(text="Filtro: Nenhum")
            self.log("Filtro removido.")
            
        elif filter_name == "clahe":
            # CLAHE - Equalização adaptativa de histograma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            filtered_img = clahe.apply(img_np)
            self.lbl_current_filter.config(text="Filtro: CLAHE")
            self.log("Filtro CLAHE aplicado.")
            
        elif filter_name == "canny":
            # Canny Edge Detection
            # Primeiro aplica blur para reduzir ruído
            blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
            filtered_img = cv2.Canny(blurred, threshold1=50, threshold2=150)
            self.lbl_current_filter.config(text="Filtro: Canny")
            self.log("Filtro Canny aplicado.")
            
        elif filter_name == "otsu":
            # Aplica CLAHE primeiro
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_img = clahe.apply(img_np)
            
            # Depois aplica Otsu Thresholding - binarização automática
            _, filtered_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.lbl_current_filter.config(text="Pré-proc: Otsu + CLAHE", foreground="green")
            self.log("✓ Filtro Otsu + CLAHE aplicado.")
            
        else:
            filtered_img = img_np.copy()
        
        # Converte para PIL e exibe no canvas preprocessed
        self.preprocessed_image = Image.fromarray(filtered_img)
        self.display_image(self.preprocessed_image, self.canvas_preprocessed, "preprocessed")

    def on_click_seed_preprocessed(self, event):
        """Captura clique na janela FILTRADA (janela 2) para segmentação."""
        if self.preprocessed_image is None:
            messagebox.showwarning("Aviso", "Aplique um filtro primeiro!")
            self.log("⚠ Nenhuma imagem filtrada disponível. Use a Seção 1 primeiro.")
            return

        # Pega coordenadas do clique
        x = event.x
        y = event.y

        # Converte coordenadas do canvas para coordenadas da imagem
        canvas_w = self.canvas_preprocessed.winfo_width()
        canvas_h = self.canvas_preprocessed.winfo_height()

        img_w = self.preprocessed_image.width
        img_h = self.preprocessed_image.height

        # Calcula o tamanho da imagem exibida considerando o zoom
        display_w = int(img_w * self.zoom_level_preprocessed)
        display_h = int(img_h * self.zoom_level_preprocessed)

        # Ajuste: centralização no canvas
        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        # Converte para coordenadas da imagem
        img_x = int((x - offset_x) / self.zoom_level_preprocessed)
        img_y = int((y - offset_y) / self.zoom_level_preprocessed)

        # Verifica se o clique está dentro da imagem
        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            self.log("⚠ Clique fora da imagem.")
            return

        self.log(f"📍 Clique na JANELA 2 (Filtrada): X={img_x}, Y={img_y}")
        
        # Executa segmentação (sempre usa a imagem filtrada)
        self.execute_segmentation_at_point(img_x, img_y, "filtered")

    def on_click_seed(self, event):
        """Captura clique no canvas ORIGINAL - mas recomenda clicar na janela 2."""
        if self.original_image is None:
            return

        # Pega coordenadas do clique
        x = event.x
        y = event.y

        # Converte coordenadas do canvas para coordenadas da imagem
        canvas_w = self.canvas_original.winfo_width()
        canvas_h = self.canvas_original.winfo_height()

        img_w = self.original_image.width
        img_h = self.original_image.height

        # Calcula o tamanho da imagem exibida considerando o zoom
        display_w = int(img_w * self.zoom_level_original)
        display_h = int(img_h * self.zoom_level_original)

        # Ajuste: centralização no canvas
        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        # Converte para coordenadas da imagem original
        img_x = int((x - offset_x) / self.zoom_level_original)
        img_y = int((y - offset_y) / self.zoom_level_original)

        # Verifica se o clique está dentro da imagem
        if img_x < 0 or img_y < 0 or img_x >= img_w or img_y >= img_h:
            self.log("⚠ Clique fora da imagem.")
            return

        self.log(f"📍 Clique na JANELA 1 (Original): X={img_x}, Y={img_y}")
        self.log(f"💡 Dica: Clique na JANELA 2 (Filtrada) para melhores resultados!")
        
        # Executa segmentação (usa a janela 2 se disponível)
        self.execute_segmentation_at_point(img_x, img_y, "original")

    def execute_segmentation_at_point(self, img_x, img_y, source_canvas):
        """
        Executa segmentação em um ponto específico.
        
        Args:
            img_x, img_y: coordenadas do clique
            source_canvas: "original" ou "filtered"
        """
        # SEMPRE USA A IMAGEM PRÉ-PROCESSADA (JANELA 2) se disponível
        if self.preprocessed_image is not None:
            img_for_seg_pil = self.preprocessed_image
            self.log(f"🎨 Usando imagem PRÉ-PROCESSADA (Janela 2) para segmentação")
        else:
            # Se não houver imagem pré-processada, usa a original
            img_for_seg_pil = self.original_image
            self.log(f"⚠️ Usando imagem ORIGINAL (sem filtros) - Aplique um filtro primeiro para melhores resultados")
        
        # Converte para numpy
        img_for_seg_np = np.array(img_for_seg_pil.convert('L'))
        
        # Executa segmentação baseado no método selecionado
        method = self.segmentation_method.get()
        
        if method == "region_growing":
            self.log(f"🌱 Método: Region Growing (threshold={self.region_growing_threshold})")
            
            if self.multi_seed_mode:
                # Modo Multi-Seed
                self.multi_seed_points.append((img_x, img_y))
                self.log(f"🎯 Multi-Seed {len(self.multi_seed_points)}: ({img_x}, {img_y})")
                self.lbl_multi_seed.config(
                    text=f"🎯 Multi-Seed: {len(self.multi_seed_points)} ponto(s)",
                    foreground="purple"
                )
                
                # Aplica region growing
                mask = self.region_growing(img_for_seg_np, (img_x, img_y), 
                                          threshold=self.region_growing_threshold,
                                          connectivity=self.connectivity_var.get())
                
                # Acumula máscaras
                if self.accumulated_mask is None:
                    self.accumulated_mask = mask.copy()
                else:
                    self.accumulated_mask = cv2.bitwise_or(self.accumulated_mask, mask)
                
                final_mask = self.apply_morphological_postprocessing(self.accumulated_mask)
                
            else:
                # Modo Single-Seed
                mask = self.region_growing(img_for_seg_np, (img_x, img_y), 
                                          threshold=self.region_growing_threshold,
                                          connectivity=self.connectivity_var.get())
                final_mask = self.apply_morphological_postprocessing(mask)
            
        elif method == "watershed":
            self.log("🎯 Método: Watershed")
            # Implementação placeholder do Watershed
            messagebox.showinfo("Em desenvolvimento", "Watershed será implementado em breve!")
            return
            
        elif method == "adaptive_threshold":
            self.log("🔲 Método: Adaptive Thresholding")
            # Implementação placeholder
            messagebox.showinfo("Em desenvolvimento", "Adaptive Threshold será implementado em breve!")
            return
            
        elif method == "kmeans":
            self.log("🧲 Método: K-Means Clustering")
            # Implementação placeholder
            messagebox.showinfo("Em desenvolvimento", "K-Means será implementado em breve!")
            return
        
        # VALIDAÇÃO: Verifica se a segmentação não excedeu o limite
        is_valid, num_pixels = self.validate_segmentation_mask(final_mask, "(MANUAL)")
        if not is_valid:
            self.log(f"\n🔄 Region Growing falhou ({num_pixels} pixels). Aplicando método alternativo...")
            # Usa método alternativo
            final_mask = self.apply_alternative_segmentation(
                img_for_seg_np, 
                method=self.alternative_segmentation_method,
                threshold=30
            )
            # Valida novamente
            is_valid_alt, num_pixels_alt = self.validate_segmentation_mask(final_mask, "(ALTERNATIVO)")
            if is_valid_alt:
                self.log(f"   ✅ Método alternativo funcionou! {num_pixels_alt} pixels")
            else:
                self.log(f"   ⚠️ Método alternativo também excedeu limite ({num_pixels_alt} pixels)")
        
        # Armazena máscara
        self.image_mask = final_mask
        
        # Extrai descritores morfológicos
        image_id = os.path.basename(self.image_path) if self.image_path else f"manual_{len(self.descriptors_list)}"
        self.log(f"\n📊 Extraindo descritores morfológicos para: {image_id}")
        descriptors = self.extrair_descritores_ventriculo(final_mask, image_id=image_id)
        if descriptors:
            self.descriptors_list.append(descriptors)
            self.log(f"   ✅ Descritores adicionados à lista (total: {len(self.descriptors_list)})")
        
        # Visualiza resultado na janela 3: SEMPRE usa a imagem PRÉ-PROCESSADA (se disponível)
        if self.preprocessed_image is not None:
            base_img = np.array(self.preprocessed_image.convert('L'))
        else:
            base_img = np.array(self.original_image.convert('L'))
        
        # Converte para BGR para desenhar contornos coloridos
        if len(base_img.shape) == 2:
            img_with_contour = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            img_with_contour = base_img.copy()
        
        # Encontra e desenha contornos
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        cv2.drawContours(img_with_contour, large_contours, -1, (0, 0, 255), 3)  # Vermelho vivo
        
        # Exibe na janela 3
        self.segmented_image = Image.fromarray(img_with_contour)
        self.display_image(self.segmented_image, self.canvas_segmented, "segmented")
        
        # Estatísticas
        num_pixels = np.sum(final_mask == 255)
        self.log(f"✅ Segmentação concluída: {len(large_contours)} região(ões) | {num_pixels} pixels")
        self.lbl_segment_status.config(
            text=f"✅ {len(large_contours)} região(ões) | {num_pixels} px",
            foreground="green"
        )

    def prepare_image_for_segmentation(self, img_np):
        """
        Prepara a imagem para segmentação.
        Como agora sempre usamos a Janela 2 (já filtrada), apenas aplicamos CLAHE adicional
        para melhorar o contraste para o region growing.
        
        Args:
            img_np: numpy array 2D (grayscale) - já vem da janela filtrada
            
        Returns:
            img_processed: imagem processada com CLAHE para melhor segmentação
        """
        # Aplica CLAHE para realçar regiões e melhorar o region growing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_processed = clahe.apply(img_np)
        self.log("   → Aplicando CLAHE adicional para melhorar região de crescimento")
        
        return img_processed

    def region_growing(self, image, seed, threshold=10, connectivity=8):
        """
        Algoritmo de Region Growing para segmentação.
        
        Args:
            image: numpy array 2D (grayscale)
            seed: (x, y) pixel inicial clicado
            threshold: variação de intensidade permitida em relação ao seed
            connectivity: 4 (4-vizinhos) ou 8 (8-vizinhos, padrão)
            
        Returns:
            mask: numpy array 2D binário (0=fundo, 255=região)
        """
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        seed_x, seed_y = seed
        seed_intensity = int(image[seed_y, seed_x])

        queue = [(seed_x, seed_y)]
        mask[seed_y, seed_x] = 255

        # Define vizinhança baseado na conectividade
        if connectivity == 4:
            # 4-connected neighbors (cima, baixo, esquerda, direita)
            neighbors = [(0, -1), (-1, 0), (1, 0), (0, 1)]
        else:
            # 8-connected neighbors (inclui diagonais)
            neighbors = [(-1, -1), (0, -1), (1, -1),
                         (-1,  0),         (1,  0),
                         (-1,  1), (0,  1), (1,  1)]

        while queue:
            x, y = queue.pop(0)

            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy

                if 0 <= nx < w and 0 <= ny < h:
                    if mask[ny, nx] == 0:  # não visitado
                        if abs(int(image[ny, nx]) - seed_intensity) < threshold:
                            mask[ny, nx] = 255
                            queue.append((nx, ny))

        return mask

    def apply_morphological_postprocessing(self, mask):
        """
        Aplica operações morfológicas para melhorar a máscara de segmentação.
        
        Args:
            mask: numpy array 2D binário (0=fundo, 255=região)
            
        Returns:
            processed_mask: numpy array 2D binário com operações morfológicas aplicadas
        """
        processed_mask = mask.copy()
        
        # Cria kernel morfológico
        kernel_size = self.morphology_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 1. Abertura (Opening) - Remove ruído pequeno
        if self.apply_opening:
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            self.log(f"   → Abertura aplicada (kernel {kernel_size}x{kernel_size})")
        
        # 2. Preenchimento de buracos (Fill Holes)
        if self.apply_fill_holes:
            # Encontra contornos
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Preenche cada contorno
            filled_mask = np.zeros_like(processed_mask)
            for cnt in contours:
                cv2.drawContours(filled_mask, [cnt], 0, 255, -1)  # -1 preenche o interior
            
            processed_mask = filled_mask
            self.log(f"   → Buracos preenchidos ({len(contours)} regiões)")
        
        # 3. Suavização de contornos (Contour Smoothing)
        if self.apply_smooth_contours:
            # Encontra contornos
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Suaviza cada contorno usando aproximação poligonal
            smoothed_mask = np.zeros_like(processed_mask)
            for cnt in contours:
                # Aproximação poligonal (epsilon = 0.5% do perímetro)
                epsilon = 0.005 * cv2.arcLength(cnt, True)
                smoothed_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(smoothed_mask, [smoothed_cnt], 0, 255, -1)
            
            processed_mask = smoothed_mask
            self.log(f"   → Contornos suavizados")
        
        return processed_mask
    
    def validate_segmentation_mask(self, mask, context=""):
        """
        Valida se a máscara de segmentação não excedeu o limite de pixels.
        Se exceder, considera que a segmentação falhou.
        
        Args:
            mask: numpy array 2D binário (0=fundo, 255=região)
            context: string para identificar o contexto (ex: "automática", "manual")
            
        Returns:
            (is_valid, num_pixels): (True/False, número de pixels segmentados)
        """
        num_pixels = np.sum(mask == 255)
        is_valid = num_pixels <= self.max_segmentation_pixels
        
        if not is_valid:
            self.log(f"\n⚠️ VALIDAÇÃO FALHOU {context}:")
            self.log(f"   Pixels encontrados: {num_pixels}")
            self.log(f"   Limite máximo: {self.max_segmentation_pixels}")
            self.log(f"   Excesso: {num_pixels - self.max_segmentation_pixels} pixels")
            self.log(f"   💡 A segmentação provavelmente capturou muito da imagem.")
            self.log(f"   🔄 Método alternativo será implementado aqui.")
        
        return is_valid, num_pixels
    
    # ============================================================================
    # MÉTODOS ALTERNATIVOS DE SEGMENTAÇÃO (para quando Region Growing falha)
    # ============================================================================
    
    def segment_roi_fixed_rectangle(self, image, roi_width=200, roi_height=200, threshold=30):
        """
        1. ROI Fixa / Central Crop (retângulo A×B)
        Recorta uma janela central e trabalha só nela.
        """
        h, w = image.shape
        center_x, center_y = w // 2, h // 2
        
        # Define ROI central
        x1 = max(0, center_x - roi_width // 2)
        y1 = max(0, center_y - roi_height // 2)
        x2 = min(w, center_x + roi_width // 2)
        y2 = min(h, center_y + roi_height // 2)
        
        # Recorta ROI
        roi = image[y1:y2, x1:x2]
        
        # Aplica threshold binário (pixels pretos)
        _, binary_roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Cria máscara completa
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = binary_roi
        
        self.log(f"   📐 ROI Fixa: ({x1},{y1}) a ({x2},{y2}), tamanho {roi_width}x{roi_height}")
        return mask
    
    def segment_spatial_mask_fixed(self, image, mask_width=200, mask_height=200, threshold=30, shape='rectangular'):
        """
        2. Máscara Espacial Fixa (retangular ou elíptica)
        Aplica uma máscara no centro e ignora o resto.
        """
        h, w = image.shape
        center_x, center_y = w // 2, h // 2
        
        # Cria máscara espacial
        spatial_mask = np.zeros((h, w), dtype=np.uint8)
        
        if shape == 'rectangular':
            x1 = max(0, center_x - mask_width // 2)
            y1 = max(0, center_y - mask_height // 2)
            x2 = min(w, center_x + mask_width // 2)
            y2 = min(h, center_y + mask_height // 2)
            spatial_mask[y1:y2, x1:x2] = 255
        else:  # elíptica
            cv2.ellipse(spatial_mask, (center_x, center_y), 
                       (mask_width // 2, mask_height // 2), 0, 0, 360, 255, -1)
        
        # Aplica threshold na imagem
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Combina: só pixels pretos dentro da máscara espacial
        mask = cv2.bitwise_and(binary, spatial_mask)
        
        self.log(f"   🎭 Máscara Espacial {shape}: {mask_width}x{mask_height}")
        return mask
    
    def segment_connected_components(self, image, threshold=30, min_area=100, max_components=5):
        """
        3. Componentes Conexos (Connected Component Labeling)
        Rotula blocos pretos e escolhe os do centro pelo tamanho + posição.
        """
        h, w = image.shape
        center_x, center_y = w // 2, h // 2
        
        # Threshold binário
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Rotula componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Calcula score para cada componente (tamanho + proximidade do centro)
        component_scores = []
        for i in range(1, num_labels):  # Ignora fundo (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            cx, cy = centroids[i]
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            
            # Score: maior área e mais próximo do centro
            score = area / (1 + dist_from_center / 100)
            component_scores.append((i, score, area, dist_from_center))
        
        # Ordena por score e pega os melhores
        component_scores.sort(key=lambda x: x[1], reverse=True)
        selected_labels = [x[0] for x in component_scores[:max_components]]
        
        # Cria máscara com componentes selecionados
        mask = np.zeros((h, w), dtype=np.uint8)
        for label in selected_labels:
            mask[labels == label] = 255
        
        self.log(f"   🔗 Componentes Conexos: {len(selected_labels)} componentes selecionados")
        return mask
    
    def segment_centroid_based(self, image, threshold=30, brain_mask=None):
        """
        4. Seleção por Centróide / Prior Espacial
        Calcula centróide do cérebro e pega componentes pretos mais próximos.
        """
        h, w = image.shape
        
        # Se não tiver máscara do cérebro, usa threshold adaptativo
        if brain_mask is None:
            _, brain_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        
        # Calcula centróide do cérebro
        moments = cv2.moments(brain_mask)
        if moments["m00"] != 0:
            brain_cx = int(moments["m10"] / moments["m00"])
            brain_cy = int(moments["m01"] / moments["m00"])
        else:
            brain_cx, brain_cy = w // 2, h // 2
        
        # Threshold para pixels pretos
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Seleciona componentes próximos do centróide
        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, num_labels):
            cx, cy = centroids[i]
            dist = np.sqrt((cx - brain_cx)**2 + (cy - brain_cy)**2)
            
            # Se estiver dentro de um raio do centróide
            if dist < min(w, h) * 0.3:  # 30% da dimensão menor
                mask[labels == i] = 255
        
        self.log(f"   📍 Centróide: ({brain_cx}, {brain_cy})")
        return mask
    
    def segment_hole_filling(self, image, threshold=30):
        """
        5. Preenchimento de Buracos (Binary Hole Filling / Hole Detection)
        Detecta cavidades internas automaticamente dentro da massa branca.
        """
        # Threshold: branco = cérebro, preto = fundo
        _, binary_white = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Inverte: agora preto = cérebro, branco = fundo
        binary_inv = cv2.bitwise_not(binary_white)
        
        # Preenche buracos (cavidades pretas dentro do branco)
        # Flood fill do fundo
        h, w = binary_inv.shape
        mask_filled = binary_inv.copy()
        
        # Preenche bordas primeiro
        cv2.floodFill(mask_filled, None, (0, 0), 255)
        cv2.floodFill(mask_filled, None, (w-1, 0), 255)
        cv2.floodFill(mask_filled, None, (0, h-1), 255)
        cv2.floodFill(mask_filled, None, (w-1, h-1), 255)
        
        # Buracos são o que ficou preto (não foi preenchido)
        holes = cv2.bitwise_not(mask_filled)
        
        self.log(f"   🕳️ Preenchimento de Buracos: detectados")
        return holes
    
    def segment_flood_fill_background(self, image, threshold=30):
        """
        6. Flood Fill do Fundo + Inversão
        Separa fundo de buracos internos.
        """
        h, w = image.shape
        
        # Threshold binário
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Flood fill do fundo (bordas)
        mask = binary.copy()
        cv2.floodFill(mask, None, (0, 0), 0)
        cv2.floodFill(mask, None, (w-1, 0), 0)
        cv2.floodFill(mask, None, (0, h-1), 0)
        cv2.floodFill(mask, None, (w-1, h-1), 0)
        
        # O que não foi preenchido são os buracos internos
        holes = cv2.bitwise_not(mask)
        holes = cv2.bitwise_and(holes, binary)  # Só dentro da região branca
        
        self.log(f"   🌊 Flood Fill do Fundo: aplicado")
        return holes
    
    def segment_distance_transform(self, image, threshold=30, brain_mask=None):
        """
        7. Transformada de Distância + Limiar no Interior
        Pega só a região mais interna da máscara do cérebro e extrai o preto dali.
        """
        h, w = image.shape
        
        # Máscara do cérebro (se não fornecida)
        if brain_mask is None:
            _, brain_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        
        # Transformada de distância (distância de cada pixel até a borda)
        dist_transform = cv2.distanceTransform(brain_mask, cv2.DIST_L2, 5)
        
        # Normaliza
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Pega apenas região mais interna (últimos 30% da distância)
        _, inner_region = cv2.threshold(dist_transform, int(255 * 0.7), 255, cv2.THRESH_BINARY)
        
        # Dentro dessa região, pega pixels pretos
        _, black_pixels = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Combina: buracos pretos dentro da região interna
        mask = cv2.bitwise_and(black_pixels, inner_region)
        
        self.log(f"   📏 Transformada de Distância: região interna")
        return mask
    
    def segment_watershed_markers(self, image, threshold=30, num_markers=3):
        """
        8. Watershed por Marcadores (Marker-based Watershed)
        Bom quando os buracos pretos se tocam e precisa dividir.
        """
        h, w = image.shape
        
        # Threshold binário
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Operação morfológica para separar objetos que se tocam
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=3)
        
        # Encontra região certa (sure foreground)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Região desconhecida
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marcadores
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(image_color, markers)
        
        # Cria máscara (marcadores > 1 são objetos)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[markers > 1] = 255
        
        self.log(f"   🌊 Watershed com Marcadores: {num_markers} marcadores")
        return mask
    
    def segment_active_contours(self, image, threshold=30, init_contour=None):
        """
        9. Active Contours / Snakes
        Contornos ativos que se ajustam às bordas.
        """
        # Threshold para criar imagem binária
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Se não tiver contorno inicial, cria um circular no centro
        h, w = image.shape
        if init_contour is None:
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 4
            theta = np.linspace(0, 2*np.pi, 100)
            init_contour = np.array([[center_x + radius * np.cos(t), 
                                     center_y + radius * np.sin(t)] for t in theta], dtype=np.int32)
        
        # Aplica active contours (simplificado - usa findContours como aproximação)
        # Em implementação completa, usaria scipy ou implementação própria
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontra contorno mais próximo do inicial
        if len(contours) > 0:
            # Seleciona maior contorno interno
            mask = np.zeros((h, w), dtype=np.uint8)
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        self.log(f"   🐍 Active Contours: aplicado")
        return mask
    
    def apply_alternative_segmentation(self, image, method='roi_fixed', **kwargs):
        """
        Aplica método alternativo de segmentação quando Region Growing falha.
        
        Args:
            image: numpy array 2D (grayscale)
            method: nome do método a usar
            **kwargs: parâmetros específicos do método
            
        Returns:
            mask: numpy array 2D binário
        """
        self.log(f"\n🔄 Aplicando método alternativo: {method}")
        
        method_map = {
            'roi_fixed': self.segment_roi_fixed_rectangle,
            'spatial_mask': self.segment_spatial_mask_fixed,
            'connected_components': self.segment_connected_components,
            'centroid_based': self.segment_centroid_based,
            'hole_filling': self.segment_hole_filling,
            'flood_fill': self.segment_flood_fill_background,
            'distance_transform': self.segment_distance_transform,
            'watershed_markers': self.segment_watershed_markers,
            'active_contours': self.segment_active_contours,
        }
        
        if method not in method_map:
            self.log(f"   ⚠️ Método '{method}' não encontrado, usando 'roi_fixed'")
            method = 'roi_fixed'
        
        try:
            mask = method_map[method](image, **kwargs)
            num_pixels = np.sum(mask == 255)
            self.log(f"   ✅ Método alternativo concluído: {num_pixels} pixels")
            return mask
        except Exception as e:
            self.log(f"   ❌ Erro no método alternativo: {e}")
            # Fallback para ROI fixa
            return self.segment_roi_fixed_rectangle(image)

    def test_segmentation_with_current_params(self):
        """
        Testa a segmentação usando os parâmetros ajustáveis atuais.
        Usa os seeds fixos e os parâmetros da interface.
        """
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Por favor, carregue uma imagem primeiro.")
            return
        
        self.log("\n" + "="*80)
        self.log("🧪 TESTE DE SEGMENTAÇÃO COM PARÂMETROS ATUAIS")
        self.log("="*80)
        self.log(f"📌 Seeds: {self.auto_seed_points}")
        self.log(f"🎚️ Threshold: {self.region_growing_threshold}")
        self.log(f"🔧 Kernel: {self.morphology_kernel_size}x{self.morphology_kernel_size}")
        self.log(f"🔹 Abertura: {'✅' if self.apply_opening else '❌'}")
        self.log(f"🔹 Fechamento: {'✅' if self.apply_closing else '❌'}")
        self.log(f"🔹 Preencher: {'✅' if self.apply_fill_holes else '❌'}")
        self.log(f"🔹 Suavizar: {'✅' if self.apply_smooth_contours else '❌'}")
        self.log(f"🖼️ Usando: Janela 2 (Pré-processada)")
        
        # SEMPRE USA A IMAGEM PRÉ-PROCESSADA (JANELA 2) se disponível
        if self.preprocessed_image is not None:
            img_np = np.array(self.preprocessed_image.convert('L'))
            self.log("✅ Usando imagem da Janela 2 (Filtrada)")
        else:
            img_np = np.array(self.original_image.convert('L'))
            self.log("⚠️ Usando imagem original - Aplique um filtro primeiro!")
        
        # Prepara imagem para segmentação (aplica CLAHE adicional)
        img_for_segmentation = self.prepare_image_for_segmentation(img_np)
        
        # Aplica Region Growing em cada seed point e combina as máscaras
        self.log(f"\n🎯 Aplicando Region Growing em {len(self.auto_seed_points)} seed points:")
        combined_mask = None
        
        for i, (seed_x, seed_y) in enumerate(self.auto_seed_points, 1):
            self.log(f"\n   Seed {i}/{len(self.auto_seed_points)}: ({seed_x}, {seed_y})")
            
            # Verifica se o seed está dentro da imagem
            if seed_x < 0 or seed_y < 0 or seed_x >= img_np.shape[1] or seed_y >= img_np.shape[0]:
                self.log(f"   ⚠ Seed fora dos limites da imagem! Ignorando...")
                continue
            
            # Aplica region growing neste seed
            mask = self.region_growing(img_for_segmentation, (seed_x, seed_y), 
                                      threshold=self.region_growing_threshold,
                                      connectivity=self.connectivity_var.get())
            
            num_pixels = np.sum(mask == 255)
            self.log(f"   ✓ {num_pixels} pixels segmentados")
            
            # Combina com a máscara acumulada
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        if combined_mask is None:
            self.log("\n❌ Nenhum seed válido! Segmentação falhou.")
            messagebox.showerror("Erro", "Nenhum seed point válido para segmentação.")
            return

        # Aplica pós-processamento morfológico
        self.log("\n🔬 Aplicando pós-processamento morfológico:")
        final_mask = self.apply_morphological_postprocessing(combined_mask)
        
        # VALIDAÇÃO: Verifica se a segmentação não excedeu o limite
        is_valid, num_pixels = self.validate_segmentation_mask(final_mask, "(TESTE)")
        if not is_valid:
            self.log(f"\n🔄 Region Growing falhou ({num_pixels} pixels). Aplicando método alternativo...")
            # Usa método alternativo
            final_mask = self.apply_alternative_segmentation(
                img_for_segmentation, 
                method=self.alternative_segmentation_method,
                threshold=30
            )
            # Valida novamente
            is_valid_alt, num_pixels_alt = self.validate_segmentation_mask(final_mask, "(ALTERNATIVO)")
            if is_valid_alt:
                self.log(f"   ✅ Método alternativo funcionou! {num_pixels_alt} pixels")
            else:
                self.log(f"   ⚠️ Método alternativo também excedeu limite ({num_pixels_alt} pixels)")
        
        self.image_mask = final_mask

        # Cria visualização: imagem original em RGB com contorno AMARELO
        img_with_contour = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Encontra contornos na máscara final
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos pequenos (ruído)
        min_area = 50
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Desenha contornos em VERMELHO VIVO (BGR: 0, 0, 255)
        cv2.drawContours(img_with_contour, large_contours, -1, (0, 0, 255), 3)
        
        # Converte para PIL e exibe no canvas segmented
        segmented_pil = Image.fromarray(img_with_contour)
        self.segmented_image = segmented_pil
        self.display_image(segmented_pil, self.canvas_segmented, "segmented")
        
        # Exibe estatísticas finais
        total_pixels = np.sum(final_mask == 255)
        total_regions = len(large_contours)
        
        self.log(f"\n✅ Segmentação concluída!")
        self.log(f"   • Total de pixels segmentados: {total_pixels}")
        self.log(f"   • Número de regiões: {total_regions}")
        self.log(f"   • Área percentual: {(total_pixels / (img_np.shape[0] * img_np.shape[1]) * 100):.2f}%")
        self.log("="*80 + "\n")
        
        self.lbl_segment_status.config(text=f"✅ Teste: {total_regions} região(ões) | {total_pixels} pixels")

    def segment_ventricles(self):
        """ Executa a segmentação AUTOMÁTICA dos ventrículos usando Region Growing com múltiplos seeds. """
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        self.log("="*60)
        self.log("🔬 SEGMENTAÇÃO AUTOMÁTICA DOS VENTRÍCULOS")
        self.log("="*60)
        self.log(f"Método: Region Growing Multi-Seed")
        self.log(f"Seeds fixos: {self.auto_seed_points}")
        self.log(f"Threshold: {self.region_growing_threshold}")
        self.log(f"Kernel morfológico: {self.morphology_kernel_size}x{self.morphology_kernel_size}")
        self.log("-"*60)

        # SEMPRE USA A IMAGEM PRÉ-PROCESSADA (JANELA 2) se disponível, senão usa a original
        if self.preprocessed_image is not None:
            img_np = np.array(self.preprocessed_image)
            self.log("🎨 Usando imagem PRÉ-PROCESSADA (Janela 2) para segmentação")
        else:
            img_np = np.array(self.original_image)
            self.log("⚠️ Usando imagem ORIGINAL (sem filtros) - Aplique um filtro primeiro para melhores resultados")
        
        # Prepara imagem para segmentação (CLAHE ou Otsu baseado na escolha)
        img_for_segmentation = self.prepare_image_for_segmentation(img_np)

        # Aplica Region Growing em cada seed point e combina as máscaras
        self.log(f"\n🎯 Aplicando Region Growing em {len(self.auto_seed_points)} seed points:")
        combined_mask = None
        
        for i, (seed_x, seed_y) in enumerate(self.auto_seed_points, 1):
            self.log(f"\n   Seed {i}/{len(self.auto_seed_points)}: ({seed_x}, {seed_y})")
            
            # Verifica se o seed está dentro da imagem
            if seed_x < 0 or seed_y < 0 or seed_x >= img_np.shape[1] or seed_y >= img_np.shape[0]:
                self.log(f"   ⚠ Seed fora dos limites da imagem! Ignorando...")
                continue
            
            # Aplica region growing neste seed
            mask = self.region_growing(img_for_segmentation, (seed_x, seed_y), 
                                      threshold=self.region_growing_threshold,
                                      connectivity=self.connectivity_var.get())
            
            num_pixels = np.sum(mask == 255)
            self.log(f"   ✓ {num_pixels} pixels segmentados")
            
            # Combina com a máscara acumulada
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        if combined_mask is None:
            self.log("\n❌ Nenhum seed válido! Segmentação falhou.")
            messagebox.showerror("Erro", "Nenhum seed point válido para segmentação.")
            return

        # Aplica pós-processamento morfológico
        self.log("\n🔬 Aplicando pós-processamento morfológico:")
        final_mask = self.apply_morphological_postprocessing(combined_mask)
        
        # VALIDAÇÃO: Verifica se a segmentação não excedeu o limite
        is_valid, num_pixels_final = self.validate_segmentation_mask(final_mask, "(AUTOMÁTICA)")
        if not is_valid:
            self.log(f"\n🔄 Region Growing falhou ({num_pixels_final} pixels). Aplicando método alternativo...")
            # Usa método alternativo
            final_mask = self.apply_alternative_segmentation(
                img_for_segmentation, 
                method=self.alternative_segmentation_method,
                threshold=30
            )
            # Valida novamente
            is_valid_alt, num_pixels_alt = self.validate_segmentation_mask(final_mask, "(ALTERNATIVO)")
            if is_valid_alt:
                self.log(f"   ✅ Método alternativo funcionou! {num_pixels_alt} pixels")
            else:
                self.log(f"   ⚠️ Método alternativo também excedeu limite ({num_pixels_alt} pixels)")
        
        self.image_mask = final_mask

        # Extrai descritores morfológicos
        image_id = os.path.basename(self.image_path) if self.image_path else "imagem_atual"
        self.log(f"\n📊 Extraindo descritores morfológicos para: {image_id}")
        descriptors = self.extrair_descritores_ventriculo(final_mask, image_id=image_id)
        if descriptors:
            self.descriptors_list.append(descriptors)
            self.log(f"   ✅ Descritores adicionados à lista (total: {len(self.descriptors_list)})")

        # Cria visualização: imagem original em RGB com contorno AMARELO
        img_with_contour = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Encontra contornos na máscara final
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos pequenos (ruído)
        min_area = 50
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Desenha contornos em VERMELHO VIVO (BGR: 0, 0, 255)
        cv2.drawContours(img_with_contour, large_contours, -1, (0, 0, 255), 3)
        
        # Converte para PIL e exibe no canvas segmented
        self.segmented_image = Image.fromarray(img_with_contour)
        self.display_image(self.segmented_image, self.canvas_segmented, "segmented")
        total_area = sum([cv2.contourArea(cnt) for cnt in large_contours])
        
        self.log("-"*60)
        self.log(f"✅ SEGMENTAÇÃO CONCLUÍDA COM SUCESSO!")
        self.log(f"   • Regiões encontradas: {len(large_contours)}")
        self.log(f"   • Pixels segmentados: {num_pixels_final}")
        self.log(f"   • Área total: {total_area:.2f} pixels²")
        self.log("="*60 + "\n")
        
        # Atualiza status na interface
        self.lbl_segment_status.config(
            text=f"✓ {len(large_contours)} região(ões) | {num_pixels_final} pixels",
            foreground="green"
        )
        
    def extract_features(self):
        """ Extrai as 6 características dos ventrículos segmentados. [cite: 79] """
        if self.image_mask is None:
            messagebox.showwarning("Aviso", "Execute a segmentação primeiro.")
            return

        self.log("Iniciando extração de características...")
        
        # TODO: Esta função deve ser executada em *todo* o dataset,
        # não apenas em uma imagem.
        # O fluxo correto seria:
        # 1. Iterar por todas as imagens do dataset
        # 2. Chamar segment_ventricles() para cada uma
        # 3. Chamar esta função (adaptada) para extrair features
        # 4. Salvar tudo em um novo DataFrame (self.features_df)
        
        # Por enquanto, extrai da imagem carregada:
        contours, _ = cv2.findContours(self.image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume o maior contorno (ou os dois maiores, para 2 ventrículos)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if not contours:
            self.log("Nenhum contorno encontrado na máscara.")
            return

        # TODO: Lógica para combinar os N maiores contornos (ventrículos)
        # Aqui, vamos usar apenas o MAIOR contorno para simplificar
        cnt = contours[0]
        
        try:
            # 1. Área [cite: 79]
            area = cv2.contourArea(cnt)
            
            # 2. Circularidade [cite: 79]
            perimeter = cv2.arcLength(cnt, True)
            circularity = 0
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter**2)
            
            # 3. Excentricidade [cite: 79]
            eccentricity = 0
            if len(cnt) >= 5: # fitEllipse precisa de pelo menos 5 pontos
                (x,y), (MA,ma), angle = cv2.fitEllipse(cnt)
                a = ma / 2
                b = MA / 2
                if a > 0 and b > 0:
                    # f = sqrt(a^2 - b^2) -> eccentricity = f/a
                    eccentricity = np.sqrt(1 - (b**2 / a**2))
            
            # 4. Característica Extra 1 (Ex: Extensão)
            rect_x, rect_y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            extent = 0
            if rect_area > 0:
                extent = area / rect_area
                
            # 5. Característica Extra 2 (Ex: Solidez)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = 0
            if hull_area > 0:
                solidity = area / hull_area
            
            # 6. Característica Extra 3 (Ex: Diâmetro Equivalente)
            equiv_diameter = np.sqrt(4 * area / np.pi)

            self.log("Características extraídas (para o maior contorno):")
            self.log(f"  Área: {area:.2f}")
            self.log(f"  Circularidade: {circularity:.4f}")
            self.log(f"  Excentricidade: {eccentricity:.4f}")
            self.log(f"  Extensão: {extent:.4f}")
            self.log(f"  Solidez: {solidity:.4f}")
            self.log(f"  Diâmetro Eq.: {equiv_diameter:.2f}")

            # TODO: Salvar em uma planilha complementar [cite: 80]
            # Ex: self.features_df.loc[self.image_path] = [area, circularity, ...]
            # E depois: self.features_df.to_csv("features_extra.csv")

        except Exception as e:
            self.log(f"Erro ao extrair características: {e}")
            messagebox.showerror("Erro na Extração", f"Erro: {e}")

    def extrair_descritores_ventriculo(self, mask, image_id=None):
        """
        Extrai descritores morfológicos do ventrículo segmentado.
        
        Args:
            mask: numpy array 2D binário (0=fundo, 255=região ou 0=fundo, 1=região)
            image_id: identificador da imagem (nome do arquivo ou índice)
            
        Returns:
            dict: dicionário com os descritores ou None se falhar
        """
        try:
            # Garante que mask seja binária (0/1)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Normaliza para 0/1 (não 0/255)
            if mask.max() > 1:
                mask = (mask > 127).astype(np.uint8)
            
            # Detecta se a máscara está invertida
            # Se pixels 1 forem a maior parte da imagem, provavelmente está invertido
            total_pixels = mask.size
            white_pixels = np.sum(mask == 1)
            white_ratio = white_pixels / total_pixels
            
            if white_ratio > 0.5:
                # Máscara provavelmente invertida (ventrículo deveria ser menor)
                mask = 1 - mask
                self.log(f"   ⚠ Máscara invertida detectada (ratio={white_ratio:.2f}), corrigindo...")
            
            # Rotula componentes conectados
            labeled_mask = measure.label(mask, connectivity=2)
            regions = measure.regionprops(labeled_mask)
            
            if len(regions) == 0:
                self.log(f"   ⚠ Nenhum componente encontrado na máscara")
                return None
            
            # Seleciona o(s) componente(s) do ventrículo
            # Se houver mais de um, une todos (ou pega o maior)
            if len(regions) > 1:
                # Ordena por área (maior primeiro)
                regions = sorted(regions, key=lambda r: r.area, reverse=True)
                # Pega o maior componente (ou pode unir todos se necessário)
                largest_region = regions[0]
                self.log(f"   ℹ {len(regions)} componentes encontrados, usando o maior (área={largest_region.area})")
            else:
                largest_region = regions[0]
            
            # Calcula descritores usando regionprops
            region = largest_region
            
            # 1. Área (A)
            area = float(region.area)
            
            # 2. Perímetro (P)
            perimeter = float(region.perimeter)
            
            # 3. Circularidade (C = 4 * pi * A / P²)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0.0
            
            # 4. Excentricidade (Ecc)
            eccentricity = float(region.eccentricity)
            
            # 5. Solidez (Solidity = A / convex_area)
            solidity = float(region.solidity)
            
            # 6. Extent (Retangularidade = A / área da bounding box)
            extent = float(region.extent)
            
            # 7. Aspect Ratio (AR = major_axis_length / minor_axis_length)
            if region.minor_axis_length > 0:
                aspect_ratio = float(region.major_axis_length / region.minor_axis_length)
            else:
                aspect_ratio = 0.0
            
            # Cria dicionário com os descritores
            descriptors = {
                'id': image_id if image_id is not None else 'unknown',
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'eccentricity': eccentricity,
                'solidity': solidity,
                'extent': extent,
                'aspect_ratio': aspect_ratio
            }
            
            # Log dos valores
            self.log(f"   📊 Descritores extraídos:")
            self.log(f"      Área: {area:.2f}")
            self.log(f"      Perímetro: {perimeter:.2f}")
            self.log(f"      Circularidade: {circularity:.4f}")
            self.log(f"      Excentricidade: {eccentricity:.4f}")
            self.log(f"      Solidez: {solidity:.4f}")
            self.log(f"      Extent: {extent:.4f}")
            self.log(f"      Aspect Ratio: {aspect_ratio:.4f}")
            
            return descriptors
            
        except Exception as e:
            self.log(f"   ❌ Erro ao extrair descritores: {e}")
            return None
    
    def salvar_descritores_csv(self, output_dir=None):
        """
        Salva os descritores acumulados em um arquivo CSV.
        
        Args:
            output_dir: diretório de saída (se None, usa diretório raiz)
        """
        if len(self.descriptors_list) == 0:
            self.log("⚠ Nenhum descritor para salvar.")
            return
        
        try:
            # Cria DataFrame
            df = pd.DataFrame(self.descriptors_list)
            
            # Remove sufixo "_axl" dos IDs
            if 'id' in df.columns:
                df['id'] = df['id'].astype(str).str.replace('_axl$', '', regex=True)
            
            # Renomeia coluna 'id' para 'MRI ID'
            df = df.rename(columns={'id': 'MRI ID'})
            
            # Garante ordem das colunas
            columns_order = ['MRI ID', 'area', 'perimeter', 'circularity', 'eccentricity', 
                           'solidity', 'extent', 'aspect_ratio']
            df = df[columns_order]
            
            # Converte colunas numéricas para string com vírgula como separador decimal
            numeric_columns = ['area', 'perimeter', 'circularity', 'eccentricity', 
                             'solidity', 'extent', 'aspect_ratio']
            for col in numeric_columns:
                if col in df.columns:
                    # Formata números com vírgula como separador decimal, mantendo precisão
                    if col in ['area', 'perimeter']:
                        # Área e perímetro: 2 casas decimais
                        df[col] = df[col].apply(lambda x: f"{float(x):.2f}".replace('.', ',') if pd.notna(x) else '')
                    elif col in ['circularity', 'eccentricity', 'solidity', 'extent']:
                        # Circularidade, excentricidade, solidez, extent: 4 casas decimais
                        df[col] = df[col].apply(lambda x: f"{float(x):.4f}".replace('.', ',') if pd.notna(x) else '')
                    elif col == 'aspect_ratio':
                        # Aspect ratio: 4 casas decimais
                        df[col] = df[col].apply(lambda x: f"{float(x):.4f}".replace('.', ',') if pd.notna(x) else '')
            
            # Define caminho de saída
            if output_dir is None:
                output_path = "descritores.csv"
            else:
                output_path = os.path.join(output_dir, "descritores.csv")
            
            # Salva CSV com ponto e vírgula como separador
            df.to_csv(output_path, index=False, sep=';')
            
            self.log(f"\n✅ CSV de descritores salvo: {output_path}")
            self.log(f"   Total de registros: {len(df)}")
            self.log(f"   Colunas: {', '.join(columns_order)}")
            
            return output_path
            
        except Exception as e:
            self.log(f"❌ Erro ao salvar CSV de descritores: {e}")
            return None
    
    def merge_csv_files(self, demographic_csv_path=None, descriptors_csv_path=None, output_path=None, show_messagebox=True):
        """
        Faz merge dos CSVs de demografia e descritores baseado na coluna MRI ID.
        
        Args:
            demographic_csv_path: caminho do CSV de demografia (oasis_longitudinal_demographic.csv)
            descriptors_csv_path: caminho do CSV de descritores (descritores.csv)
            output_path: caminho de saída para o CSV merged (merged_data.csv)
            show_messagebox: se True, mostra messagebox em caso de erro/sucesso (padrão: True)
        """
        try:
            # Define caminhos padrão se não fornecidos
            if demographic_csv_path is None:
                demographic_csv_path = "oasis_longitudinal_demographic.csv"
            if descriptors_csv_path is None:
                descriptors_csv_path = os.path.join("output", "descritores.csv")
            if output_path is None:
                output_path = "merged_data.csv"
            
            self.log("\n" + "="*80)
            self.log("🔄 MERGE DE CSVs")
            self.log("="*80)
            self.log(f"📄 CSV Demografia: {demographic_csv_path}")
            self.log(f"📄 CSV Descritores: {descriptors_csv_path}")
            self.log(f"📄 CSV Saída: {output_path}")
            self.log("-"*80)
            
            # Verifica se os arquivos existem
            if not os.path.exists(demographic_csv_path):
                self.log(f"❌ Arquivo não encontrado: {demographic_csv_path}")
                if show_messagebox:
                    messagebox.showerror("Erro", f"Arquivo não encontrado: {demographic_csv_path}")
                return None
            
            if not os.path.exists(descriptors_csv_path):
                self.log(f"❌ Arquivo não encontrado: {descriptors_csv_path}")
                if show_messagebox:
                    messagebox.showerror("Erro", f"Arquivo não encontrado: {descriptors_csv_path}")
                return None
            
            # Lê os CSVs
            self.log("📖 Lendo CSVs...")
            
            # Detecta separador do CSV de demografia
            with open(demographic_csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                demographic_sep = ';' if ';' in first_line else ','
            
            # Lê CSV de demografia
            df_demographic = pd.read_csv(demographic_csv_path, sep=demographic_sep, encoding='utf-8')
            self.log(f"   ✓ Demografia: {len(df_demographic)} registros, {len(df_demographic.columns)} colunas")
            
            # Lê CSV de descritores (sempre usa ; como separador)
            df_descriptors = pd.read_csv(descriptors_csv_path, sep=';', encoding='utf-8')
            self.log(f"   ✓ Descritores: {len(df_descriptors)} registros, {len(df_descriptors.columns)} colunas")
            
            # Verifica se a coluna MRI ID existe em ambos
            if 'MRI ID' not in df_demographic.columns:
                self.log("❌ Coluna 'MRI ID' não encontrada no CSV de demografia")
                self.log(f"   Colunas disponíveis: {list(df_demographic.columns)}")
                if show_messagebox:
                    messagebox.showerror("Erro", "Coluna 'MRI ID' não encontrada no CSV de demografia")
                return None
            
            if 'MRI ID' not in df_descriptors.columns:
                self.log("❌ Coluna 'MRI ID' não encontrada no CSV de descritores")
                self.log(f"   Colunas disponíveis: {list(df_descriptors.columns)}")
                if show_messagebox:
                    messagebox.showerror("Erro", "Coluna 'MRI ID' não encontrada no CSV de descritores")
                return None
            
            # Remove espaços em branco dos IDs (se houver)
            df_demographic['MRI ID'] = df_demographic['MRI ID'].astype(str).str.strip()
            df_descriptors['MRI ID'] = df_descriptors['MRI ID'].astype(str).str.strip()
            
            # Faz o merge
            self.log("\n🔗 Fazendo merge baseado em 'MRI ID'...")
            merged_df = pd.merge(df_demographic, df_descriptors, on='MRI ID', how='inner')
            
            self.log(f"   ✓ Merge concluído: {len(merged_df)} registros combinados")
            self.log(f"   ✓ Colunas totais: {len(merged_df.columns)}")
            
            # Estatísticas do merge
            self.log(f"\n📊 Estatísticas do merge:")
            self.log(f"   • Registros no CSV demografia: {len(df_demographic)}")
            self.log(f"   • Registros no CSV descritores: {len(df_descriptors)}")
            self.log(f"   • Registros após merge: {len(merged_df)}")
            
            # IDs que não foram combinados
            ids_demographic = set(df_demographic['MRI ID'].unique())
            ids_descriptors = set(df_descriptors['MRI ID'].unique())
            ids_merged = set(merged_df['MRI ID'].unique())
            
            ids_only_demographic = ids_demographic - ids_merged
            ids_only_descriptors = ids_descriptors - ids_merged
            
            if ids_only_demographic:
                self.log(f"   ⚠ IDs apenas em demografia (não combinados): {len(ids_only_demographic)}")
                if len(ids_only_demographic) <= 10:
                    self.log(f"      {list(ids_only_demographic)}")
            
            if ids_only_descriptors:
                self.log(f"   ⚠ IDs apenas em descritores (não combinados): {len(ids_only_descriptors)}")
                if len(ids_only_descriptors) <= 10:
                    self.log(f"      {list(ids_only_descriptors)}")
            
            # Salva o CSV merged
            # Usa o mesmo separador do CSV de demografia
            merged_df.to_csv(output_path, index=False, sep=demographic_sep, encoding='utf-8')
            
            self.log(f"\n✅ CSV merged salvo: {output_path}")
            self.log(f"   Total de registros: {len(merged_df)}")
            self.log(f"   Total de colunas: {len(merged_df.columns)}")
            self.log("="*80 + "\n")
            
            if show_messagebox:
                messagebox.showinfo(
                    "Merge Concluído",
                    f"CSV merged salvo com sucesso!\n\n"
                    f"Arquivo: {output_path}\n"
                    f"Registros: {len(merged_df)}\n"
                    f"Colunas: {len(merged_df.columns)}"
                )
            
            return output_path
            
        except Exception as e:
            error_msg = f"Erro ao fazer merge dos CSVs: {e}"
            self.log(f"❌ {error_msg}")
            if show_messagebox:
                messagebox.showerror("Erro", error_msg)
            import traceback
            self.log(traceback.format_exc())
            return None

    def show_scatterplot(self):
        """ Gera gráficos de dispersão (scatterplots). [cite: 81] """
        if self.features_df is None or self.dataframe is None:
            messagebox.showwarning("Aviso", "Carregue o CSV e extraia características primeiro.")
            self.log("Gere o DataFrame de features (self.features_df) primeiro.")
            return

        self.log("Gerando scatterplot...")

        # TODO: Unir self.features_df com self.dataframe (CSV original)
        # merged_df = pd.merge(self.dataframe, self.features_df, ...)
        merged_df = None # Placeholder
        
        # Exemplo de dados (SUBSTITUIR PELO MERGED_DF)
        # Você precisa ter um DataFrame com colunas: 'Area', 'Circularidade', 'Group'
        # merged_df = ...
        
        # --- Placeholder ---
        self.log("TODO: Implementar lógica de merge e plot.")
        messagebox.showinfo("TODO", "Implementar merge do DataFrame de features com o CSV e plotar com Matplotlib.")
        # --- Fim do Placeholder ---

        # Exemplo de código de plotagem (quando 'merged_df' existir):
        #
        # feature1 = 'Area'
        # feature2 = 'Circularidade'
        #
        # plt.figure(figsize=(8, 6))
        #
        # Cores [cite: 83]
        # colors = {'Nondemented': 'blue', 'Demented': 'red', 'Converted': 'black'}
        #
        # for group, color in colors.items():
        #     subset = merged_df[merged_df['Group'] == group]
        #     plt.scatter(subset[feature1], subset[feature2], c=color, label=group, alpha=0.6)
        #
        # plt.xlabel(feature1)
        # plt.ylabel(feature2)
        # plt.title(f'{feature1} vs {feature2}')
        # plt.legend()
        # plt.grid(True)
        # plt.show() # Abre em nova janela

    # --- 5. FUNÇÕES DE MACHINE LEARNING ---

    def prepare_data(self):
        """ Prepara os dados para os modelos de ML/DL. [cite: 88-93] """
        if self.dataframe is None:
            self.log("Carregue o CSV primeiro.")
            return None
        
        self.log("Preparando dados...")
        
        # 1. Criar a coluna 'Patient_ID' (o PDF não especifica, assumindo 'Subject ID' do CSV)
        # Se o seu CSV tiver um ID de sujeito, use-o.
        # Vamos assumir que o 'Subject ID' está no CSV.
        if 'Subject ID' not in self.dataframe.columns:
            self.log("ERRO: Coluna 'Subject ID' não encontrada no CSV. Necessária para divisão de pacientes.")
            return None
        
        # 2. Ajustar classes (Item 9) [cite: 90]
        # CUIDADO: Este apply() pode precisar de ajustes se 'CDR' tiver NaNs
        def map_class(row):
            if row['Group'] == 'Nondemented':
                return 'NonDemented'
            if row['Group'] == 'Demented':
                return 'Demented'
            if row['Group'] == 'Converted':
                return 'Demented' if row['CDR'] > 0 else 'NonDemented'
            return None
        
        df_copy = self.dataframe.copy()
        df_copy['Target_Class'] = df_copy.apply(map_class, axis=1)
        df_copy = df_copy.dropna(subset=['Target_Class']) # Remove linhas que não são de nenhuma classe
        
        # 3. Dividir em Treino/Teste (baseado em pacientes) [cite: 88, 93]
        # 80% treino, 20% teste
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        patient_ids = df_copy['Subject ID']
        groups = df_copy['Subject ID']
        
        # Garante divisão estratificada por classe [cite: 91]
        # Isso é complexo com grupos. O GroupShuffleSplit não suporta 'stratify'
        # A forma mais simples é estratificar os *pacientes*
        # TODO: Implementar estratificação por paciente
        
        # Divisão simples por grupo:
        train_idx, test_idx = next(gss.split(df_copy, groups=groups))
        
        train_patients = df_copy.iloc[train_idx]['Subject ID'].unique()
        test_patients = df_copy.iloc[test_idx]['Subject ID'].unique()

        train_df = df_copy[df_copy['Subject ID'].isin(train_patients)]
        test_df = df_copy[df_copy['Subject ID'].isin(test_patients)]

        # 4. Dividir Treino em Treino/Validação (Item 9) [cite: 92]
        # 20% do treino vira validação
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42) # 20% de 80%
        train_val_idx, val_idx = next(gss_val.split(train_df, groups=train_df['Subject ID']))
        
        val_df = train_df.iloc[val_idx]
        train_df = train_df.iloc[train_val_idx] # Redefine o train_df

        self.log(f"Dados divididos:")
        self.log(f"  Pacientes de Treino: {len(train_df['Subject ID'].unique())}")
        self.log(f"  Pacientes de Validação: {len(val_df['Subject ID'].unique())}")
        self.log(f"  Pacientes de Teste: {len(test_df['Subject ID'].unique())}")

        # TODO: Preparar X e y para os modelos
        # X_train_features, y_train_class, y_train_age
        # X_val_features, y_val_class, y_val_age
        # X_test_features, y_test_class, y_test_age
        
        # Ex:
        # features_list = ['Area', 'Circularidade', 'Excentricidade', 'Extensao', 'Solidez', 'DiamEquiv']
        # features_list.extend(['Age', 'Educ', 'MMSE', 'eTIV', 'nWBV']) # Adiciona features do CSV
        
        # X_train = train_df[features_list]
        # y_train_class = train_df['Target_Class'].map({'NonDemented': 0, 'Demented': 1})
        # y_train_age = train_df['Age']
        
        # ... fazer o mesmo para val e test ...
        
        # ... escalar os dados (StandardScaler) ...

        return # train_data, val_data, test_data (retorne os dataframes ou arrays)

    def run_shallow_classifier(self):
        """ Classificador Raso: XGBoost [cite: 55, 94] """
        self.log("Iniciando Classificador Raso (XGBoost)...")
        messagebox.showinfo("TODO", "Carregar dados das features (X) e classes (y), treinar o XGBClassifier e mostrar métricas.")

        # TODO:
        # 1. Chamar self.prepare_data() para obter X_train, y_train, X_test, y_test
        #    (usando as features extraídas)
        # 2. Escalar os dados (StandardScaler)
        # 3. Treinar o modelo:
        #    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        #    model.fit(X_train_scaled, y_train)
        # 4. Fazer predições:
        #    y_pred = model.predict(X_test_scaled)
        # 5. Calcular e logar métricas[cite: 95]:
        #    acc = accuracy_score(y_test, y_pred)
        #    cm = confusion_matrix(y_test, y_pred)
        #    report = classification_report(y_test, y_pred, target_names=['NonDemented', 'Demented'])
        #    self.log(f"--- XGBoost (Shallow) ---")
        #    self.log(f"Acurácia: {acc:.4f}")
        #    self.log(f"Matriz de Confusão:\n{cm}")
        #    self.log(f"Relatório:\n{report}")
        pass

    def run_shallow_regressor(self):
        """ Regressor Raso: Regressão Linear [cite: 54, 99] """
        self.log("Iniciando Regressor Raso (Linear)...")
        messagebox.showinfo("TODO", "Carregar dados das features (X) e idade (y), treinar o LinearRegression e mostrar métricas.")

        # TODO:
        # 1. Chamar self.prepare_data() para obter X_train, y_train_age, X_test, y_test_age
        #    (usando as features extraídas + features do CSV)
        # 2. Escalar os dados (StandardScaler)
        # 3. Treinar o modelo:
        #    model = LinearRegression()
        #    model.fit(X_train_scaled, y_train_age)
        # 4. Fazer predições:
        #    y_pred_age = model.predict(X_test_scaled)
        # 5. Calcular e logar métricas:
        #    mse = mean_squared_error(y_test_age, y_pred_age)
        #    r2 = r2_score(y_test_age, y_pred_age)
        #    self.log(f"--- Regressão Linear (Shallow) ---")
        #    self.log(f"MSE (Erro Quadrático Médio): {mse:.2f}")
        #    self.log(f"R2 Score: {r2:.4f}")
        # 6. Analisar resultados [cite: 101, 102]
        #    (Ex: plotar y_test_age vs y_pred_age)
        pass

    def run_deep_classifier(self):
        """ Classificador Profundo: ResNet50 [cite: 57, 96] """
        self.log("Iniciando Classificador Profundo (ResNet50)...")
        messagebox.showinfo("TODO", "Implementar o pipeline de dados de imagem, criar o modelo ResNet50 e treiná-lo.")

        # TODO: Esta é a função MAIS COMPLEXA.
        # 1. Obter os dataframes de treino/val/teste de self.prepare_data()
        # 2. Criar um pipeline de dados (ex: tf.keras.preprocessing.image.ImageDataGenerator
        #    ou um tf.data.Dataset) que leia as *imagens* do disco.
        #    - Você precisará de uma coluna no seu DF que tenha o caminho da imagem.
        #    - As imagens precisam ser redimensionadas para o ResNet50 (ex: 224x224)
        #      e convertidas para 3 canais (RGB), já que o ResNet50 espera 3 canais.
        #      (img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB))
        
        # 3. Criar o modelo:
        #    img_shape = (224, 224, 3)
        #    base_model = ResNet50(weights='imagenet', include_top=False,
        #                          input_tensor=Input(shape=img_shape))
        #
        #    # Congelar o modelo base (para fine-tuning) [cite: 97]
        #    base_model.trainable = False
        #
        #    # Adicionar camadas no topo
        #    x = base_model.output
        #    x = GlobalAveragePooling2D()(x)
        #    x = Dense(128, activation='relu')(x)
        #    # Saída binária (Demented/NonDemented)
        #    predictions = Dense(1, activation='sigmoid')(x)
        #
        #    model = Model(inputs=base_model.input, outputs=predictions)
        
        # 4. Compilar o modelo:
        #    model.compile(optimizer=Adam(learning_rate=1e-4),
        #                  loss='binary_crossentropy',
        #                  metrics=['accuracy'])

        # 5. Treinar o modelo:
        #    history = model.fit(train_generator,
        #                        validation_data=val_generator,
        #                        epochs=10) # Ajustar épocas

        # 6. (Opcional) Fazer fine-tuning:
        #    base_model.trainable = True # Descongelar
        #    # ... re-compilar com LR baixo e treinar por mais algumas épocas ...
        
        # 7. Plotar gráficos de aprendizado [cite: 98]
        #    plt.plot(history.history['accuracy'], label='Train Acc')
        #    plt.plot(history.history['val_accuracy'], label='Val Acc')
        #    ...
        
        # 8. Avaliar no conjunto de teste [cite: 95]
        #    y_pred_probs = model.predict(test_generator)
        #    y_pred = (y_pred_probs > 0.5).astype(int)
        #    ... calcular métricas ...
        pass

    def run_deep_regressor(self):
        """ Regressor Profundo: ResNet50 [cite: 57, 100] """
        self.log("Iniciando Regressor Profundo (ResNet50)...")
        messagebox.showinfo("TODO", "Implementar o pipeline de dados de imagem, criar o modelo ResNet50 para regressão e treiná-lo.")

        # TODO:
        # 1. Similar ao 'run_deep_classifier', mas o pipeline de dados
        #    deve ter (imagem, idade) em vez de (imagem, classe).
        #
        # 2. A arquitetura do modelo muda na última camada:
        #    ...
        #    x = GlobalAveragePooling2D()(x)
        #    x = Dense(128, activation='relu')(x)
        #    # Saída de regressão (idade) - 1 neurônio, ativação linear
        #    predictions = Dense(1, activation='linear')(x)
        #
        #    model = Model(inputs=base_model.input, outputs=predictions)
        
        # 3. Compilação diferente:
        #    model.compile(optimizer=Adam(learning_rate=1e-4),
        #                  loss='mean_squared_error', # Loss de regressão
        #                  metrics=['mae']) # Métrica (Mean Absolute Error)

        # 4. Treinar e avaliar.
        
        # 5. Analisar resultados [cite: 101, 102]
        #    - As entradas (só imagem) são suficientes?
        #    - Exames posteriores resultam em idades maiores?
        #      (Pegar predições do test_df, ordenar por 'Subject ID' e 'Visit' e verificar)
        pass

################################################################################
# --- 6. EXECUÇÃO DA APLICAÇÃO ---
################################################################################

if __name__ == "__main__":
    # Garante que o TensorFlow não aloque toda a VRAM (se houver GPU)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs encontradas: {len(gpus)}")
    except Exception as e:
        print(f"Erro ao configurar GPUs: {e}")

    # Inicia a aplicação Tkinter
    main_root = tk.Tk()
    app = AlzheimerApp(main_root)
    main_root.mainloop()