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
        self.region_growing_threshold = 50  # FIXO: 50
        self.use_histogram_equalization = True  # FIXO: CLAHE sempre ativado
        self.use_otsu_for_segmentation = False  # Usar Otsu (binarizada) ou CLAHE (escala de cinza)
        self.multi_seed_mode = False  # Modo de múltiplos seeds manual
        self.accumulated_mask = None  # Máscara acumulada de múltiplos cliques
        self.multi_seed_points = []  # Pontos coletados no modo multi-seed
        
        # --- Variáveis de Pós-processamento Morfológico (FIXAS) ---
        self.morphology_kernel_size = 15  # FIXO: 15x15
        self.apply_closing = True  # FIXO: Fechamento ativado
        self.apply_fill_holes = True  # FIXO: Preencher buracos ativado
        self.apply_opening = True  # FIXO: Abertura ativado (remover ruído)
        self.apply_smooth_contours = True  # FIXO: Suavizar contornos ativado
        
        # --- Variáveis de medição de coordenadas ---
        self.show_coordinates = True  # Mostrar coordenadas do mouse
        self.current_mouse_x = 0
        self.current_mouse_y = 0
        
        # --- Seed points FIXOS para segmentação automática ---
        self.auto_seed_points = [
            (164, 91),   # Ponto 1 - Ventrículo
            (171, 114),  # Ponto 2 - Ventrículo
        ]
        
        # Configura a fonte padrão
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(size=self.current_font_size)
        
        # --- Layout Principal ---
        # Menu Superior
        self.create_menu()
        
        # Frame Principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Frame Superior: Grid de Imagens (2 colunas: imagens + controles)
        images_and_controls_frame = ttk.Frame(main_frame)
        images_and_controls_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Frame de Imagens (à esquerda)
        images_container = ttk.Frame(images_and_controls_frame)
        images_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Linha 1: Original + Preprocessed
        row1_frame = ttk.Frame(images_container)
        row1_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        
        # Canvas Original
        original_frame = ttk.Frame(row1_frame, relief=tk.RIDGE, padding="2")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.lbl_image_original = ttk.Label(original_frame, text="📷 Imagem Original", font=("Arial", 10, "bold"))
        self.lbl_image_original.pack(pady=2)
        self.canvas_original = tk.Canvas(original_frame, bg="gray", width=300, height=300)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Canvas Preprocessed
        preprocessed_frame = ttk.Frame(row1_frame, relief=tk.RIDGE, padding="2")
        preprocessed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.lbl_image_preprocessed = ttk.Label(preprocessed_frame, text="🔧 Pré-processada (Filtros)", font=("Arial", 10, "bold"))
        self.lbl_image_preprocessed.pack(pady=2)
        self.canvas_preprocessed = tk.Canvas(preprocessed_frame, bg="gray", width=300, height=300)
        self.canvas_preprocessed.pack(fill=tk.BOTH, expand=True)
        
        # Linha 2: Segmented
        row2_frame = ttk.Frame(images_container)
        row2_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=2)
        
        # Canvas Segmented
        segmented_frame = ttk.Frame(row2_frame, relief=tk.RIDGE, padding="2")
        segmented_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self.lbl_image_segmented = ttk.Label(segmented_frame, text="✂️ Segmentada (Contorno Amarelo)", font=("Arial", 10, "bold"))
        self.lbl_image_segmented.pack(pady=2)
        self.canvas_segmented = tk.Canvas(segmented_frame, bg="gray", width=600, height=300)
        self.canvas_segmented.pack(fill=tk.BOTH, expand=True)
        
        # Frame de Controle e Log (à direita)
        control_frame = ttk.Frame(images_and_controls_frame, width=280)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.lbl_csv_status = ttk.Label(control_frame, text="CSV não carregado", foreground="red")
        self.lbl_csv_status.pack(pady=5, fill=tk.X)
        
        btn_load_csv = ttk.Button(control_frame, text="Carregar CSV", command=self.load_csv)
        btn_load_csv.pack(pady=5, fill=tk.X)
        
        btn_load_image = ttk.Button(control_frame, text="Carregar Imagem", command=self.load_image)
        btn_load_image.pack(pady=5, fill=tk.X)
        
        # Frame de botões de zoom em grid
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(pady=5, fill=tk.X)
        btn_reset_original = ttk.Button(zoom_frame, text="↻ Orig", command=lambda: self.reset_zoom("original"))
        btn_reset_original.pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        btn_reset_preprocessed = ttk.Button(zoom_frame, text="↻ Prep", command=lambda: self.reset_zoom("preprocessed"))
        btn_reset_preprocessed.pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        btn_reset_segmented = ttk.Button(zoom_frame, text="↻ Seg", command=lambda: self.reset_zoom("segmented"))
        btn_reset_segmented.pack(side=tk.LEFT, padx=1, expand=True, fill=tk.X)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Seção de Pré-processamento (FIXO: Otsu + CLAHE)
        ttk.Label(control_frame, text="🔧 Pré-processamento (FIXO):", font=("Arial", 9, "bold")).pack(pady=(5,0))
        
        btn_apply_otsu = ttk.Button(control_frame, text="Aplicar Otsu + CLAHE", command=lambda: self.apply_filter("otsu"))
        btn_apply_otsu.pack(pady=5, fill=tk.X)
        
        self.lbl_current_filter = ttk.Label(control_frame, text="Pré-proc: Não aplicado", foreground="gray")
        self.lbl_current_filter.pack(pady=2)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Seção de Segmentação
        ttk.Label(control_frame, text="✂️ Segmentação:", font=("Arial", 9, "bold")).pack(pady=(5,0))
        
        # Escolha de imagem para segmentação
        ttk.Label(control_frame, text="Imagem para Region Growing:", foreground="blue").pack(pady=(5,0))
        self.segmentation_mode = tk.StringVar(value="clahe")
        
        rb_frame = ttk.Frame(control_frame)
        rb_frame.pack(pady=2, fill=tk.X)
        
        rb_clahe = ttk.Radiobutton(rb_frame, text="CLAHE (Escala Cinza)", 
                                    variable=self.segmentation_mode, value="clahe")
        rb_clahe.pack(anchor=tk.W, padx=20)
        
        rb_otsu = ttk.Radiobutton(rb_frame, text="Otsu (Binarizada)", 
                                   variable=self.segmentation_mode, value="otsu")
        rb_otsu.pack(anchor=tk.W, padx=20)
        
        ttk.Label(control_frame, text="Parâmetros FIXOS:", foreground="blue").pack(pady=(5,0))
        ttk.Label(control_frame, text="• Threshold: 50", foreground="gray").pack(anchor=tk.W, padx=20)
        ttk.Label(control_frame, text="• Kernel: 15x15", foreground="gray").pack(anchor=tk.W, padx=20)
        ttk.Label(control_frame, text="• Morfologia: Completa", foreground="gray").pack(anchor=tk.W, padx=20)
        
        # Segmentação Automática
        ttk.Label(control_frame, text="1. Automática (Seeds Fixos):", foreground="blue").pack(pady=(5,0))
        ttk.Label(control_frame, text="• Seed 1: (164, 91)", foreground="gray").pack(anchor=tk.W, padx=20)
        ttk.Label(control_frame, text="• Seed 2: (171, 114)", foreground="gray").pack(anchor=tk.W, padx=20)
        
        btn_segment_auto = ttk.Button(control_frame, text="▶ Segmentação Automática", command=self.segment_ventricles)
        btn_segment_auto.pack(pady=5, fill=tk.X)
        
        # Segmentação Manual (Multi-Seed)
        ttk.Label(control_frame, text="2. Manual (Multi-Seed):", foreground="blue").pack(pady=(5,0))
        
        multi_seed_buttons = ttk.Frame(control_frame)
        multi_seed_buttons.pack(pady=5, fill=tk.X)
        
        btn_multi_start = ttk.Button(multi_seed_buttons, text="Iniciar Multi-Seed", command=self.start_multi_seed)
        btn_multi_start.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        btn_multi_finish = ttk.Button(multi_seed_buttons, text="Finalizar", command=self.finish_multi_seed)
        btn_multi_finish.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        self.lbl_multi_seed = ttk.Label(control_frame, text="", foreground="purple")
        self.lbl_multi_seed.pack(pady=2)
        
        btn_export_multi_seeds = ttk.Button(control_frame, text="💾 Salvar Pontos Multi-Seed", 
                                            command=self.export_multi_seed_points)
        btn_export_multi_seeds.pack(pady=5, fill=tk.X)
        
        self.lbl_segment_status = ttk.Label(control_frame, text="", foreground="green")
        self.lbl_segment_status.pack(pady=2)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Seção de Coordenadas
        ttk.Label(control_frame, text="📍 Coordenadas do Mouse:", font=("Arial", 9, "bold")).pack(pady=(5,0))
        
        self.lbl_mouse_coords = ttk.Label(control_frame, text="X: -- | Y: --", 
                                          font=("Courier", 11), foreground="blue")
        self.lbl_mouse_coords.pack(pady=5)
        
        ttk.Label(control_frame, text="Clique para registrar ponto:", foreground="gray").pack()
        self.lbl_clicked_coords = ttk.Label(control_frame, text="Nenhum ponto registrado", 
                                            font=("Courier", 9), foreground="green")
        self.lbl_clicked_coords.pack(pady=2)
        
        points_buttons_frame = ttk.Frame(control_frame)
        points_buttons_frame.pack(pady=5, fill=tk.X)
        
        btn_clear_points = ttk.Button(points_buttons_frame, text="Limpar", command=self.clear_registered_points)
        btn_clear_points.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        btn_export_points = ttk.Button(points_buttons_frame, text="Exportar", command=self.export_registered_points)
        btn_export_points.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # Lista de pontos registrados
        self.registered_points = []
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Seção de Processamento em Lote
        ttk.Label(control_frame, text="📁 Processamento em Lote:", font=("Arial", 9, "bold")).pack(pady=(5,0))
        
        btn_batch = ttk.Button(control_frame, text="🔄 Segmentar Pasta Inteira (.nii)", 
                               command=self.batch_segment_folder)
        btn_batch.pack(pady=5, fill=tk.X)
        
        self.lbl_batch_status = ttk.Label(control_frame, text="", foreground="blue")
        self.lbl_batch_status.pack(pady=2)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        btn_extract = ttk.Button(control_frame, text="2. Extrair Características", command=self.extract_features)
        btn_extract.pack(pady=5, fill=tk.X)
        
        btn_scatterplot = ttk.Button(control_frame, text="3. Gerar Gráf. Dispersão", command=self.show_scatterplot)
        btn_scatterplot.pack(pady=5, fill=tk.X)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Botões de Modelos
        btn_class_shallow = ttk.Button(control_frame, text="Classif. Raso (XGBoost)", command=self.run_shallow_classifier)
        btn_class_shallow.pack(pady=5, fill=tk.X)
        
        btn_regr_shallow = ttk.Button(control_frame, text="Regressão Rasa (Linear)", command=self.run_shallow_regressor)
        btn_regr_shallow.pack(pady=5, fill=tk.X)
        
        btn_class_deep = ttk.Button(control_frame, text="Classif. Profunda (ResNet50)", command=self.run_deep_classifier)
        btn_class_deep.pack(pady=5, fill=tk.X)
        
        btn_regr_deep = ttk.Button(control_frame, text="Regressão Profunda (ResNet50)", command=self.run_deep_regressor)
        btn_regr_deep.pack(pady=5, fill=tk.X)
        
        # Log
        self.log_text = tk.Text(control_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Configura os bindings do sistema depois da criação da interface
        self.root.after(100, self.setup_bindings)

    def log(self, message):
        """ Adiciona uma mensagem ao log na GUI. """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

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
            self.lbl_mouse_coords.config(text=f"X: {img_x:3d} | Y: {img_y:3d}")
        else:
            self.lbl_mouse_coords.config(text="X: -- | Y: --")

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
            self.lbl_mouse_coords.config(text=f"X: {img_x:3d} | Y: {img_y:3d} [PRÉ-PROC]")
        else:
            self.lbl_mouse_coords.config(text="X: -- | Y: --")

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
            self.preprocessed_image = None
            self.segmented_image = None
            self.image_mask = None
            self.canvas_preprocessed.delete("all") # Limpa canvas preprocessed
            self.canvas_segmented.delete("all") # Limpa canvas segmented
            self.current_filter = "none"
            self.lbl_current_filter.config(text="Filtro: Nenhum")

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
        
        # Processa cada arquivo
        success_count = 0
        error_count = 0
        
        for idx, filename in enumerate(nii_files, 1):
            try:
                self.log(f"\n[{idx}/{len(nii_files)}] Processando: {filename}")
                self.lbl_batch_status.config(
                    text=f"Processando {idx}/{len(nii_files)}: {filename[:30]}...",
                    foreground="blue"
                )
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
                    
                    mask = self.region_growing(img_for_segmentation, (seed_x, seed_y), 
                                              threshold=self.region_growing_threshold)
                    
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
                
                # Cria imagem segmentada com contorno amarelo
                img_with_contour = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
                cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
                
                # Salva os resultados
                base_name = os.path.splitext(filename)[0]
                if filename.endswith('.nii.gz'):
                    base_name = base_name.replace('.nii', '')
                
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
        
        # Relatório final
        self.log("-"*70)
        self.log(f"✅ PROCESSAMENTO EM LOTE CONCLUÍDO!")
        self.log(f"   • Total processado: {len(nii_files)}")
        self.log(f"   • Sucessos: {success_count}")
        self.log(f"   • Erros: {error_count}")
        self.log(f"   • Pasta de saída: {output_folder}")
        self.log("="*70 + "\n")
        
        self.lbl_batch_status.config(
            text=f"✓ Concluído: {success_count}/{len(nii_files)} arquivos",
            foreground="green"
        )
        
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
        """Captura clique na janela pré-processada e usa a imagem Otsu+CLAHE diretamente."""
        if self.preprocessed_image is None:
            messagebox.showwarning("Aviso", "Aplique 'Otsu + CLAHE' primeiro!")
            self.log("⚠ Nenhuma imagem pré-processada disponível. Aplique um filtro primeiro.")
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
            self.log("Clique fora da imagem pré-processada.")
            return

        self.log(f"🖼️ Clique na IMAGEM PRÉ-PROCESSADA detectado!")
        self.log(f"📍 Coordenadas: ({img_x}, {img_y})")
        self.log(f"🎯 Usando imagem Otsu+CLAHE diretamente para segmentação")
        self.log(f"⚙️ Configuração: Threshold={self.region_growing_threshold}, Kernel={self.morphology_kernel_size}x{self.morphology_kernel_size}")
        self.log(f"🔬 Morfologia: Abertura + Fechamento + Preenchimento + Suavização")
        
        # Converte a imagem pré-processada para numpy
        img_preprocessed_np = np.array(self.preprocessed_image)
        
        # Pega a imagem original para visualização final
        img_original_np = np.array(self.original_image)
        
        # Usa a imagem PRÉ-PROCESSADA diretamente para Region Growing
        if self.multi_seed_mode:
            self.multi_seed_points.append((img_x, img_y))
            self.log(f"🎯 Multi-Seed {len(self.multi_seed_points)}: ({img_x}, {img_y})")
            self.lbl_multi_seed.config(
                text=f"🎯 Multi-Seed: {len(self.multi_seed_points)} ponto(s) - PRÉ-PROC",
                foreground="purple"
            )
            
            # Segmenta usando a imagem pré-processada
            mask = self.region_growing(img_preprocessed_np, (img_x, img_y), 
                                      threshold=self.region_growing_threshold)
            
            # Acumula máscaras
            if self.accumulated_mask is None:
                self.accumulated_mask = mask.copy()
            else:
                self.accumulated_mask = cv2.bitwise_or(self.accumulated_mask, mask)
            
            final_mask = self.apply_morphological_postprocessing(self.accumulated_mask)
            self.image_mask = final_mask
            
            # Visualiza na imagem ORIGINAL
            img_with_contour = cv2.cvtColor(img_original_np, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
            
            self.segmented_image = Image.fromarray(img_with_contour)
            self.display_image(self.segmented_image, self.canvas_segmented, "segmented")
            
            num_pixels = np.sum(final_mask == 255)
            self.log(f"   ✓ Acumulado: {num_pixels} pixels totais\n")
            
        else:
            # Modo normal: segmentação com um único seed usando imagem pré-processada
            self.log(f"Aplicando Region Growing na imagem PRÉ-PROCESSADA...")
            mask = self.region_growing(img_preprocessed_np, (img_x, img_y), 
                                      threshold=self.region_growing_threshold)

            # Aplica pós-processamento morfológico
            self.log("Aplicando pós-processamento morfológico...")
            final_mask = self.apply_morphological_postprocessing(mask)
            self.image_mask = final_mask

            # Visualiza na imagem ORIGINAL com contorno amarelo
            img_with_contour = cv2.cvtColor(img_original_np, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
            
            self.segmented_image = Image.fromarray(img_with_contour)
            self.display_image(self.segmented_image, self.canvas_segmented, "segmented")

            num_pixels = np.sum(final_mask == 255)
            self.log(f"✓ Segmentação concluída. {len(large_contours)} região(ões) | {num_pixels} pixels")

    def on_click_seed(self, event):
        """Captura clique no canvas para selecionar seed point do region growing."""
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
            self.log("Clique fora da imagem.")
            return

        # Registra o ponto clicado (para rastreamento)
        self.registered_points.append((img_x, img_y))
        self.lbl_clicked_coords.config(text=f"Último: ({img_x}, {img_y}) | Total: {len(self.registered_points)}")
        
        self.log(f"📍 Ponto {len(self.registered_points)} registrado: X={img_x}, Y={img_y}")
        
        # Se está no modo Multi-Seed, apenas acumula pontos e máscaras
        if self.multi_seed_mode:
            self.multi_seed_points.append((img_x, img_y))
            self.log(f"🎯 Multi-Seed {len(self.multi_seed_points)}: ({img_x}, {img_y})")
            self.lbl_multi_seed.config(
                text=f"🎯 Multi-Seed: {len(self.multi_seed_points)} ponto(s) coletado(s)",
                foreground="purple"
            )
            
            # Segmenta este ponto e acumula
            img_np = np.array(self.original_image)
            img_for_segmentation = self.prepare_image_for_segmentation(img_np)
            
            mask = self.region_growing(img_for_segmentation, (img_x, img_y), 
                                      threshold=self.region_growing_threshold)
            
            # Acumula máscaras
            if self.accumulated_mask is None:
                self.accumulated_mask = mask.copy()
            else:
                self.accumulated_mask = cv2.bitwise_or(self.accumulated_mask, mask)
            
            # Aplica pós-processamento e visualiza
            final_mask = self.apply_morphological_postprocessing(self.accumulated_mask)
            self.image_mask = final_mask
            
            # Visualiza
            img_with_contour = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
            cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
            
            self.segmented_image = Image.fromarray(img_with_contour)
            self.display_image(self.segmented_image, self.canvas_segmented, "segmented")
            
            num_pixels = np.sum(final_mask == 255)
            self.log(f"   ✓ Acumulado: {num_pixels} pixels totais\n")
            
        else:
            # Modo normal: segmentação com um único seed
            self.log(f"🎯 Seed selecionado: ({img_x}, {img_y})")
            
            # Converte imagem PIL → numpy
            img_np = np.array(self.original_image)
            
            # Prepara imagem para segmentação (CLAHE ou Otsu)
            img_for_segmentation = self.prepare_image_for_segmentation(img_np)

            # Aplica region growing
            self.log(f"Aplicando Region Growing (threshold={self.region_growing_threshold})...")
            mask = self.region_growing(img_for_segmentation, (img_x, img_y), threshold=self.region_growing_threshold)

            # Aplica pós-processamento morfológico
            self.log("Aplicando pós-processamento morfológico...")
            final_mask = self.apply_morphological_postprocessing(mask)
            
            self.image_mask = final_mask

            # Cria visualização: imagem original em RGB com contorno AMARELO
            img_with_contour = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            
            # Encontra contornos na máscara final
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtra contornos pequenos (ruído)
            min_area = 50
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Desenha contornos em AMARELO (BGR: 0, 255, 255)
            cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
            
            # Converte para PIL e exibe no canvas segmented
            self.segmented_image = Image.fromarray(img_with_contour)
            self.display_image(self.segmented_image, self.canvas_segmented, "segmented")

            num_pixels = np.sum(final_mask == 255)
            self.log(f"✓ Region growing concluído. {len(large_contours)} região(ões) | {num_pixels} pixels segmentados")

    def prepare_image_for_segmentation(self, img_np):
        """
        Prepara a imagem para segmentação baseado na escolha do usuário.
        
        Args:
            img_np: numpy array 2D (grayscale)
            
        Returns:
            img_processed: imagem processada (CLAHE ou Otsu)
        """
        mode = self.segmentation_mode.get()
        
        if mode == "otsu":
            # Aplica CLAHE primeiro
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_img = clahe.apply(img_np)
            
            # Depois aplica Otsu
            _, img_processed = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.log("   → Usando imagem OTSU (binarizada) para segmentação")
            
        else:  # "clahe"
            # Apenas CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_processed = clahe.apply(img_np)
            self.log("   → Usando imagem CLAHE (escala de cinza) para segmentação")
        
        return img_processed

    def region_growing(self, image, seed, threshold=10):
        """
        Algoritmo de Region Growing para segmentação.
        
        Args:
            image: numpy array 2D (grayscale)
            seed: (x, y) pixel inicial clicado
            threshold: variação de intensidade permitida em relação ao seed
            
        Returns:
            mask: numpy array 2D binário (0=fundo, 255=região)
        """
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        seed_x, seed_y = seed
        seed_intensity = int(image[seed_y, seed_x])

        queue = [(seed_x, seed_y)]
        mask[seed_y, seed_x] = 255

        # 8-connected neighbors
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
        
        # 2. Fechamento (Closing) - Fecha buracos pequenos
        if self.apply_closing:
            processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            self.log(f"   → Fechamento aplicado (kernel {kernel_size}x{kernel_size})")
        
        # 3. Preenchimento de buracos (Fill Holes)
        if self.apply_fill_holes:
            # Encontra contornos
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Preenche cada contorno
            filled_mask = np.zeros_like(processed_mask)
            for cnt in contours:
                cv2.drawContours(filled_mask, [cnt], 0, 255, -1)  # -1 preenche o interior
            
            processed_mask = filled_mask
            self.log(f"   → Buracos preenchidos ({len(contours)} regiões)")
        
        # 4. Suavização de contornos (Contour Smoothing)
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

        # Converte a imagem PIL para formato OpenCV (Numpy array)
        img_np = np.array(self.original_image)
        
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
                                      threshold=self.region_growing_threshold)
            
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
        
        self.image_mask = final_mask

        # Cria visualização: imagem original em RGB com contorno AMARELO
        img_with_contour = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        
        # Encontra contornos na máscara final
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos pequenos (ruído)
        min_area = 50
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Desenha contornos em AMARELO (BGR: 0, 255, 255)
        cv2.drawContours(img_with_contour, large_contours, -1, (0, 255, 255), 2)
        
        # Converte para PIL e exibe no canvas segmented
        self.segmented_image = Image.fromarray(img_with_contour)
        self.display_image(self.segmented_image, self.canvas_segmented, "segmented")

        num_pixels_final = np.sum(final_mask == 255)
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