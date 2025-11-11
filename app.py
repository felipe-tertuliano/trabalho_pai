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
        self.processed_image = None # Imagem PIL processada (segmentada)
        self.image_mask = None # Máscara (numpy) da segmentação
        self.features_df = None # DataFrame com características extraídas
        
        # --- Variáveis de Zoom ---
        self.zoom_level_original = 1.0
        self.zoom_level_processed = 1.0
        self.pan_start_x_original = 0
        self.pan_start_y_original = 0
        self.pan_start_x_processed = 0
        self.pan_start_y_processed = 0
        self.is_panning_original = False
        self.is_panning_processed = False
        
        # Configura a fonte padrão
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(size=self.current_font_size)
        
        # --- Layout Principal ---
        # Menu Superior
        self.create_menu()
        
        # Frame Principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Frame de Exibição de Imagem (à esquerda)
        image_frame = ttk.Frame(main_frame, relief=tk.RIDGE, padding="5")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.lbl_image_original = ttk.Label(image_frame, text="Imagem Original")
        self.lbl_image_original.pack(pady=5)
        self.canvas_original = tk.Canvas(image_frame, bg="gray", width=400, height=400)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        self.lbl_image_processed = ttk.Label(image_frame, text="Imagem Processada (Segmentada)")
        self.lbl_image_processed.pack(pady=5)
        self.canvas_processed = tk.Canvas(image_frame, bg="gray", width=400, height=400)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)
        
        # Frame de Controle e Log (à direita)
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.lbl_csv_status = ttk.Label(control_frame, text="CSV não carregado", foreground="red")
        self.lbl_csv_status.pack(pady=5, fill=tk.X)
        
        btn_load_csv = ttk.Button(control_frame, text="Carregar CSV", command=self.load_csv)
        btn_load_csv.pack(pady=5, fill=tk.X)
        
        btn_load_image = ttk.Button(control_frame, text="Carregar Imagem", command=self.load_image)
        btn_load_image.pack(pady=5, fill=tk.X)
        
        btn_reset_original = ttk.Button(control_frame, text="Reiniciar Zoom Orig.", command=lambda: self.reset_zoom("original"))
        btn_reset_original.pack(pady=5, fill=tk.X)
        
        btn_reset_processed = ttk.Button(control_frame, text="Reiniciar Zoom Proc.", command=lambda: self.reset_zoom("processed"))
        btn_reset_processed.pack(pady=5, fill=tk.X)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').pack(pady=10, fill=tk.X)
        
        # Botões de Processamento
        btn_segment = ttk.Button(control_frame, text="1. Segmentar Ventrículos", command=self.segment_ventricles)
        btn_segment.pack(pady=5, fill=tk.X)
        
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
        
        # Imagem processada
        self.canvas_processed.bind("<MouseWheel>", lambda e: self.zoom_image(e, "processed"))
        self.canvas_processed.bind("<Button-4>", lambda e: self.zoom_image(e, "processed"))
        self.canvas_processed.bind("<Button-5>", lambda e: self.zoom_image(e, "processed"))
        self.canvas_processed.bind("<ButtonPress-1>", lambda e: self.start_pan(e, "processed"))
        self.canvas_processed.bind("<B1-Motion>", lambda e: self.pan_image(e, "processed"))
        self.canvas_processed.bind("<ButtonRelease-1>", lambda e: self.stop_pan(e, "processed"))

    def zoom_image(self, event, canvas_type):
        """Controle de zoom"""
        if canvas_type == "original" and self.original_image is None:
            return
        if canvas_type == "processed" and self.processed_image is None:
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
            self.zoom_level_original = max(0.1, min(5.0, self.zoom_level_original))  # Limit zoom
            self.display_image_zoomed(
                self.original_image, self.canvas_original, 
                self.zoom_level_original,
                "original",
            )
        else:
            self.zoom_level_processed *= zoom_factor
            self.zoom_level_processed = max(0.1, min(5.0, self.zoom_level_processed))
            self.display_image_zoomed(
                self.processed_image, self.canvas_processed,
                self.zoom_level_processed,
                "processed",
            )

    def display_image_zoomed(self, pil_image, canvas, zoom_level, canvas_type):
        """Exibe imagem com zoom aplicado."""
        if pil_image is None:
            return
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2:
            canvas_width, canvas_height = 400, 400
        
        # Calcular novas dimensões com base no zoom
        new_width = int(pil_image.width * zoom_level)
        new_height = int(pil_image.height * zoom_level)
        
        # Redimensionar imagem
        img_resized = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Criar imagem
        if canvas_type == "original":
            self.tk_image_original = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_original
        else:
            self.tk_image_processed = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_processed
        
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
            self.canvas_original.config(cursor="fleur")
        else:
            self.is_panning_processed = True
            self.pan_start_x_processed = event.x
            self.pan_start_y_processed = event.y
            self.canvas_processed.config(cursor="fleur")

    def pan_image(self, event, canvas_type):
        """Exibe imagem por enquanto é arrastado."""
        if canvas_type == "original" and self.is_panning_original:
            # Calcula a distância
            dx = event.x - self.pan_start_x_original
            dy = event.y - self.pan_start_y_original
            
            # Move elementos do canvas
            self.canvas_original.move("all", dx, dy)
            
            # Atualiza posições iniciais
            self.pan_start_x_original = event.x
            self.pan_start_y_original = event.y
            
        elif canvas_type == "processed" and self.is_panning_processed:
            dx = event.x - self.pan_start_x_processed
            dy = event.y - self.pan_start_y_processed
            
            self.canvas_processed.move("all", dx, dy)
            self.pan_start_x_processed = event.x
            self.pan_start_y_processed = event.y

    def stop_pan(self, event, canvas_type):
        """Parar operação de exibição"""
        if canvas_type == "original":
            self.is_panning_original = False
            self.canvas_original.config(cursor="")
        else:
            self.is_panning_processed = False
            self.canvas_processed.config(cursor="")

    def reset_zoom(self, canvas_type):
        """Reinicia zoom e display para default"""
        if canvas_type == "original":
            self.zoom_level_original = 1.0
            if self.original_image is not None:
                self.display_image(self.original_image, self.canvas_original)
        else:
            self.zoom_level_processed = 1.0
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.canvas_processed)

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
            self.display_image(self.original_image, self.canvas_original)
            self.processed_image = None
            self.image_mask = None
            self.canvas_processed.delete("all") # Limpa canvas processado

        except Exception as e:
            messagebox.showerror("Erro ao Carregar Imagem", f"Erro: {e}")
            self.log(f"Falha ao carregar imagem: {e}")

    def display_image(self, pil_image, canvas):
        """ Exibe uma imagem PIL em um canvas Tkinter, com zoom/redimensionamento. [cite: 76]"""
        # Reinicia zoom ao exibir uma nova imagem
        if canvas == self.canvas_original:
            self.zoom_level_original = 1.0
        else:
            self.zoom_level_processed = 1.0
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2: # Canvas não está pronto
            canvas_width, canvas_height = 400, 400 # Valor padrão

        # Redimensiona a imagem para caber no canvas mantendo a proporção
        img_resized = ImageOps.contain(pil_image, (canvas_width, canvas_height))

        # Criar PhotoImage
        if canvas == self.canvas_original:
            self.tk_image_original = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_original
        else:
            self.tk_image_processed = ImageTk.PhotoImage(img_resized)
            img_tk = self.tk_image_processed
        
        canvas.delete("all")
        canvas.create_image(
            canvas_width / 2, canvas_height / 2,
            anchor=tk.CENTER, image=img_tk
        )

    # --- 4. FUNÇÕES DE PROCESSAMENTO DE IMAGEM ---

    def segment_ventricles(self):
        """ Executa a segmentação dos ventrículos laterais. [cite: 78] """
        if self.original_image is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        self.log("Iniciando segmentação dos ventrículos...")

        # Converte a imagem PIL para formato OpenCV (Numpy array)
        img_cv = np.array(self.original_image)
        
        # --- TODO: LÓGICA DE SEGMENTAÇÃO ---
        # Esta é a parte MAIS CRÍTICA do trabalho.
        # O método abaixo é um PLACEHOLDER SIMPLES (Thresholding) e
        # provavelmente NÃO dará bons resultados.
        #
        # O PDF diz "usando qualquer método"[cite: 78].
        # Vocês devem pesquisar e implementar um método robusto, como:
        # 1. Thresholding de Otsu + Operações Morfológicas (Abertura/Fechamento)
        # 2. Region Growing (Crescimento de Regiões)
        # 3. Active Contours (Contornos Ativos / "Snakes")
        # 4. Watershed (Divisor de Águas)
        
        # Exemplo de Placeholder (Thresholding Simples):
        # Ventrículos são escuros (fluido) em T1
        _, mask = cv2.threshold(img_cv, 40, 255, cv2.THRESH_BINARY)
        
        # Aplicar operações morfológicas para limpar o ruído
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Salva a máscara
        self.image_mask = mask # Este é o resultado binário (numpy)
        
        # --- Fim do TODO de Segmentação ---

        # Cria uma imagem de visualização com o contorno (como na Fig. pag 4)
        img_with_contours = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(self.image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contornos pequenos (ruído)
        min_area = 100
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        cv2.drawContours(img_with_contours, large_contours, -1, (0, 255, 0), 2) # Desenha em verde
        
        # Converte de volta para PIL e exibe
        self.processed_image = Image.fromarray(img_with_contours)
        self.display_image(self.processed_image, self.canvas_processed)
        
        self.log(f"Segmentação concluída. {len(large_contours)} contornos encontrados.")
        
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