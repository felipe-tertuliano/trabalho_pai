"""
Parte 10: Classificadores para Diagnóstico de Alzheimer

Implementa:
1. Classificador Raso (XGBoost) usando descritores manuais
2. Classificador Profundo (ResNet50) usando imagens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path

# Tentar importar seaborn, usar matplotlib se não disponível
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Machine Learning
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Deep Learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Imagens
from PIL import Image
import cv2

warnings.filterwarnings('ignore')

# Configurar TensorFlow para não usar toda a GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass


def find_image_path_column(df):
    """Encontra automaticamente a coluna que contém o caminho das imagens"""
    possible_cols = ["filepath", "path", "image_path", "ImagePath", "filename", "FileName", "Image Path"]
    
    for col in possible_cols:
        if col in df.columns:
            return col
    
    return None


def get_image_path(row, df, base_image_dir="images"):
    """
    Determina o caminho da imagem para uma linha do DataFrame.
    Tenta várias estratégias automaticamente.
    """
    # Estratégia 1: Coluna explícita de caminho
    path_col = find_image_path_column(df)
    if path_col:
        path = row[path_col]
        if pd.notna(path) and path:
            if os.path.exists(path):
                return path
            # Tentar com base_image_dir
            full_path = os.path.join(base_image_dir, os.path.basename(path))
            if os.path.exists(full_path):
                return full_path
    
    # Estratégia 2: Usar MRI ID ou Image Data ID
    id_cols = ["MRI ID", "MRI_ID", "Image Data ID", "ImageDataID", "ImageID"]
    for col in id_cols:
        if col in df.columns:
            img_id = row[col]
            if pd.notna(img_id):
                # Tentar diferentes extensões
                for ext in ['.png', '.jpg', '.jpeg', '.nii', '.nii.gz']:
                    path = os.path.join(base_image_dir, f"{img_id}{ext}")
                    if os.path.exists(path):
                        return path
                    # Tentar sem extensão no nome
                    path = os.path.join(base_image_dir, str(img_id))
                    if os.path.exists(path):
                        return path
    
    return None


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Carrega e pré-processa uma imagem para o ResNet50.
    Converte para RGB e redimensiona.
    """
    try:
        if image_path.endswith(('.nii', '.nii.gz')):
            # Carregar NIfTI
            import nibabel as nib
            nii_img = nib.load(image_path)
            img_data = nii_img.get_fdata()
            
            # Se for 3D, pegar slice coronal central
            if len(img_data.shape) == 3:
                slice_idx = img_data.shape[1] // 2
                img_data = img_data[:, slice_idx, :]
            
            # Normalizar para 0-255
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data) + 1e-8)
            img_data = (img_data * 255).astype(np.uint8)
            img = Image.fromarray(img_data).convert('RGB')
        else:
            # Carregar imagem normal
            img = Image.open(image_path).convert('RGB')
        
        # Redimensionar
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Converter para array e normalizar para ImageNet
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0
        
        # Normalização ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        return img_array
    except Exception as e:
        print(f"Erro ao carregar imagem {image_path}: {e}")
        return None


def train_xgboost_classifier():
    """Treina e avalia o classificador XGBoost"""
    print("\n" + "="*70)
    print("CLASSIFICADOR RASO (XGBOOST)")
    print("="*70)
    
    # Ler CSVs
    print("\n1. Carregando dados...")
    train_df = pd.read_csv("train_split.csv", sep=";", decimal=".")
    val_df = pd.read_csv("val_split.csv", sep=";", decimal=".")
    test_df = pd.read_csv("test_split.csv", sep=";", decimal=".")
    
    print(f"   Treino: {len(train_df)} exames")
    print(f"   Validação: {len(val_df)} exames")
    print(f"   Teste: {len(test_df)} exames")
    
    # Features base
    base_features = ["area", "perimeter", "eccentricity", "extent", "solidity"]
    
    # Verificar quais features existem
    available_features = [f for f in base_features if f in train_df.columns]
    missing_features = [f for f in base_features if f not in train_df.columns]
    
    if missing_features:
        print(f"   Aviso: Features não encontradas: {missing_features}")
    print(f"   Features usadas: {available_features}")
    
    if not available_features:
        print("   ERRO: Nenhuma feature disponível!")
        return
    
    # Preparar dados
    X_train = train_df[available_features].copy()
    X_val = val_df[available_features].copy()
    X_test = test_df[available_features].copy()
    
    # Converter para numérico
    for col in available_features:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_val[col] = pd.to_numeric(X_val[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Preencher NaN com média
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # Target: ClassBinary -> 0/1
    y_train = (train_df['ClassBinary'] == 'Demented').astype(int)
    y_val = (val_df['ClassBinary'] == 'Demented').astype(int)
    y_test = (test_df['ClassBinary'] == 'Demented').astype(int)
    
    print(f"\n2. Distribuição de classes:")
    print(f"   Treino - NonDemented: {sum(y_train == 0)}, Demented: {sum(y_train == 1)}")
    print(f"   Validação - NonDemented: {sum(y_val == 0)}, Demented: {sum(y_val == 1)}")
    print(f"   Teste - NonDemented: {sum(y_test == 0)}, Demented: {sum(y_test == 1)}")
    
    # Normalizar dados (importante para melhor performance)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Converter de volta para DataFrame para manter nomes das colunas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=available_features, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)
    
    # Calcular scale_pos_weight para balanceamento
    pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1.0
    print(f"\n3. scale_pos_weight calculado: {pos_weight:.2f}")
    print("   Dados normalizados com StandardScaler")
    
    # Random Search para otimizar hiperparâmetros
    # IMPORTANTE: Usar apenas treino, não combinar com validação!
    print("\n4. Executando Random Search para otimizar hiperparâmetros...")
    print("   (Isso pode levar alguns minutos - testando 100 combinações com 5-fold CV)...")
    
    # Definir espaço de busca de parâmetros MELHORADO
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2.0]
    }
    
    # Modelo base
    base_model = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Random Search com validação cruzada
    # Usar 'roc_auc' ou 'f1' em vez de 'accuracy' para melhor generalização
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=100,  # Aumentado para 100 iterações
        scoring='roc_auc',  # ROC-AUC é melhor para problemas desbalanceados
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )
    
    # Usar APENAS treino no Random Search
    random_search.fit(X_train_scaled, y_train)
    
    # Melhores parâmetros encontrados
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n   ✓ Random Search concluído!")
    print(f"   Melhor ROC-AUC (CV): {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"   Melhores parâmetros encontrados:")
    for param, value in sorted(best_params.items()):
        print(f"     • {param}: {value}")
    
    # Treinar XGBoost com os melhores parâmetros e EARLY STOPPING
    print("\n5. Treinando XGBoost com os melhores parâmetros e early stopping...")
    
    # Ajustar n_estimators para permitir early stopping
    n_estimators_final = max(best_params.get('n_estimators', 300), 500)  # Mínimo 500 para early stopping
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators_final,
        learning_rate=best_params.get('learning_rate', 0.05),
        max_depth=best_params.get('max_depth', 4),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        min_child_weight=best_params.get('min_child_weight', 1),
        gamma=best_params.get('gamma', 0),
        reg_alpha=best_params.get('reg_alpha', 0),
        reg_lambda=best_params.get('reg_lambda', 1),
        eval_metric='logloss',
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    # Treinar com histórico e early stopping
    # Nota: XGBoost 3.x usa callbacks para early stopping
    eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
    
    try:
        # Tentar usar callback (XGBoost 3.x)
        from xgboost.callback import EarlyStopping
        early_stop = EarlyStopping(
            rounds=50,  # Para após 50 rounds sem melhoria
            save_best=True,
            maximize=False,  # Minimizar logloss
            min_delta=0.001
        )
        model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            callbacks=[early_stop],
            verbose=False
        )
        # Usar o melhor número de estimadores encontrado pelo early stopping
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration is None:
            best_iteration = n_estimators_final
    except (ImportError, AttributeError, TypeError):
        # Fallback para versões antigas ou se callback não funcionar
        # Usar n_estimators menor para evitar overfitting
        n_estimators_safe = min(n_estimators_final, 300)
        model = xgb.XGBClassifier(
            n_estimators=n_estimators_safe,
            learning_rate=best_params.get('learning_rate', 0.05),
            max_depth=best_params.get('max_depth', 4),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            min_child_weight=best_params.get('min_child_weight', 1),
            gamma=best_params.get('gamma', 0),
            reg_alpha=best_params.get('reg_alpha', 0),
            reg_lambda=best_params.get('reg_lambda', 1),
            eval_metric='logloss',
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        best_iteration = n_estimators_safe
    
    print(f"   Melhor iteração encontrada: {best_iteration}")
    
    # Extrair histórico de logloss
    results = model.evals_result()
    train_logloss = results['validation_0']['logloss']
    val_logloss = results['validation_1']['logloss']
    
    # Calcular acurácia em alguns pontos selecionados durante o treinamento
    # Para eficiência, vamos fazer isso de forma incremental
    print("   Calculando curva de aprendizado...")
    train_acc = []
    val_acc = []
    
    # Usar n_estimators do melhor modelo (ou do early stopping)
    n_estimators_best = best_iteration
    learning_rate_best = best_params.get('learning_rate', 0.05)
    max_depth_best = best_params.get('max_depth', 4)
    subsample_best = best_params.get('subsample', 0.8)
    colsample_best = best_params.get('colsample_bytree', 0.8)
    min_child_weight_best = best_params.get('min_child_weight', 1)
    gamma_best = best_params.get('gamma', 0)
    reg_alpha_best = best_params.get('reg_alpha', 0)
    reg_lambda_best = best_params.get('reg_lambda', 1)
    
    # Usar menos pontos para ser mais rápido
    n_points = min(10, n_estimators_best // 50)
    step = max(1, n_estimators_best // n_points)
    check_points = list(range(step, n_estimators_best + 1, step))
    if check_points[-1] != n_estimators_best:
        check_points.append(n_estimators_best)
    
    for n_est in check_points:
        if n_est > n_estimators_best:
            continue
            
        # Criar modelo temporário com os melhores parâmetros
        temp_model = xgb.XGBClassifier(
            n_estimators=n_est,
            learning_rate=learning_rate_best,
            max_depth=max_depth_best,
            subsample=subsample_best,
            colsample_bytree=colsample_best,
            min_child_weight=min_child_weight_best,
            gamma=gamma_best,
            reg_alpha=reg_alpha_best,
            reg_lambda=reg_lambda_best,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_train_scaled, y_train, verbose=False)
        
        train_pred = temp_model.predict(X_train_scaled)
        val_pred = temp_model.predict(X_val_scaled)
        train_acc.append(accuracy_score(y_train, train_pred))
        val_acc.append(accuracy_score(y_val, val_pred))
    
    # Adicionar ponto final (modelo completo)
    train_pred_final = model.predict(X_train_scaled)
    val_pred_final = model.predict(X_val_scaled)
    train_acc.append(accuracy_score(y_train, train_pred_final))
    val_acc.append(accuracy_score(y_val, val_pred_final))
    check_points.append(n_estimators_best)
    
    # Plotar curva de aprendizado
    print("\n6. Gerando gráfico de aprendizado...")
    if len(train_acc) > 0:
        x_axis = check_points[:len(train_acc)]
        if len(x_axis) != len(train_acc):
            x_axis = list(range(1, len(train_acc) + 1))
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, train_acc, label='Treino', marker='o', markersize=4)
        plt.plot(x_axis, val_acc, label='Validação', marker='s', markersize=4)
    else:
        # Fallback: usar logloss se não conseguir calcular acurácia
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_logloss) + 1), train_logloss, label='Treino (LogLoss)', marker='o', markersize=3)
        plt.plot(range(1, len(val_logloss) + 1), val_logloss, label='Validação (LogLoss)', marker='s', markersize=3)
        plt.ylabel('LogLoss', fontsize=12)
        print("   Usando LogLoss como métrica (mais eficiente)")
    plt.xlabel('Iteração (Boosting Round)', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Curva de Aprendizado - XGBoost', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve_xgb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Salvo: learning_curve_xgb.png")
    
    # Avaliar no teste
    print("\n7. Avaliando no conjunto de teste...")
    y_test_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_test_pred)
    sensitivity = recall_score(y_test, y_test_pred)  # Recall da classe positiva (Demented)
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Especificidade = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['NonDemented', 'Demented'],
                    yticklabels=['NonDemented', 'Demented'])
    else:
        # Usar matplotlib se seaborn não estiver disponível
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['NonDemented', 'Demented'])
        plt.yticks(tick_marks, ['NonDemented', 'Demented'])
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    plt.title('Matriz de Confusão - XGBoost (Teste)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_xgb.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Salvo: confusion_xgb.png")
    
    # Exibir resultados
    print("\n" + "="*70)
    print("=== CLASSIFICADOR RASO (XGBOOST) - TESTE ===")
    print("="*70)
    print(f"Parâmetros otimizados via Random Search (100 iterações, 5-fold CV, ROC-AUC)")
    print(f"Melhor ROC-AUC (CV): {best_score:.4f} ({best_score*100:.2f}%)")
    print(f"\nResultados no conjunto de TESTE:")
    print(f"  Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Sensibilidade (Recall Demented): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
    print(f"  Especificidade: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"[[{tn:4d} {fp:4d}]  <- NonDemented")
    print(f" [{fn:4d} {tp:4d}]]  <- Demented")
    print("="*70 + "\n")
    
    return model, accuracy, sensitivity, specificity


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=2):
    """Cria modelo ResNet50 com fine-tuning"""
    # Carregar ResNet50 pré-treinado
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar camadas iniciais
    for layer in base_model.layers[:-10]:  # Congelar todas exceto últimas 10
        layer.trainable = False
    
    # Adicionar camadas de classificação
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


def train_resnet50_classifier(base_image_dir="images"):
    """Treina e avalia o classificador ResNet50"""
    print("\n" + "="*70)
    print("CLASSIFICADOR PROFUNDO (RESNET50)")
    print("="*70)
    
    # Ler CSVs
    print("\n1. Carregando dados...")
    train_df = pd.read_csv("train_split.csv", sep=";", decimal=".")
    val_df = pd.read_csv("val_split.csv", sep=";", decimal=".")
    test_df = pd.read_csv("test_split.csv", sep=";", decimal=".")
    
    print(f"   Treino: {len(train_df)} exames")
    print(f"   Validação: {len(val_df)} exames")
    print(f"   Teste: {len(test_df)} exames")
    
    # Verificar se base_image_dir existe
    if not os.path.exists(base_image_dir):
        print(f"\n   AVISO: Pasta '{base_image_dir}' não encontrada!")
        print("   Tentando encontrar pasta de imagens...")
        # Tentar outras pastas comuns
        possible_dirs = ["output", "input_axl", "axl", "cor", "sag"]
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                base_image_dir = dir_name
                print(f"   Usando: {base_image_dir}")
                break
        else:
            print("   ERRO: Não foi possível encontrar pasta de imagens!")
            print("   Por favor, ajuste a variável 'base_image_dir' no código.")
            return None
    
    # Preparar dados de treino
    print("\n2. Preparando dados de treino...")
    X_train = []
    y_train = []
    train_valid_indices = []
    
    for idx, row in train_df.iterrows():
        img_path = get_image_path(row, train_df, base_image_dir)
        if img_path and os.path.exists(img_path):
            img = load_and_preprocess_image(img_path)
            if img is not None:
                X_train.append(img)
                label = 1 if row['ClassBinary'] == 'Demented' else 0
                y_train.append(label)
                train_valid_indices.append(idx)
        else:
            print(f"   Aviso: Imagem não encontrada para {row.get('MRI ID', 'N/A')}")
    
    # Preparar dados de validação
    print("\n3. Preparando dados de validação...")
    X_val = []
    y_val = []
    
    for idx, row in val_df.iterrows():
        img_path = get_image_path(row, val_df, base_image_dir)
        if img_path and os.path.exists(img_path):
            img = load_and_preprocess_image(img_path)
            if img is not None:
                X_val.append(img)
                label = 1 if row['ClassBinary'] == 'Demented' else 0
                y_val.append(label)
    
    # Preparar dados de teste
    print("\n4. Preparando dados de teste...")
    X_test = []
    y_test = []
    
    for idx, row in test_df.iterrows():
        img_path = get_image_path(row, test_df, base_image_dir)
        if img_path and os.path.exists(img_path):
            img = load_and_preprocess_image(img_path)
            if img is not None:
                X_test.append(img)
                label = 1 if row['ClassBinary'] == 'Demented' else 0
                y_test.append(label)
    
    if len(X_train) == 0:
        print("\n   ERRO: Nenhuma imagem válida encontrada para treino!")
        print("   Verifique o caminho das imagens e ajuste 'base_image_dir'.")
        return None
    
    print(f"\n   Imagens carregadas - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")
    
    # Converter para arrays numpy
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    
    # Converter labels para categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Calcular class_weight para balanceamento
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"\n5. Class weights: {class_weight_dict}")
    
    # Criar modelo
    print("\n6. Criando modelo ResNet50...")
    model = create_resnet50_model()
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Treinar
    print("\n7. Treinando ResNet50...")
    print("   (Isso pode levar alguns minutos...)")
    
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=20,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Plotar curva de aprendizado
    print("\n8. Gerando gráfico de aprendizado...")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Treino', marker='o', markersize=3)
    plt.plot(history.history['val_accuracy'], label='Validação', marker='s', markersize=3)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.title('Curva de Aprendizado - ResNet50', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve_resnet50.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Salvo: learning_curve_resnet50.png")
    
    # Avaliar no teste
    if len(X_test) > 0:
        print("\n9. Avaliando no conjunto de teste...")
        y_test_pred_proba = model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        
        accuracy = accuracy_score(y_test, y_test_pred)
        sensitivity = recall_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Plotar matriz de confusão
        plt.figure(figsize=(8, 6))
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['NonDemented', 'Demented'],
                        yticklabels=['NonDemented', 'Demented'])
        else:
            # Usar matplotlib se seaborn não estiver disponível
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['NonDemented', 'Demented'])
            plt.yticks(tick_marks, ['NonDemented', 'Demented'])
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.title('Matriz de Confusão - ResNet50 (Teste)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_resnet50.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Salvo: confusion_resnet50.png")
        
        # Exibir resultados
        print("\n" + "="*70)
        print("=== CLASSIFICADOR PROFUNDO (RESNET50) - TESTE ===")
        print("="*70)
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Sensibilidade (Recall Demented): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Especificidade: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(f"[[{tn:4d} {fp:4d}]")
        print(f" [{fn:4d} {tp:4d}]]")
        print("="*70 + "\n")
        
        return model, accuracy, sensitivity, specificity
    else:
        print("\n   Aviso: Nenhuma imagem de teste válida encontrada!")
        return model, None, None, None


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("PARTE 10: CLASSIFICADORES PARA DIAGNÓSTICO DE ALZHEIMER")
    print("="*70)
    
    # Verificar se os CSVs existem
    required_files = ["train_split.csv", "val_split.csv", "test_split.csv"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\nERRO: Arquivos não encontrados: {missing_files}")
        print("Por favor, execute a Parte 9 primeiro para gerar os splits.")
        return
    
    # Treinar XGBoost
    try:
        train_xgboost_classifier()
    except Exception as e:
        print(f"\nERRO ao treinar XGBoost: {e}")
        import traceback
        traceback.print_exc()
    
    # Treinar ResNet50
    try:
        train_resnet50_classifier()
    except Exception as e:
        print(f"\nERRO ao treinar ResNet50: {e}")
        import traceback
        traceback.print_exc()
        print("\nNota: Se o erro for relacionado a imagens não encontradas,")
        print("ajuste a variável 'base_image_dir' na função train_resnet50_classifier().")
    
    print("\n" + "="*70)
    print("PARTE 10 CONCLUÍDA!")
    print("="*70)
    print("\nArquivos gerados:")
    print("  - learning_curve_xgb.png")
    print("  - confusion_xgb.png")
    print("  - learning_curve_resnet50.png")
    print("  - confusion_resnet50.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

