# 📚 CONTEXTO E OBJETIVOS DO TRABALHO PRÁTICO

## 🎯 OBJETIVO GERAL

**Trabalho Prático: Métodos de segmentação e reconhecimento de imagens aplicados ao diagnóstico da Doença de Alzheimer**

Desenvolver um sistema completo que:
1. **Segmenta** os ventrículos laterais em imagens de ressonância magnética (MRI)
2. **Extrai características** morfológicas dos ventrículos segmentados
3. **Classifica** pacientes como Demented ou NonDemented (2 classificadores)
4. **Estima a idade** dos pacientes (2 regressores)
5. **Compara** os resultados entre diferentes abordagens

---

## 📊 DATASET: OASIS-2

### **Características:**
- **150 indivíduos** com idades entre 60-96 anos
- **373 sessões de imagem** (longitudinal - múltiplas visitas)
- **72 indivíduos:** Não dementes ao longo do estudo
- **64 indivíduos:** Dementes desde a visita inicial
- **14 indivíduos:** Converted (não dementes → dementes)

### **Dados Disponíveis:**
- **Imagens MRI:** Formatos Nifti, PNG, JPG
- **Planos:** Axial, Sagital, Coronal (depende do grupo)
- **Dados demográficos:** CSV com:
  - Group (Demented/Nondemented/Converted)
  - Age, Sex, Education, CDR, MMSE
  - eTIV, nWBV (volumes cerebrais)
  - E outros...

### **Pré-processamento já feito:**
- Extração do cérebro (fslr)
- Conversão para 8 bits
- Registro com Atlas MNI152
- Extração de planos específicos

---

## 🔧 ESPECIFICAÇÕES TÉCNICAS DO TRABALHO

### **1. Ambiente Gráfico (Interface)**
✅ **Implementado:**
- Menu completo com todas funcionalidades
- Acessibilidade (aumento de texto)
- Leitura e exibição de imagens (Nifti, PNG, JPG)
- Zoom nas imagens
- Interface intuitiva com abas

### **2. Segmentação dos Ventrículos Laterais**
✅ **Implementado:**
- **Método:** Region Growing (Crescimento de Região)
- **Parâmetros configuráveis:**
  - Threshold (variação de intensidade)
  - Conectividade (4 ou 8 vizinhos)
  - Operações morfológicas (Abertura, Preencher buracos, Suavizar)
- **Modos:**
  - Manual (clicando na imagem)
  - Automático (seeds fixos)
  - Processamento em lote

### **3. Caracterização (Descritores)**
✅ **Implementado:**
- **6 descritores morfológicos:**
  1. **Área** (area)
  2. **Perímetro** (perimeter)
  3. **Circularidade** (circularity)
  4. **Excentricidade** (eccentricity)
  5. **Solidez** (solidity)
  6. **Extensão** (extent)
- **Planilha gerada:** `descritores.csv` (complementar ao dataset)

### **4. Gráficos de Dispersão (Scatterplots)**
✅ **Implementado:**
- Plotagem de características aos pares
- **Cores:**
  - 🔵 Azul: NonDemented
  - 🔴 Vermelho: Demented
  - ⚫ Preto: Converted
- Permite verificar separabilidade das classes

### **5. Separação dos Dados**
✅ **Implementado:**
- **80% treino** / **20% teste** (por paciente, não por exame)
- **20% do treino** para validação
- **Balanceamento:** 4:1 em cada conjunto
- **Classes:** Demented vs NonDemented
  - Converted com CDR=0 → NonDemented
  - Converted com CDR>0 → Demented
- **Sem mistura:** Mesmo paciente não aparece em treino e teste

### **6. Classificadores**

#### **Classificador Raso: XGBoost** ✅
- **Entrada:** 5 descritores morfológicos (área, perímetro, excentricidade, extensão, solidez)
- **Otimização:** Random Search (100 iterações, 3-fold CV)
- **Métrica:** ROC-AUC
- **Early Stopping:** 50 rounds sem melhoria
- **Avaliação:** Acurácia, Sensibilidade, Especificidade, Matriz de Confusão

#### **Classificador Profundo: ResNet50** ✅
- **Entrada:** Imagens completas (224x224, RGB)
- **Transfer Learning:** Fine-tuning do ImageNet
- **Estratégia:**
  - Estágio 1: Backbone congelado (treina apenas head)
  - Estágio 2: Fine-tuning (descongela últimas camadas)
- **Loss:** Focal Loss (alpha=0.75, gamma=2.5)
- **Data Augmentation:** Rotação, zoom, translação, contraste, ruído
- **Avaliação:** Acurácia, Sensibilidade, Especificidade, Matriz de Confusão, ROC, Precision-Recall

### **7. Regressores**

#### **Regressor Raso: Regressão Linear** ✅
- **Entrada:** 5 descritores morfológicos
- **Pipeline:** StandardScaler + LinearRegression
- **Avaliação:** MAE, RMSE, R²

#### **Regressor Profundo: ResNet50** ✅
- **Entrada:** Imagens completas
- **Transfer Learning:** Fine-tuning do ImageNet
- **Loss:** Huber Loss
- **Avaliação:** MAE, RMSE, R²

### **8. Comparação de Resultados** ✅
- Comparação entre classificadores (raso vs profundo)
- Comparação entre regressores (raso vs profundo)
- Análise de limitações e recomendações

---

## 📈 RESULTADOS GERADOS

### **Gráficos:**
1. **Curvas de aprendizado:**
   - `learning_curve_xgb.png` (XGBoost)
   - `learning_curve_resnet50.png` (ResNet50)

2. **Matrizes de confusão:**
   - `confusion_xgb.png`
   - `confusion_resnet50.png`

3. **Curvas ROC e Precision-Recall:**
   - `roc_pr_curves_resnet50.png`

4. **Scatterplots:**
   - Múltiplos gráficos de características aos pares

5. **Regressão:**
   - `pred_vs_real_raso.png` (Regressor Linear)
   - `pred_vs_real_profundo.png` (ResNet50)

### **Arquivos CSV:**
- `train_split.csv`, `val_split.csv`, `test_split.csv`
- `descritores.csv` (características extraídas)
- `merged_data.csv` (dados combinados)

---

## 🎓 CONCEITOS IMPLEMENTADOS

### **Segmentação:**
- **Region Growing:** Algoritmo de crescimento de região
- **Operações Morfológicas:** Abertura, Preenchimento, Suavização
- **Validação:** Limite de pixels para detectar falhas

### **Classificação:**
- **XGBoost:** Gradient Boosting com otimização de hiperparâmetros
- **ResNet50:** Deep Learning com Transfer Learning
- **Focal Loss:** Para lidar com classes desbalanceadas
- **Fine-tuning:** Estratégia de treinamento em 2 estágios

### **Regressão:**
- **Regressão Linear:** Modelo simples e interpretável
- **ResNet50:** Deep Learning para estimar idade

### **Pré-processamento:**
- **CLAHE:** Equalização adaptativa de histograma
- **Normalização:** Clipping percentil, min-max scaling
- **Data Augmentation:** Rotação, zoom, translação, etc.

---

## 📝 ESTRUTURA DO CÓDIGO

### **Arquivo Único:** `app.py` (6541 linhas)
- Interface gráfica completa (Tkinter)
- Todas as funcionalidades integradas
- Processamento de imagens (OpenCV, PIL)
- Machine Learning (XGBoost, TensorFlow/Keras)
- Visualização (Matplotlib, Seaborn)

### **Funcionalidades Principais:**
1. **Parte 1-7:** Interface de segmentação e extração de características
2. **Parte 8:** Geração de scatterplots
3. **Parte 9:** Split de dados (treino/validação/teste)
4. **Parte 10:** Classificadores (XGBoost e ResNet50)
5. **Parte 11:** Regressores (Linear e ResNet50)
6. **Parte 12:** Comparação de resultados

---

## 🎯 PONTOS-CHAVE PARA APRESENTAÇÃO

### **1. Problema:**
- Doença de Alzheimer causa variações volumétricas no cérebro
- Ventrículos laterais aumentam com a doença
- Necessidade de diagnóstico auxiliado por computador

### **2. Solução:**
- Segmentação automática dos ventrículos
- Extração de características morfológicas
- Classificação usando métodos rasos e profundos
- Estimação de idade para análise longitudinal

### **3. Diferenciais:**
- **Interface completa:** Tudo em um único arquivo
- **Múltiplos métodos:** Comparação entre abordagens
- **Validação rigorosa:** Separação por paciente (não por exame)
- **Transfer Learning:** Aproveitamento de modelos pré-treinados

### **4. Resultados:**
- Classificadores: Acurácia, Sensibilidade, Especificidade
- Regressores: MAE, RMSE, R²
- Análise comparativa entre métodos

---

## 📚 REFERÊNCIAS IMPORTANTES

**Dataset:**
- OASIS-2: Open Access Series of Imaging Studies
- Marcus, D.S., et al. "Open access series of imaging studies: longitudinal MRI data in nondemented and demented older adults." Journal of cognitive neuroscience 22.12 (2010): 2677-2684

**Bibliotecas Utilizadas:**
- OpenCV (processamento de imagens)
- TensorFlow/Keras (deep learning)
- XGBoost (gradient boosting)
- Scikit-learn (métricas, pré-processamento)
- Matplotlib/Seaborn (visualização)
- Tkinter (interface gráfica)

---

## ✅ CHECKLIST DE ENTREGA

- [x] Arquivo-fonte único (`app.py`)
- [x] Planilhas CSV geradas
- [x] Documentação em LaTeX e PDF
- [x] Todas as funcionalidades implementadas
- [x] Interface gráfica completa
- [x] Segmentação funcional
- [x] Classificadores implementados
- [x] Regressores implementados
- [x] Gráficos gerados
- [x] Comparação de resultados

---

## 💡 MENSAGEM FINAL

Este trabalho demonstra a aplicação prática de:
- **Processamento de Imagens:** Segmentação de estruturas anatômicas
- **Machine Learning:** Classificação e regressão
- **Deep Learning:** Transfer Learning com ResNet50
- **Análise de Dados:** Caracterização morfológica e visualização

**Objetivo alcançado:** Sistema completo para auxiliar no diagnóstico da Doença de Alzheimer através da análise de imagens de ressonância magnética.

