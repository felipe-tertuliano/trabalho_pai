## Trabalho PAI – Aplicativo de Segmentação e Análise de Imagens (AlzheimerApp)

Este projeto implementa uma aplicação **Tkinter** para:
- visualizar cortes de RM (NIfTI, PNG, JPG);
- segmentar automaticamente os **ventrículos cerebrais** (região em “X”);
- extrair características geométricas;
- preparar dados para modelos de ML/DL;
- processar **todo um conjunto de arquivos .nii em lote**.

A interface foi construída para permitir um fluxo de trabalho totalmente visual e controlado pelo usuário.

---

## 1. Pré‑requisitos

Certifique‑se de ter instalado:

- **Python 3.8 ou superior**
- **pip** (gerenciador de pacotes)

### Ambiente virtual (recomendado)

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Instalar dependências

Com o ambiente virtual ativo:

```bash
pip install -r requirements.txt
```

---

## 2. Executando o aplicativo

Na pasta do projeto:

```bash
python app.py
```

Uma janela Tkinter será aberta com a interface gráfica do **AlzheimerApp**.

---

## 3. Visão geral da interface

A janela principal possui:

- **Painel esquerdo (imagens)**  
  - `Imagem Original`: corte coronal em escala de cinza.  
  - `Pré-processada (Filtros)`: visualização de filtros (Otsu+CLAHE).  
  - `Segmentada (Contorno Amarelo)`: imagem original com o contorno dos ventrículos em **amarelo**.

- **Painel direito (controles)**  
  - Carregamento de **CSV** e **imagens**.  
  - Controles de zoom/reset das três janelas.  
  - Pré‑processamento (Otsu + CLAHE).  
  - Segmentação automática e manual (multi‑seed).  
  - Processamento em lote (`.nii`).  
  - Extração de características e demais módulos de ML/DL (placeholders).

Em todas as janelas é possível dar **zoom com a roda do mouse** e **arrastar** com o botão esquerdo pressionado.

---

## 4. Fluxo de segmentação (imagem única)

### 4.1 Carregar imagem

1. Clique em **“Carregar Imagem”**.  
2. Selecione um arquivo:
   - `*.nii` ou `*.nii.gz` (NIfTI 2D/3D – para 3D é usado o slice coronal central);
   - ou `*.png`, `*.jpg`, `*.jpeg`, `*.bmp`.
3. A imagem é convertida para **escala de cinza** e exibida em `Imagem Original`.

### 4.2 Pré‑processamento (Otsu + CLAHE)

- Clique em **“Aplicar Otsu + CLAHE”**.  
- A imagem binarizada (branco = cérebro, preto = fundo/regiões escuras) é exibida em `Pré-processada (Filtros)`.  
- Esta imagem é usada quando você clica diretamente na janela de pré‑processamento.

### 4.3 Escolha da imagem usada no Region Growing

No painel **Segmentação** existem duas opções (radiobutton):

- **CLAHE (Escala Cinza)**  
  Region Growing é aplicado sobre a imagem original equalizada (mais informação de intensidade).

- **Otsu (Binarizada)**  
  Region Growing é aplicado sobre uma versão binarizada (CLAHE + limiarização de Otsu).

> Observação: quando você clica diretamente na janela **Pré-processada**, o algoritmo usa exatamente a imagem Otsu+CLAHE que está sendo exibida, independentemente do radiobutton.

### 4.4 Segmentação automática (seeds fixos)

- Na seção **“1. Automática (Seeds Fixos)”**:
  - Seeds pré-definidos: `(164, 91)` e `(171, 114)` (coordenadas no plano da imagem).  
  - Threshold do Region Growing: **50** (fixo).  
  - Kernel morfológico: **15×15**.  
  - Morfologia: **Abertura + Fechamento + Preenchimento de buracos + Suavização de contornos**.

Ao clicar em **“▶ Segmentação Automática”**:

1. A imagem é preparada (CLAHE ou Otsu, conforme escolha).  
2. O Region Growing é executado para cada seed.  
3. As máscaras são combinadas (união).  
4. É aplicado o pós‑processamento morfológico completo.  
5. O resultado final (máscara) é salvo em `self.image_mask`.  
6. A imagem original é convertida para RGB e os contornos da máscara são desenhados em **amarelo**.  
7. O resultado é exibido em `Segmentada (Contorno Amarelo)` e um resumo aparece no log (nº de regiões, pixels, área).

### 4.5 Segmentação manual (Multi‑Seed)

Na seção **“2. Manual (Multi-Seed)”**:

- **“Iniciar Multi-Seed”**  
  - Ativa o modo multi‑seed manual.  
  - Cada clique na **Imagem Original** ou na **Pré‑processada** adiciona um seed:
    - Os pontos ficam armazenados em `self.multi_seed_points`.  
    - A cada clique o Region Growing é rodado para aquele ponto e a máscara é **acumulada** em `self.accumulated_mask`.  
    - A máscara acumulada é pós‑processada e exibida em `Segmentada`.

- **“Finalizar”**  
  - Desativa o modo multi‑seed e mantém a máscara acumulada.

- **“💾 Salvar Pontos Multi-Seed”**  
  - Exporta os pontos coletados para o **log**, em formato Python:
    ```python
    auto_seed_points = [
        (x1, y1),  # Ponto 1
        (x2, y2),  # Ponto 2
        ...
    ]
    ```
  - Esses pontos podem ser copiados e colados em `self.auto_seed_points` para criar uma segmentação automática personalizada.

Além disso, existe uma seção de **Coordenadas do Mouse**, que mostra em tempo real:
- `X: xxx | Y: yyy` quando o cursor está sobre a imagem original;  
- `X: xxx | Y: yyy [PRÉ-PROC]` quando está sobre a pré‑processada.

Também é possível registrar e exportar pontos simples (fora do modo multi‑seed) para estudo.

---

## 5. Processamento em lote (.nii)

Na seção **“📁 Processamento em Lote”**:

- Botão **“🔄 Segmentar Pasta Inteira (.nii)”**:
  1. Solicita a **pasta de entrada** contendo arquivos `.nii` ou `.nii.gz`.  
  2. Solicita a **pasta de saída** onde serão salvos os resultados.  
  3. Lista todos os arquivos `.nii`/`.nii.gz` da pasta.  
  4. Para cada arquivo:
     - Carrega o NIfTI com `nibabel`.  
     - Se for 3D, extrai o **slice central** do eixo coronal.  
     - Normaliza o contraste para 0–255.  
     - Prepara a imagem (CLAHE ou Otsu, usando a mesma função de segmentação individual).  
     - Executa o Region Growing multi‑seed com os seeds fixos (`self.auto_seed_points`).  
     - Aplica o mesmo pós‑processamento morfológico completo.  
     - Salva:
       - `nome_mask.png` → máscara binária.  
       - `nome_segmented.png` → imagem com contorno amarelo.
  5. Exibe um **relatório final** no log (arquivos processados, sucessos, erros) e atualiza o status na interface.

Isso permite aplicar a mesma segmentação em todo o dataset de forma automática e consistente.

---

## 6. Extração de características e ML/DL

O código já possui estrutura para:

- **`extract_features`**  
  - Usa `self.image_mask` para encontrar contornos dos ventrículos.  
  - Calcula diversas métricas geométricas (área, circularidade, excentricidade, extensão, solidez, diâmetro equivalente).  
  - Futuramente esses valores podem ser salvos em um `DataFrame` e exportados para CSV.

- **Funções de ML/DL (`prepare_data`, `run_shallow_classifier`, `run_shallow_regressor`, `run_deep_classifier`, `run_deep_regressor`)**  
  - Estão estruturadas como **TODOs**, com comentários guiando como integrar as features extraídas com regressão linear, XGBoost e ResNet50.

Essas partes podem ser completadas posteriormente, reutilizando as máscaras geradas pelo módulo de segmentação.

---

## 7. Resumo do que o código faz hoje

- Abre uma **GUI Tkinter** para manipulação de imagens de RM.  
- Permite carregar imagens (`.nii`, `.nii.gz`, `.png`, `.jpg`...).  
- Exibe **três visões**: original, pré‑processada (Otsu+CLAHE) e segmentada (contorno amarelo).  
- Implementa **Region Growing** 8‑conexo com threshold fixo (=50).  
- Aplica um pipeline de **pós‑processamento morfológico completo** (kernel 15×15).  
- Suporta:
  - segmentação automática com seeds fixos;  
  - segmentação manual multi‑seed com exportação dos pontos;  
  - processamento em lote de arquivos `.nii`;  
  - extração básica de características geométricas dos ventrículos.  

Esse README descreve o comportamento atual do aplicativo para facilitar o uso, manutenção e documentação do trabalho.
