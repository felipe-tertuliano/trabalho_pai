Olá! Este é um trabalho prático bem completo de Processamento e Análise de Imagens. [cite_start]O objetivo principal é que você e seu grupo apliquem métodos de segmentação e reconhecimento de imagens para ajudar no diagnóstico da Doença de Alzheimer[cite: 4, 7].

Vou resumir o que precisa ser feito, passo a passo:

### 🎯 O Objetivo Central

[cite_start]O trabalho consiste em criar um programa de computador que consegue analisar imagens de ressonância magnética do cérebro[cite: 12]. Este programa deverá:
1.  [cite_start]Isolar (segmentar) uma região específica do cérebro (os ventrículos laterais)[cite: 78].
2.  [cite_start]Extrair medidas (características) dessa região[cite: 79].
3.  [cite_start]Usar essas medidas e as próprias imagens para treinar modelos de Inteligência Artificial (Machine Learning)[cite: 94, 96, 99, 101].
4.  Esses modelos devem tentar fazer duas coisas:
    * [cite_start]**Classificar:** Dizer se um paciente é "Demente" ou "Não Demente"[cite: 89].
    * [cite_start]**Regredir:** Estimar a idade do paciente no momento do exame[cite: 99].

---

### 📋 Suas Tarefas (Passo a Passo)

[cite_start]Aqui está o que seu grupo (de 3 ou 4 pessoas) [cite: 9] precisa fazer:

**1. Definir suas Ferramentas (Sorteio)**
O trabalho especifica quais modelos de Machine Learning vocês vão usar. [cite_start]Isso é definido pela soma dos números de matrícula dos membros do grupo[cite: 49]. Vocês precisam calcular 4 números (DS, NR, NC, ND) para saber:
* [cite_start]Qual corte do cérebro usar (coronal, sagital ou axial) [cite: 50-53].
* [cite_start]Qual será seu regressor "raso" (Linear ou XGBoost)[cite: 54].
* [cite_start]Qual será seu classificador "raso" (XGBoost ou SVM)[cite: 55].
* [cite_start]Qual será seu classificador/regressor "profundo" (ResNet50, DenseNet, EfficientNet ou MobileNet) [cite: 56-60].

**2. Construir o Programa (Base)**
[cite_start]Vocês devem criar um programa em C++, Python ou Java [cite: 66] que tenha:
* [cite_start]Uma **interface gráfica** com um menu[cite: 74].
* [cite_start]Uma função de acessibilidade (ex: aumentar o texto dos menus)[cite: 75].
* [cite_start]Uma função para ler e exibir as imagens (formatos Nifti, PNG, JPG) com opção de **zoom**[cite: 76, 77].

**3. Segmentação e Extração de Características**
Esta é a parte central do processamento de imagem:
* [cite_start]**Segmentar os Ventrículos Laterais:** Implementar uma função que consiga "desenhar" o contorno dos ventrículos laterais nas imagens (como mostrado na Figura da página 4)[cite: 78]. O método (como fazer isso) é de escolha livre do grupo.
* [cite_start]**Extrair Características:** Após segmentar, vocês devem calcular 6 medidas (descritores) dessa região: área, circularidade, excentricidade e mais 3 que vocês escolherem[cite: 79].
* [cite_start]**Visualizar:** Criar gráficos de dispersão (scatterplots) comparando essas características, usando cores diferentes para cada classe de paciente (Demente, Não Demente, Convertido)[cite: 81, 83].

**4. Preparar os Dados para IA**
Antes de treinar os modelos, vocês precisam organizar os dados:
* [cite_start]**Dividir os Dados:** Separar 80% dos pacientes para treino e 20% para teste[cite: 88]. [cite_start]Do conjunto de treino, separar 20% para validação[cite: 92].
* [cite_start]**Regra Crucial:** Exames do *mesmo paciente* não podem estar misturados nos conjuntos de treino e teste[cite: 93].
* [cite_start]**Ajustar Classes:** O grupo "Converted" deve ser dividido: exames com CDR=0 vão para a classe "NonDemented" e exames com CDR>0 vão para a "Demented"[cite: 90].

**5. Treinar os Modelos de Classificação (Demente vs. Não Demente)**
Vocês implementarão os dois classificadores que foram "sorteados" para o seu grupo (Passo 1):
* [cite_start]**Classificador Raso (ex: SVM):** Deve usar as 6 características que vocês extraíram (área, circularidade, etc.) como entrada[cite: 96].
* [cite_start]**Classificador Profundo (ex: ResNet50):** Deve usar as próprias imagens como entrada[cite: 96]. [cite_start]Vocês devem usar "fine-tuning" (ajustar os pesos)[cite: 97].
* [cite_start]**Avaliar:** Mostrar a acurácia, sensibilidade, especificidade e as matrizes de confusão para o conjunto de teste[cite: 95].

**6. Treinar os Modelos de Regressão (Estimar Idade)**
[cite_start]Fazer o mesmo processo, mas agora para estimar a idade do paciente[cite: 99]:
* [cite_start]**Regressor Raso (ex: Linear):** Usa as 6 características como entrada[cite: 100].
* [cite_start]**Regressor Profundo (ex: ResNet50):** Usa as imagens como entrada[cite: 101].
* [cite_start]**Analisar:** Vocês devem discutir se os resultados são bons e se os modelos conseguem prever idades maiores para exames feitos em visitas posteriores[cite: 101, 102].

**7. Documentação (O Artigo)**
[cite_start]Todo o trabalho deve ser documentado como um artigo científico no formato LaTeX (estilo SBC)[cite: 104]. Este artigo deve conter:
* [cite_start]Descrição do problema, do dataset e das técnicas usadas (principalmente a segmentação)[cite: 105, 106, 107].
* [cite_start]Resultados, análise, gráficos, exemplos de acertos e erros[cite: 111].
* [cite_start]Referências (bibliotecas usadas, etc.)[cite: 109, 112].

---

### ⚠️ Regras e Entregas (Muito Importante!)

* **Arquivo Fonte ÚNICO:** O programa (C++, Python ou Java) deve ser entregue em um único arquivo. [cite_start]**Notebooks (como .ipynb) não são aceitos**[cite: 66, 67].
* **Prazo:** Não se admite atraso. [cite_start]A entrega fora do prazo anula a nota[cite: 9, 122].
* **Tamanho:** O arquivo .zip final (com código, planilhas, .tex e .pdf) não pode passar de **10 Mbytes**. [cite_start]**NÃO inclua a base de dados**[cite: 119, 120].
* **Plágio:** Tolerância zero para cópias ou trabalhos gerados por IA. [cite_start]Isso zera a nota do grupo[cite: 123].

Isso resume o trabalho. É um projeto desafiador que cobre todo o fluxo de um problema de visão computacional, desde a leitura da imagem até a avaliação de modelos de deep learning.

Posso ajudar a detalhar algum desses passos que ainda pareça confuso?