"""
Partes 8 e 9 do Trabalho Prático de Processamento e Análise de Imagens

Parte 8: Scatterplots aos pares das características ventriculares
Parte 9: Split treino/validação/teste por paciente (sem vazamento)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def parte_8_scatterplots(df, output_dir="scatterplots"):
    """
    Parte 8: Cria scatterplots aos pares das características ventriculares.
    
    Args:
        df: DataFrame com os dados
        output_dir: Diretório para salvar os gráficos
    """
    print("=" * 60)
    print("PARTE 8: Scatterplots aos pares")
    print("=" * 60)
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Definir características ventriculares esperadas
    descriptor_cols = [
        'area', 'perimeter', 'circularity', 'eccentricity', 
        'solidity', 'extent', 'aspect_ratio'
    ]
    
    # Verificar quais colunas existem no DataFrame
    available_cols = [col for col in descriptor_cols if col in df.columns]
    missing_cols = [col for col in descriptor_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Aviso: Colunas não encontradas: {missing_cols}")
        print(f"Usando colunas disponíveis: {available_cols}")
    
    if len(available_cols) < 2:
        print("Erro: Menos de 2 características ventriculares encontradas!")
        return
    
    # Converter valores string para numérico
    for col in available_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Mapear cores por Group
    color_map = {
        'Converted': 'black',
        'NonDemented': 'blue',
        'Nondemented': 'blue',  # Variante possível
        'Demented': 'red'
    }
    
    # Normalizar nomes de Group
    df['Group_normalized'] = df['Group'].str.strip()
    
    # Gerar todos os pares de características
    pairs = list(itertools.combinations(available_cols, 2))
    print(f"\nGerando {len(pairs)} scatterplots...")
    
    for feat_i, feat_j in pairs:
        # Filtrar dados válidos (sem NaN)
        valid_mask = df[[feat_i, feat_j]].notna().all(axis=1)
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            print(f"Aviso: Sem dados válidos para {feat_i} vs {feat_j}")
            continue
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        
        # Plotar pontos por grupo
        for group_name, color in color_map.items():
            group_data = df_valid[df_valid['Group_normalized'].str.contains(group_name, case=False, na=False)]
            if len(group_data) > 0:
                plt.scatter(
                    group_data[feat_i], 
                    group_data[feat_j],
                    c=color,
                    label=group_name,
                    alpha=0.6,
                    s=50
                )
        
        # Configurar gráfico
        plt.xlabel(feat_i.replace('_', ' ').title(), fontsize=12)
        plt.ylabel(feat_j.replace('_', ' ').title(), fontsize=12)
        plt.title(f'{feat_i.replace("_", " ").title()} vs {feat_j.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Salvar figura
        filename = f"{feat_i}_vs_{feat_j}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [OK] {filename}")
    
    print(f"\nScatterplots salvos em: {output_dir}/")
    print(f"Total: {len(pairs)} gráficos gerados")


def parte_9_split_pacientes(df):
    """
    Parte 9: Split treino/validação/teste por paciente (sem vazamento).
    
    Args:
        df: DataFrame com os dados
    """
    print("\n" + "=" * 60)
    print("PARTE 9: Split treino/validação/teste por paciente")
    print("=" * 60)
    
    # Copiar DataFrame para não modificar o original
    df_work = df.copy()
    
    # Converter CDR para numérico
    df_work['CDR'] = pd.to_numeric(df_work['CDR'], errors='coerce')
    
    # Normalizar Group
    df_work['Group'] = df_work['Group'].str.strip()
    
    print("\n1. Reclassificando 'Converted' para 2 classes binárias...")
    
    # Reclassificar Converted baseado em CDR
    converted_mask = df_work['Group'].str.contains('Converted', case=False, na=False)
    
    # Converted com CDR == 0 → NonDemented
    converted_cdr0 = converted_mask & (df_work['CDR'] == 0)
    df_work.loc[converted_cdr0, 'Group'] = 'Nondemented'
    
    # Converted com CDR > 0 → Demented
    converted_cdr_pos = converted_mask & (df_work['CDR'] > 0)
    df_work.loc[converted_cdr_pos, 'Group'] = 'Demented'
    
    # Normalizar novamente após reclassificação
    df_work['Group'] = df_work['Group'].str.strip()
    
    # Criar coluna binária (case-insensitive)
    def classify_binary(group_str):
        group_lower = str(group_str).lower()
        if 'demented' in group_lower and 'non' not in group_lower:
            return 'Demented'
        elif 'nondemented' in group_lower or ('non' in group_lower and 'demented' in group_lower):
            return 'NonDemented'
        else:
            return None
    
    df_work['ClassBinary'] = df_work['Group'].apply(classify_binary)
    
    # Remover linhas que não são Demented ou NonDemented
    df_bin = df_work[df_work['ClassBinary'].isin(['Demented', 'NonDemented'])].copy()
    
    print(f"   Total de exames após reclassificação: {len(df_bin)}")
    print(f"   Demented: {len(df_bin[df_bin['ClassBinary'] == 'Demented'])}")
    print(f"   NonDemented: {len(df_bin[df_bin['ClassBinary'] == 'NonDemented'])}")
    
    print("\n2. Definindo classe por paciente...")
    
    # Definir classe do paciente: Demented se qualquer exame for Demented, senão NonDemented
    def get_patient_class(class_series):
        # Se qualquer exame for Demented, o paciente é Demented
        if 'Demented' in class_series.values:
            return 'Demented'
        else:
            return 'NonDemented'
    
    patient_labels = df_bin.groupby('Subject ID')['ClassBinary'].apply(get_patient_class).reset_index()
    patient_labels.columns = ['Subject ID', 'PatientClass']
    
    print(f"   Total de pacientes únicos: {len(patient_labels)}")
    print(f"   Pacientes Demented: {len(patient_labels[patient_labels['PatientClass'] == 'Demented'])}")
    print(f"   Pacientes NonDemented: {len(patient_labels[patient_labels['PatientClass'] == 'NonDemented'])}")
    
    print("\n3. Fazendo split treino/teste (80/20) por paciente...")
    
    # Split treino/teste (80/20) estratificado por paciente
    train_patients, test_patients = train_test_split(
        patient_labels['Subject ID'].values,
        test_size=0.2,
        stratify=patient_labels['PatientClass'].values,
        random_state=42
    )
    
    print(f"   Pacientes no treino: {len(train_patients)}")
    print(f"   Pacientes no teste: {len(test_patients)}")
    
    # Filtrar exames por lista de pacientes
    df_train_full = df_bin[df_bin['Subject ID'].isin(train_patients)].copy()
    df_test = df_bin[df_bin['Subject ID'].isin(test_patients)].copy()
    
    print("\n4. Separando validação do treino (20% dos pacientes do treino)...")
    
    # Dentro do treino, separar validação (20% dos pacientes)
    train_patient_labels = patient_labels[patient_labels['Subject ID'].isin(train_patients)]
    
    train_patients_final, val_patients = train_test_split(
        train_patient_labels['Subject ID'].values,
        test_size=0.2,
        stratify=train_patient_labels['PatientClass'].values,
        random_state=42
    )
    
    print(f"   Pacientes no treino final: {len(train_patients_final)}")
    print(f"   Pacientes na validação: {len(val_patients)}")
    
    # Filtrar exames
    df_train = df_bin[df_bin['Subject ID'].isin(train_patients_final)].copy()
    df_val = df_bin[df_bin['Subject ID'].isin(val_patients)].copy()
    
    print("\n5. Estatísticas dos splits:")
    print("\n   TREINO:")
    print(f"      Exames: {len(df_train)}")
    print(f"      Pacientes: {df_train['Subject ID'].nunique()}")
    print(f"      Demented: {len(df_train[df_train['ClassBinary'] == 'Demented'])} exames")
    print(f"      NonDemented: {len(df_train[df_train['ClassBinary'] == 'NonDemented'])} exames")
    train_patients_classes = df_train.groupby('Subject ID')['ClassBinary'].apply(get_patient_class)
    print(f"      Pacientes Demented: {len(train_patients_classes[train_patients_classes == 'Demented'])}")
    print(f"      Pacientes NonDemented: {len(train_patients_classes[train_patients_classes == 'NonDemented'])}")
    
    print("\n   VALIDAÇÃO:")
    print(f"      Exames: {len(df_val)}")
    print(f"      Pacientes: {df_val['Subject ID'].nunique()}")
    print(f"      Demented: {len(df_val[df_val['ClassBinary'] == 'Demented'])} exames")
    print(f"      NonDemented: {len(df_val[df_val['ClassBinary'] == 'NonDemented'])} exames")
    val_patients_classes = df_val.groupby('Subject ID')['ClassBinary'].apply(get_patient_class)
    print(f"      Pacientes Demented: {len(val_patients_classes[val_patients_classes == 'Demented'])}")
    print(f"      Pacientes NonDemented: {len(val_patients_classes[val_patients_classes == 'NonDemented'])}")
    
    print("\n   TESTE:")
    print(f"      Exames: {len(df_test)}")
    print(f"      Pacientes: {df_test['Subject ID'].nunique()}")
    print(f"      Demented: {len(df_test[df_test['ClassBinary'] == 'Demented'])} exames")
    print(f"      NonDemented: {len(df_test[df_test['ClassBinary'] == 'NonDemented'])} exames")
    test_patients_classes = df_test.groupby('Subject ID')['ClassBinary'].apply(get_patient_class)
    print(f"      Pacientes Demented: {len(test_patients_classes[test_patients_classes == 'Demented'])}")
    print(f"      Pacientes NonDemented: {len(test_patients_classes[test_patients_classes == 'NonDemented'])}")
    
    # Verificar vazamento
    train_ids = set(df_train['Subject ID'].unique())
    val_ids = set(df_val['Subject ID'].unique())
    test_ids = set(df_test['Subject ID'].unique())
    
    print("\n6. Verificação de vazamento:")
    if train_ids & val_ids:
        print("   [ERRO] Vazamento entre treino e validação!")
    else:
        print("   [OK] Sem vazamento entre treino e validação")
    
    if train_ids & test_ids:
        print("   [ERRO] Vazamento entre treino e teste!")
    else:
        print("   [OK] Sem vazamento entre treino e teste")
    
    if val_ids & test_ids:
        print("   [ERRO] Vazamento entre validação e teste!")
    else:
        print("   [OK] Sem vazamento entre validação e teste")
    
    print("\n7. Salvando CSVs...")
    
    # Salvar CSVs com sep=";" e decimal="."
    # Nota: O pandas salva com ponto como decimal por padrão
    df_train.to_csv('train_split.csv', sep=';', decimal='.', index=False)
    df_val.to_csv('val_split.csv', sep=';', decimal='.', index=False)
    df_test.to_csv('test_split.csv', sep=';', decimal='.', index=False)
    
    print("   [OK] train_split.csv")
    print("   [OK] val_split.csv")
    print("   [OK] test_split.csv")
    
    print("\n" + "=" * 60)
    print("Parte 9 concluída com sucesso!")
    print("=" * 60)


def main():
    """Função principal"""
    print("\n" + "=" * 60)
    print("TRABALHO PRÁTICO - PARTES 8 E 9")
    print("=" * 60)
    
    # Ler CSV
    print("\nLendo merged_data.csv...")
    # O CSV usa vírgula como separador decimal, mas o usuário especificou decimal="."
    # Vamos ler com vírgula e depois converter se necessário
    try:
        df = pd.read_csv("merged_data.csv", sep=";", decimal=",")
        print("   CSV lido com sucesso (decimal=',')")
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        raise
    
    print(f"   Total de linhas: {len(df)}")
    print(f"   Total de colunas: {len(df.columns)}")
    print(f"   Colunas: {list(df.columns)}")
    
    # Executar Parte 8
    parte_8_scatterplots(df)
    
    # Executar Parte 9
    parte_9_split_pacientes(df)
    
    print("\n" + "=" * 60)
    print("TODAS AS PARTES CONCLUÍDAS!")
    print("=" * 60)


if __name__ == "__main__":
    main()

