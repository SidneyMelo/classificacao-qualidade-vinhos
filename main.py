"""
🍷 Sistema de Classificação de Qualidade de Vinhos

Pipeline completo para análise, classificação e comparação de modelos preditivos
aplicados ao Wine Quality Dataset. Este sistema:

- Implementa Random Forest como baseline de Machine Learning tradicional
- Integra o Google Gemini em modos few-shot e zero-shot
- Oferece suporte a modelos fine-tuned para classificação especializada
- Fornece interface interativa com menu para fácil navegação
- Gera visualizações comparativas de desempenho entre modelos
- Permite testes personalizados com valores reais ou exemplos pré-definidos
- Exporta arquivos para fine-tuning e resultados para análise posterior

Desenvolvido para pesquisa e experimentação na comparação entre 
abordagens tradicionais de ML e Modelos de Linguagem de Grande Escala (LLMs)
em tarefas de classificação estruturada.
"""
# Bibliotecas padrão do Python
import os
import time
import json
import re
import datetime
import glob

# Bibliotecas de dados e análise
import numpy as np
import pandas as pd

# Bibliotecas de visualização
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Bibliotecas de machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Bibliotecas do Google AI - RENOMEADAS para evitar conflitos
from google import genai as vertex_genai  # Para Vertex AI
from google.genai import types as vertex_types
import google.generativeai as genai  # Para Gemini API padrão
from google.oauth2 import service_account

# Bibliotecas de ambiente
from dotenv import load_dotenv

# Configuração de ambiente e API
try:
    load_dotenv()
except ImportError:
    print("python-dotenv não encontrado. Usando variáveis de ambiente do sistema.")

# Configuração da API do Gemini
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        print("AVISO: GOOGLE_API_KEY não encontrada.")
except ImportError:
    GEMINI_AVAILABLE = False
    print("google-generativeai não encontrado.")

# Configuração da API do Gemini com Fine-tuning
try:
    # Obter variáveis do ambiente para o modelo com fine-tuning
    PROJECT_ID_TUNED = os.getenv("PROJECT_ID_TUNED")
    MODEL_TUNED = os.getenv("MODEL_TUNED")
    LOCATION_TUNED = os.getenv("LOCATION_TUNED")

    # Verificar se todas as variáveis necessárias para o modelo fine-tuned estão definidas
    if PROJECT_ID_TUNED and MODEL_TUNED and LOCATION_TUNED:
        GEMINI_TUNED_AVAILABLE = True
        #print(f"Configuração de modelo fine-tuned encontrada.")
        #print(f"- Projeto: {PROJECT_ID_TUNED}")
        #print(f"- Localização: {LOCATION_TUNED}")
        #print(f"- Modelo: {MODEL_TUNED.split('/')[-1] if '/' in MODEL_TUNED else MODEL_TUNED}")
    else:
        GEMINI_TUNED_AVAILABLE = False
        missing_vars = []
        if not PROJECT_ID_TUNED: missing_vars.append("PROJECT_ID_TUNED")
        if not MODEL_TUNED: missing_vars.append("MODEL_TUNED")
        if not LOCATION_TUNED: missing_vars.append("LOCATION_TUNED")
        print(f"AVISO: Variáveis de ambiente necessárias não encontradas: {', '.join(missing_vars)}")
        print("O modelo fine-tuned não estará disponível.")
except Exception as e:
    GEMINI_TUNED_AVAILABLE = False
    print(f"Erro ao configurar modelo fine-tuned: {e}")
    print("O modelo fine-tuned não estará disponível.")

# Mapeamentos para formatação de dados
COL_DESCRIPTIONS = {
    'fixed_acidity': 'acidez fixa',
    'volatile_acidity': 'acidez volátil',
    'citric_acid': 'ácido cítrico',
    'residual_sugar': 'açúcar residual',
    'chlorides': 'cloretos',
    'free_sulfur_dioxide': 'dióxido de enxofre livre',
    'total_sulfur_dioxide': 'dióxido de enxofre total',
    'density': 'densidade',
    'ph': 'pH',
    'sulphates': 'sulfatos',
    'alcohol': 'teor alcoólico'
}

UNITS = {
    'acidez fixa': 'g/dm³',
    'acidez volátil': 'g/dm³',
    'ácido cítrico': 'g/dm³',
    'açúcar residual': 'g/dm³',
    'cloretos': 'g/dm³',
    'dióxido de enxofre livre': 'mg/L',
    'dióxido de enxofre total': 'mg/L',
    'densidade': 'g/cm³',
    'pH': '',
    'sulfatos': 'g/dm³',
    'teor alcoólico': '%'
}

CLASS_MAPPING = {
    'baixa': 'low',
    'baixo': 'low',
    'média': 'medium',
    'medio': 'medium',
    'media': 'medium',
    'alta': 'high',
    'alto': 'high'
}

def load_and_prepare(path):
    """Carrega e prepara o dataset de qualidade do vinho."""
    try:
        df = pd.read_csv(path)
        
        # Verificar se temos as colunas esperadas
        expected_columns = ['fixed_acidity', 'residual_sugar', 'alcohol', 'density', 'quality_label']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"AVISO: Colunas esperadas não encontradas: {missing_columns}")
        
        # Garantir que as colunas numéricas são realmente numéricas
        for col in df.columns:
            if col != 'quality_label' and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remover valores faltantes
        if df.isna().any().any():
            df = df.dropna()
        
        return df
        
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None

def split_data_for_finetune(df, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """Divide o dataset em conjuntos de treino, validação e teste."""
    # Verificar se as proporções somam 1.0
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Proporções devem somar 1.0"
    
    # Primeiro, dividir em treino e o restante (para validação + teste)
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state, stratify=df['quality_label'])
    
    # Calcular a proporção do conjunto temporário para validação
    val_ratio = val_size / (val_size + test_size)
    
    # Dividir o conjunto temporário em validação e teste
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=random_state, stratify=temp_df['quality_label'])
    
    print(f"Divisão: Treino={len(train_df)}, Validação={len(val_df)}, Teste={len(test_df)}")
    
    return train_df, val_df, test_df

def generate_prompt(row, feature_cols):
    """Gera um prompt para classificação com base nas características do vinho."""
    feature_descriptions = []
    for col in feature_cols:
        description = COL_DESCRIPTIONS.get(col, col)
        unit = UNITS.get(description, '')
        
        # Formatar o valor com precisão apropriada
        value_str = f"{row[col]:.6f}" if 'densidade' in description.lower() else f"{row[col]:.2f}"
        
        # Adicionar à lista de descrições
        if unit:
            feature_descriptions.append(f"{description} de {value_str} {unit}")
        else:
            feature_descriptions.append(f"{description} de {value_str}")
    
    # Criar prompt
    prompt = "Considere um vinho com " + ", ".join(feature_descriptions[:-1])
    if len(feature_descriptions) > 1:
        prompt += f" e {feature_descriptions[-1]}. "
    else:
        prompt += f". "
    prompt += "Qual é a classe de qualidade deste vinho? Classifique como 'low', 'medium' ou 'high'."
    
    return prompt

def prepare_finetune_data(train_df, output_file="gemini_finetune_data.jsonl", format_type="vertex"):
    """Prepara os dados para fine-tuning do Gemini."""
    print(f"\n=== Preparando dados em formato {format_type} com {len(train_df)} exemplos ===")
    
    # Obter colunas de features
    feature_cols = [col for col in train_df.columns if col != 'quality_label']
    
    # Preparar dados no formato para fine-tuning
    data_for_finetune = []
    
    for _, row in train_df.iterrows():
        # Gerar prompt
        prompt = generate_prompt(row, feature_cols)
        
        # Obter resposta
        target_response = row['quality_label']
        
        # Criar exemplo no formato adequado
        if format_type.lower() == "chatcompletions":
            example = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "model", "content": target_response}
                ]
            }
        elif format_type.lower() == "generatecontent":
            example = {
                "inputText": prompt,
                "outputText": target_response
            }
        else:  # vertex (padrão)
            example = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    },
                    {
                        "role": "model",
                        "parts": [{"text": target_response}]
                    }
                ]
            }
        
        data_for_finetune.append(example)
    
    # Guardar os dados de fine-tuning no formato JSONL
    with open(output_file, "w") as f:
        for example in data_for_finetune:
            f.write(json.dumps(example) + "\n")
    
    print(f"Dados em formato {format_type} salvos em {output_file} ({len(data_for_finetune)} exemplos)")
    
    return output_file

def train_random_forest(train_df, val_df):
    """Treina um modelo Random Forest com os dados de treino e valida."""
    # Separar features e target
    feature_cols = [col for col in train_df.columns if col != 'quality_label']
    
    X_train = train_df[feature_cols]
    y_train = train_df['quality_label']
    
    X_val = val_df[feature_cols]
    y_val = val_df['quality_label']
    
    # Pipeline com normalização
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Treinar o modelo
    rf_pipeline.fit(X_train, y_train)
    
    # Avaliar no conjunto de validação
    y_pred = rf_pipeline.predict(X_val)
    
    # Calcular métricas
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAcurácia do Random Forest (validação): {accuracy:.4f}")
    print(classification_report(y_val, y_pred))
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_val, y_pred)
    
    # Importância das features
    feature_importance = rf_pipeline.named_steps['classifier'].feature_importances_
    
    return {
        'model': rf_pipeline,
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_values': y_val,
        'feature_importance': feature_importance,
        'feature_cols': feature_cols,
        'confusion_matrix': conf_matrix
    }

def evaluate_with_gemini(eval_df, model_id="gemini-2.0-flash-lite", batch_size=29, pause_seconds=60, use_finetune=False, use_few_shot=True):
    """Avalia o modelo Gemini no conjunto de dados fornecido."""
    
    # Permitir escolha do modelo Gemini
    gemini_models = {
        "1": {"name": "gemini-2.0-flash", "display": "Gemini 2.0 Flash (base)"},
        "2": {"name": "gemini-2.0-flash-lite", "display": "Gemini 2.0 Flash Lite"},
        "3": {"name": MODEL_TUNED, 
              "display": "Modelo Fine-tunado (Wine Quality)", 
              "is_finetuned": True,
              "project_id": PROJECT_ID_TUNED,
              "location": LOCATION_TUNED}
    }
    
    # Mostrar opções de modelos e solicitar escolha
    print("\nModelos Gemini disponíveis:")
    for k, model_info in gemini_models.items():
        print(f"{k}. {model_info['display']}")
    
    model_choice = input("\nEscolha um modelo (1-3) ou pressione Enter para usar o padrão: ").strip()
    
    # Configurar com base na escolha
    if model_choice in gemini_models:
        model_info = gemini_models[model_choice]
        model_id = model_info["name"]
        use_finetune = model_info.get("is_finetuned", False)
        print(f"Usando modelo: {model_info['display']}")
    else:
        print(f"Usando modelo padrão: {model_id}")
    
    # Definir abordagem
    approach = "fine-tuned" if use_finetune else ("few-shot" if use_few_shot else "zero-shot")
    print(f"\n=== Avaliando com Gemini ({approach}): {model_id} ===")
    
    # Obter colunas de features
    feature_cols = [col for col in eval_df.columns if col != 'quality_label']
    
    # Preparar prompts para avaliação
    prompts = []
    true_labels = []
    
    for _, row in eval_df.iterrows():
        # Criar prompt
        prompt = generate_prompt(row, feature_cols)
        prompts.append(prompt)
        true_labels.append(row['quality_label'])
    
    # Configurar o modelo
    try:
        if use_finetune:
            print("Configurando cliente Vertex AI para modelo fine-tuned...")
            try:
                # Caminho para o arquivo de credenciais
                credentials_path = "gen-lang-cred.json"
                
                # Verificar se o arquivo existe
                if not os.path.exists(credentials_path):
                    print(f"AVISO: Arquivo de credenciais '{credentials_path}' não encontrado.")
                    raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {credentials_path}")
                
                # Carregar as credenciais do arquivo JSON
                print(f"Carregando credenciais do arquivo: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                
                # AQUI: Usar as credenciais ao criar o cliente Vertex AI
                print("Criando cliente Vertex AI com credenciais de serviço...")
                client = vertex_genai.Client(
                    vertexai=True,
                    project=PROJECT_ID_TUNED,  # Usar o project_id do modelo fine-tuned
                    location=LOCATION_TUNED,     # Usar a localização do modelo fine-tuned
                    credentials=credentials   # Passar as credenciais carregadas do arquivo JSON
                )
                print("Cliente Vertex AI configurado com sucesso!")
                
                # Definir configuração para o modelo fine-tuned
                generate_content_config = vertex_types.GenerateContentConfig(
                    temperature=0.2,  # Temperatura baixa para resultados mais determinísticos
                    top_p=0.95,
                    max_output_tokens=8192,
                    response_modalities=["TEXT"],
                    safety_settings=[
                        vertex_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                        vertex_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        vertex_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                        vertex_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                    ],
                )
                
                # Definir função para gerar resposta usando o modelo fine-tuned
                def generate_with_vertex(prompt_text):
                    contents = [
                        vertex_types.Content(
                            role="user",
                            parts=[vertex_types.Part(text=prompt_text)]
                        )
                    ]
                    
                    response_text = ""
                    for chunk in client.models.generate_content_stream(
                        model=model_id,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if hasattr(chunk, "text"):
                            response_text += chunk.text
                    
                    return response_text
                
                # Testar o modelo
                print("\nRealizando teste com o modelo fine-tuned via Vertex AI...")
                test_prompt = "Considere um vinho com acidez fixa de 7.4 g/dm³, açúcar residual de 1.9 g/dm³, teor alcoólico de 9.4% e densidade de 0.9978 g/cm³. Qual é a classe de qualidade deste vinho? Classifique como 'low', 'medium' ou 'high'."
                test_response = generate_with_vertex(test_prompt)
                print(f"Prompt de teste: {test_prompt}")
                print(f"Resposta de teste: {test_response}")
                
                # Se chegou aqui, a configuração foi bem-sucedida
                print("Teste bem-sucedido! Continuando com a avaliação...")
                
                # Modelo configurado com sucesso
                vertex_client = client
                use_vertex = True
                
            except Exception as setup_error:
                print(f"Erro ao configurar cliente Vertex AI: {setup_error}")
                print("Voltando ao modelo padrão Gemini...")
                use_finetune = False
                model_id = "gemini-2.0-flash-lite"
                
                # Configurar modelo padrão Gemini
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                
                gemini_model = genai.GenerativeModel(model_id)
                temperature = 0.7 if use_few_shot else 0.4
                gemini_config = genai.GenerationConfig(
                    temperature=temperature,
                    top_p=0.95,
                    top_k=40
                )
                use_vertex = False
        else:
            # Para modelos padrão Gemini
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            
            gemini_model = genai.GenerativeModel(model_id)
            temperature = 0.7 if use_few_shot else 0.4
            gemini_config = genai.GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40
            )
            use_vertex = False
            
    except Exception as e:
        print(f"Erro ao configurar modelo: {e}")
        return None
    
    # Exemplos few-shot (se necessário)
    few_shot_examples = """
Aqui estão alguns exemplos de classificação de vinho:

Exemplo 1: Vinho com acidez fixa de 7.5 g/dm³, açúcar residual de 1.8 g/dm³, teor alcoólico de 9.5% e densidade de 0.9978 g/cm³.
Resposta: low

Exemplo 2: Vinho com acidez fixa de 6.9 g/dm³, açúcar residual de 3.2 g/dm³, teor alcoólico de 11.8% e densidade de 0.9986 g/cm³.
Resposta: medium

Exemplo 3: Vinho com acidez fixa de 7.2 g/dm³, açúcar residual de 2.5 g/dm³, teor alcoólico de 13.5% e densidade de 0.9940 g/cm³.
Resposta: high

Agora classifique o seguinte vinho:
""" if use_few_shot and not use_finetune else ""
    
    # Estatísticas para acompanhamento
    prediction_counts = {'low': 0, 'medium': 0, 'high': 0, 'outros': 0}
    preds = []
    
    # Processar em batches
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        print(f"\nBatch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1} ({batch_start+1}-{batch_end} de {len(prompts)})")
        
        batch_preds = []
        for i, prompt in enumerate(tqdm(batch_prompts)):
            try:
                # Construir prompt completo
                if use_finetune:
                    # Para modelo fine-tunado, prompt direto
                    full_prompt = prompt
                else:
                    # Para modelo base, adicionar few-shot e instruções
                    full_prompt = (
                        f"{few_shot_examples}\n\n{prompt}\n\n"
                        "IMPORTANTE: Responda APENAS com 'low', 'medium' ou 'high'."
                    )
                
                # Chamar API
                try:
                    if use_finetune and use_vertex:
                        # Usar Vertex AI para modelo fine-tuned
                        resp_text = generate_with_vertex(full_prompt)
                    else:
                        # Usar API padrão Gemini
                        response = gemini_model.generate_content(full_prompt, generation_config=gemini_config)
                        resp_text = response.text
                    
                    resp_text = resp_text.strip().lower()
                    
                    # Mostrar detalhes apenas para o primeiro exemplo de cada batch
                    if i == 0:
                        print(f"\nExemplo de prompt: {full_prompt[:100]}...")
                        print(f"Exemplo de resposta: {resp_text}")
                        
                except Exception as api_error:
                    print(f"\nErro na chamada da API (exemplo {batch_start+i+1}): {api_error}")
                    raise api_error
                
                # Extrair classificação
                english_match = re.search(r'\b(low|medium|high)\b', resp_text)
                portuguese_match = re.search(r'\b(baixa|baixo|média|medio|media|alta|alto)\b', resp_text)
                
                prediction = None
                if english_match:
                    prediction = english_match.group(1)
                    prediction_counts[prediction] += 1
                elif portuguese_match:
                    pt_response = portuguese_match.group(1)
                    prediction = CLASS_MAPPING.get(pt_response)
                    prediction_counts[prediction] += 1
                else:
                    prediction_counts['outros'] += 1
                    print(f"\nResposta não reconhecida: '{resp_text}'")
                
                batch_preds.append(prediction)
                time.sleep(0.2)  # Pequena pausa entre requisições
                
            except Exception as e:
                print(f"\nErro ao processar exemplo {batch_start+i+1}: {e}")
                batch_preds.append(None)
                prediction_counts['outros'] += 1
        
        # Adicionar resultados do batch às predições totais
        preds.extend(batch_preds)
        
        # Exibir estatísticas parciais
        print("\n\nEstatísticas parciais:")
        print(f"- Predições 'low': {prediction_counts['low']}")
        print(f"- Predições 'medium': {prediction_counts['medium']}")
        print(f"- Predições 'high': {prediction_counts['high']}")
        print(f"- Não reconhecidas/erros: {prediction_counts['outros']}")
        
        # Pausa entre batches se necessário
        if batch_end < len(prompts):
            remaining = len(prompts) - batch_end
            print(f"\nAguardando {pause_seconds}s antes do próximo batch ({remaining} exemplos restantes)...")
            time.sleep(pause_seconds)
    
    # Remover exemplos com predições nulas
    valid_results = [(true, pred) for true, pred in zip(true_labels, preds) if pred is not None]
    
    if not valid_results:
        print("Nenhuma predição válida foi obtida.")
        return None
    
    valid_true, valid_preds = zip(*valid_results)
    
    # Calcular matriz de confusão
    conf_matrix = pd.crosstab(
        pd.Series(valid_true, name='Real'),
        pd.Series(valid_preds, name='Predito')
    )
    
    # Calcular acurácia
    accuracy = accuracy_score(valid_true, valid_preds)
    print(f"\n=== Resultados Finais ===")
    print(f"Total de exemplos processados: {len(valid_results)} de {len(prompts)}")
    print(f"Acurácia do Gemini ({approach}): {accuracy:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    print("\nRelatório de Classificação:")
    print(classification_report(valid_true, valid_preds, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'predictions': valid_preds,
        'true_values': valid_true,
        'model': model_id,
        'approach': approach,
        'confusion_matrix': conf_matrix
    }

def test_final_models(test_df, rf_results, gemini_base_results=None, gemini_finetune_results=None):
    """Avalia os modelos finais no conjunto de teste."""
    print("\n=== Avaliação Final no Conjunto de Teste ===")
    
    # Separar features e target
    feature_cols = rf_results['feature_cols']
    X_test = test_df[feature_cols]
    y_test = test_df['quality_label']
    
    # Avaliar Random Forest
    print("\nAvaliando Random Forest no conjunto de teste...")
    rf_pipeline = rf_results['model']
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_test_accuracy = accuracy_score(y_test, rf_test_pred)
    
    print(f"Acurácia do Random Forest (teste): {rf_test_accuracy:.4f}")
    print("\n=== Random Forest Classification Report (teste) ===")
    rf_report = classification_report(y_test, rf_test_pred)
    print(rf_report)
    
    rf_test_cm = confusion_matrix(y_test, rf_test_pred)
    print("\n=== Matriz de Confusão - Random Forest (teste) ===")
    print(rf_test_cm)
    
    # Avaliar Gemini Base (em comparação com os mesmos exemplos de teste do RF)
    gemini_base_test_metrics = None
    if gemini_base_results:
        print("\nAnalisando resultados do Gemini Base no conjunto de teste...")
        gemini_base_test_metrics = {
            'accuracy': gemini_base_results['accuracy'],
            'confusion_matrix': gemini_base_results['confusion_matrix'],
            'approach': gemini_base_results['approach'],
            'model': gemini_base_results.get('model', 'gemini-pro')
        }
        print(f"Acurácia do Gemini ({gemini_base_test_metrics['approach']}): {gemini_base_test_metrics['accuracy']:.4f}")
    
    # Avaliar Gemini Fine-tunado (se disponível)
    gemini_finetune_test_metrics = None
    if gemini_finetune_results:
        print("\nAnalisando resultados do Gemini Fine-tunado no conjunto de teste...")
        gemini_finetune_test_metrics = {
            'accuracy': gemini_finetune_results['accuracy'],
            'confusion_matrix': gemini_finetune_results['confusion_matrix'],
            'approach': gemini_finetune_results['approach'],
            'model': gemini_finetune_results.get('model', 'gemini-fine-tuned')
        }
        print(f"Acurácia do Gemini ({gemini_finetune_test_metrics['approach']}): {gemini_finetune_test_metrics['accuracy']:.4f}")
    
    # Obter data e hora atual para registro
    test_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Registrar informações da distribuição das classes no teste
    class_distribution = test_df['quality_label'].value_counts().to_dict()
    
    # Obter importância das features do Random Forest
    feature_importance = rf_results['feature_importance'].tolist() if hasattr(rf_results['feature_importance'], 'tolist') else rf_results['feature_importance']
    
    # Adicionar detalhes sobre os modelos e dataset ao resultado
    result_info = {
        'dataset_size': len(test_df),
        'class_distribution': class_distribution,
        'test_timestamp': test_timestamp,
        'rf': {
            'accuracy': rf_test_accuracy,
            'predictions': rf_test_pred.tolist() if hasattr(rf_test_pred, 'tolist') else list(rf_test_pred),
            'confusion_matrix': rf_test_cm.tolist() if hasattr(rf_test_cm, 'tolist') else rf_test_cm,
            'report': rf_report,
            'feature_cols': feature_cols,
            'feature_importance': feature_importance
        },
        'gemini_base': gemini_base_test_metrics,
        'gemini_finetune': gemini_finetune_test_metrics
    }
    
    # Visualizar comparativo dos resultados
    print("\n=== COMPARATIVO DE MODELOS ===")
    print(f"Dataset de teste: {len(test_df)} exemplos")
    print(f"Random Forest: {rf_test_accuracy:.4f}")
    
    if gemini_base_test_metrics:
        approach = gemini_base_test_metrics['approach']
        acc = gemini_base_test_metrics['accuracy']
        print(f"Gemini ({approach}): {acc:.4f}")
        
        # Calcular a diferença em relação ao Random Forest
        diff = acc - rf_test_accuracy
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"  Diferença vs. RF: {diff_str}")
    
    if gemini_finetune_test_metrics:
        approach = gemini_finetune_test_metrics['approach']
        acc = gemini_finetune_test_metrics['accuracy']
        print(f"Gemini Fine-tuned ({approach}): {acc:.4f}")
        
        # Calcular a diferença em relação ao Random Forest
        diff = acc - rf_test_accuracy
        diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        print(f"  Diferença vs. RF: {diff_str}")
    
    # Identificar o melhor modelo
    best_acc = rf_test_accuracy
    best_model = "Random Forest"
    
    if gemini_base_test_metrics and gemini_base_test_metrics['accuracy'] > best_acc:
        best_acc = gemini_base_test_metrics['accuracy']
        best_model = f"Gemini ({gemini_base_test_metrics['approach']})"
    
    if gemini_finetune_test_metrics and gemini_finetune_test_metrics['accuracy'] > best_acc:
        best_acc = gemini_finetune_test_metrics['accuracy']
        best_model = f"Gemini Fine-tuned ({gemini_finetune_test_metrics['approach']})"
    
    result_info['best_model'] = {
        'name': best_model,
        'accuracy': best_acc
    }
    
    print(f"\nMelhor modelo: {best_model} (Acurácia: {best_acc:.4f})")
    
    return result_info

def visualize_results(rf_results, gemini_base_results=None, gemini_finetune_results=None, final_test_results=None):
    """Gera visualizações comparativas dos resultados dos modelos."""
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Importância das features
    plt.figure(figsize=(10, 6))
    feature_importance_df = pd.DataFrame({
        'Feature': rf_results['feature_cols'],
        'Importance': rf_results['feature_importance']
    }).sort_values(by='Importance', ascending=False)
    
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#2C7BB6')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.title('Importância das Features no Random Forest')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    
    # 2. Comparação de acurácia
    if gemini_base_results or gemini_finetune_results:
        # Colher dados
        acuracias = [rf_results['accuracy']]
        modelos = ['Random Forest']
        cores = ['#2C7BB6']  # Azul
        
        if gemini_base_results:
            acuracias.append(gemini_base_results['accuracy'])
            modelos.append(f'Gemini ({gemini_base_results["approach"]})')
            cores.append('#D7191C')  # Vermelho
        
        if gemini_finetune_results:
            acuracias.append(gemini_finetune_results['accuracy'])
            modelos.append(f'Gemini ({gemini_finetune_results["approach"]})')
            cores.append('#33A02C')  # Verde
        
        # Criar gráfico
        plt.figure(figsize=(10, 6))
        bars = plt.bar(modelos, acuracias, color=cores)
        plt.ylim(0, 1.0)
        plt.ylabel('Acurácia')
        plt.title('Comparação de Acurácia entre Modelos')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3. Matriz de confusão para o Random Forest
    plt.figure(figsize=(8, 6))
    cm = rf_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(set(rf_results['true_values'])),
                yticklabels=sorted(set(rf_results['true_values'])))
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão - Random Forest')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png')

def predict_user_input(rf_model, feature_cols):
    """
    Permite ao usuário inserir valores para predição manual.
    
    Parâmetros:
    - rf_model: Modelo RandomForest treinado
    - feature_cols: Lista de colunas de features usadas no treinamento
    
    Retorna:
    - Nada, apenas imprime os resultados
    """

    print("\n=== Teste de Predição Manual ===")
    
    # Definir apenas os 4 parâmetros principais para input
    main_features = ['fixed_acidity', 'residual_sugar', 'alcohol', 'density']
    
    # Verificar quais das features principais estão disponíveis no modelo
    available_main_features = [f for f in main_features if f in feature_cols]
    
    if len(available_main_features) < len(main_features):
        print(f"AVISO: Algumas features principais não estão disponíveis no modelo.")
        print(f"Features disponíveis: {available_main_features}")
    
    # Gemini Models Options
    gemini_models = {
        "1": {"name": "gemini-2.0-flash", "display": "Gemini 2.0 Flash (base)"},
        "2": {"name": "gemini-2.0-flash-lite", "display": "Gemini 2.0 Flash Lite"},
        "3": {"name": MODEL_TUNED, 
              "display": "Modelo Fine-tunado (Wine Quality)", 
              "is_finetuned": True,
              "project_id": PROJECT_ID_TUNED,
              "location": LOCATION_TUNED}
    }
    
    while True:
        print("\nOpções de entrada:")
        print("1. Inserir valores manualmente")
        print("2. Usar valores de exemplo (vinho de qualidade baixa)")
        print("3. Usar valores de exemplo (vinho de qualidade média)")
        print("4. Usar valores de exemplo (vinho de qualidade alta)")
        print("0. Voltar ao menu principal")
        
        input_choice = input("\nEscolha uma opção (0-4): ").strip()
        
        if input_choice == "0":
            return
        
        # Dicionário para armazenar os valores
        user_values = {}
        
        if input_choice == "1":
            # Entrada manual apenas para os 4 parâmetros principais
            print("\nInsira os valores para as características principais do vinho:")
            
            # Solicitar valores para cada feature principal
            for col in available_main_features:
                description = COL_DESCRIPTIONS.get(col, col)
                unit = UNITS.get(description, '')
                unit_text = f" ({unit})" if unit else ""
                
                while True:
                    try:
                        value_str = input(f"{description}{unit_text}: ")
                        value = float(value_str.replace(',', '.'))  # Aceitar tanto vírgula quanto ponto decimal
                        user_values[col] = value
                        break
                    except ValueError:
                        print("Valor inválido. Por favor, insira um número válido.")
        
        elif input_choice == "2":
            # Exemplo de vinho de qualidade baixa (apenas parameters principais)
            user_values = {
                'fixed_acidity': 7.4,
                'residual_sugar': 1.9,
                'alcohol': 9.4,
                'density': 0.9978
            }
            print("\nUsando valores de exemplo (vinho de qualidade baixa).")
        
        elif input_choice == "3":
            # Exemplo de vinho de qualidade média (apenas parameters principais)
            user_values = {
                'fixed_acidity': 7.0,
                'residual_sugar': 20.7,
                'alcohol': 8.8,
                'density': 1.001
            }
            print("\nUsando valores de exemplo (vinho de qualidade média).")
        
        elif input_choice == "4":
            # Exemplo de vinho de qualidade alta (apenas parameters principais)
            user_values = {
                'fixed_acidity': 8.0,
                'residual_sugar': 2.0,
                'alcohol': 12.8,
                'density': 0.9952
            }
            print("\nUsando valores de exemplo (vinho de qualidade alta).")
        
        else:
            print("Opção inválida. Por favor escolha uma opção entre 0 e 4.")
            continue
        
        # Preencher os valores faltantes com médias ou valores padrão
        # Para features principais não disponíveis
        for col in main_features:
            if col not in available_main_features and col not in user_values:
                user_values[col] = 0.0
                print(f"AVISO: Feature principal '{col}' não disponível no modelo. Usando valor padrão 0.0")
        
        # Para outras features do modelo
        default_values = {
            'volatile_acidity': 0.35,
            'citric_acid': 0.3,
            'chlorides': 0.05,
            'free_sulfur_dioxide': 30.0,
            'total_sulfur_dioxide': 100.0,
            'ph': 3.3,
            'sulphates': 0.5
        }
        
        for col in feature_cols:
            if col not in user_values:
                user_values[col] = default_values.get(col, 0.0)
                print(f"Feature '{col}' preenchida automaticamente com valor: {user_values[col]}")
                
        # Mostrar os valores que serão usados para predição
        print("\nValores usados para predição:")
        # Primeiro mostrar as features principais
        for col in main_features:
            if col in user_values:
                description = COL_DESCRIPTIONS.get(col, col)
                unit = UNITS.get(description, '')
                unit_text = f" {unit}" if unit else ""
                print(f"- {description}: {user_values[col]}{unit_text}")
        
        # Depois mostrar as outras features (se houver)
        other_features = [col for col in feature_cols if col not in main_features and col in user_values]
        if other_features:
            print("\nOutras características (preenchidas automaticamente):")
            for col in other_features:
                description = COL_DESCRIPTIONS.get(col, col)
                unit = UNITS.get(description, '')
                unit_text = f" {unit}" if unit else ""
                print(f"- {description}: {user_values[col]}{unit_text}")
        
        # Criar um DataFrame com os valores
        user_df = pd.DataFrame([user_values])
        
        # Realizar a predição com o RandomForest
        rf_pred = rf_model.predict(user_df)[0]
        rf_proba = rf_model.predict_proba(user_df)[0]
        
        # Obter as classes do modelo
        classes = rf_model.named_steps['classifier'].classes_
        
        # Exibir resultados do RandomForest
        print("\nResultados da Predição:")
        print(f"Random Forest prediz: {rf_pred}")
        print("Probabilidades por classe:")
        for cls, prob in zip(classes, rf_proba):
            print(f"  - {cls}: {prob:.4f} ({prob*100:.1f}%)")
        
        # Realizar predição com Gemini, se disponível
        if GEMINI_AVAILABLE:
            try:
                # Primeiro perguntar qual modelo do Gemini usar
                print("\nModelos Gemini disponíveis:")
                for k, model_info in gemini_models.items():
                    print(f"{k}. {model_info['display']}")
                
                model_choice = input("\nEscolha um modelo (1-3) ou pressione Enter para usar Gemini 2.0 Flash Lite: ").strip()
                
                # Configuração do modelo escolhido
                if model_choice in gemini_models:
                    model_info = gemini_models[model_choice]
                else:
                    model_info = gemini_models["1"]  # Default
                
                model_id = model_info["name"]
                is_finetuned = model_info.get("is_finetuned", False)
                
                print(f"\nUsando modelo: {model_info['display']}")
                
                # Gerar prompt a partir dos dados do usuário
                prompt = generate_prompt(user_values, feature_cols)
                
                print("\nConsultando Gemini para predição...")
                
                if is_finetuned:
                    print("Configurando cliente Vertex AI para modelo fine-tuned...")

                    try:
                        # Caminho para o arquivo de credenciais
                        credentials_path = "gen-lang-cred.json"
                        
                        # Verificar se o arquivo existe
                        if not os.path.exists(credentials_path):
                            print(f"AVISO: Arquivo de credenciais '{credentials_path}' não encontrado.")
                            raise FileNotFoundError(f"Arquivo de credenciais não encontrado: {credentials_path}")

                        # Carregar as credenciais do arquivo JSON
                        print(f"Carregando credenciais do arquivo: {credentials_path}")
                        credentials = service_account.Credentials.from_service_account_file(
                            credentials_path,
                            scopes=["https://www.googleapis.com/auth/cloud-platform"]
                        )
                        
                        # Criar cliente Vertex AI com credenciais
                        print("Criando cliente Vertex AI com credenciais de serviço...")

                        client = vertex_genai.Client(
                            vertexai=True,
                            project=model_info.get("project_id", PROJECT_ID_TUNED),
                            location=model_info.get("location", LOCATION_TUNED),
                            credentials=credentials
                        )
                        
                        # Configurar geração de conteúdo
                        generate_content_config = vertex_types.GenerateContentConfig(
                            temperature=0.2,
                            top_p=0.95,
                            max_output_tokens=8192,
                            response_modalities=["TEXT"],
                            safety_settings=[
                                vertex_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                                vertex_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                                vertex_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                                vertex_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
                            ],
                        )
                        
                        # Preparar conteúdo do prompt
                        contents = [
                            vertex_types.Content(
                                role="user",
                                parts=[vertex_types.Part(text=prompt)]
                            )
                        ]
                        
                        print("Enviando prompt para modelo fine-tuned via Vertex AI...")
                        
                        # Gerar conteúdo com streaming
                        response_text = ""
                        for chunk in client.models.generate_content_stream(
                            model=model_id,
                            contents=contents,
                            config=generate_content_config,
                        ):
                            if hasattr(chunk, "text"):
                                response_text += chunk.text
                        
                        resp_text = response_text.strip().lower()
                        print(f"Resposta recebida: {resp_text}")
                    
                    except Exception as e:
                        print(f"Erro ao acessar modelo fine-tuned: {e}")
                        print("Tentando usar modelo padrão Gemini...")
                        
                        # Fallback para modelo padrão
                        api_key = os.getenv("GOOGLE_API_KEY")
                        if api_key:
                            genai.configure(api_key=api_key)
                        
                        model_id = "gemini-2.0-flash-lite"
                        gemini_model = genai.GenerativeModel(model_id)
                        gemini_config = genai.GenerationConfig(
                            temperature=0.4,
                            top_p=0.95,
                            top_k=40
                        )
                        
                        full_prompt = f"{prompt}\n\nIMPORTANTE: Responda APENAS com 'low', 'medium' ou 'high'."
                        response = gemini_model.generate_content(full_prompt, generation_config=gemini_config)
                        resp_text = response.text.strip().lower()
                else:
                    # Usar API padrão Gemini para modelos não fine-tuned
                    print("Utilizando modelo Gemini padrão...")

                    api_key = os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        genai.configure(api_key=api_key)
                    
                    gemini_model = genai.GenerativeModel(model_id)
                    gemini_config = genai.GenerationConfig(
                        temperature=0.4,
                        top_p=0.95,
                        top_k=40
                    )
                    
                    # Adicionar exemplos few-shot para melhorar a predição
                    few_shot_examples = """
Aqui estão alguns exemplos de classificação de vinho:

Exemplo 1: Vinho com acidez fixa de 7.5 g/dm³, açúcar residual de 1.8 g/dm³, teor alcoólico de 9.5% e densidade de 0.9978 g/cm³.
Resposta: low

Exemplo 2: Vinho com acidez fixa de 6.9 g/dm³, açúcar residual de 3.2 g/dm³, teor alcoólico de 11.8% e densidade de 0.9986 g/cm³.
Resposta: medium

Exemplo 3: Vinho com acidez fixa de 7.2 g/dm³, açúcar residual de 2.5 g/dm³, teor alcoólico de 13.5% e densidade de 0.9940 g/cm³.
Resposta: high

Agora classifique o seguinte vinho:
"""
                    
                    full_prompt = f"{few_shot_examples}\n\n{prompt}\n\nIMPORTANTE: Responda APENAS com 'low', 'medium' ou 'high'."
                    
                    response = gemini_model.generate_content(full_prompt, generation_config=gemini_config)
                    resp_text = response.text.strip().lower()
                    print(f"Resposta recebida: {resp_text}")
                
                # Extrair classificação
                english_match = re.search(r'\b(low|medium|high)\b', resp_text)
                portuguese_match = re.search(r'\b(baixa|baixo|média|medio|media|alta|alto)\b', resp_text)
                
                gemini_prediction = None
                if english_match:
                    gemini_prediction = english_match.group(1)
                elif portuguese_match:
                    pt_response = portuguese_match.group(1)
                    gemini_prediction = CLASS_MAPPING.get(pt_response)
                    
                if gemini_prediction:
                    print(f"Gemini prediz: {gemini_prediction}")
                else:
                    print(f"Gemini não conseguiu classificar. Resposta: {resp_text}")
                    
            except Exception as e:
                print(f"Erro ao consultar Gemini: {e}")
                print("Detalhes do erro:", str(e))
        else:
            print("\nPredição com Gemini não disponível. Configure a GOOGLE_API_KEY para utilizar.")
        
        # Perguntar se deseja fazer outra predição
        continue_choice = input("\nDeseja fazer outra predição? (s/n): ").lower()
        if not continue_choice.startswith('s'):
            break

def save_results(results_dict, filename="model_results.json"):
    """
    Salva os resultados dos modelos em um arquivo JSON.
    
    Parâmetros:
    - results_dict: Dicionário com os resultados dos modelos
    - filename: Nome do arquivo para salvar
    
    Retorna:
    - Caminho do arquivo salvo
    """
    
    # Adicionar timestamp nos resultados
    results_with_timestamp = results_dict.copy()
    results_with_timestamp['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Converter tipos não serializáveis (numpy, etc.) para tipos Python padrão
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        elif hasattr(obj, 'tolist'):  # Para arrays numpy
            return obj.tolist()
        elif hasattr(obj, 'to_dict'):  # Para DataFrames do pandas
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif str(type(obj)) == "<class 'pandas.core.indexes.base.Index'>":
            return list(obj)
        else:
            return obj
    
    # Preparar dados serializáveis
    serializable_results = convert_to_serializable(results_with_timestamp)
    
    # Garantir que o diretório 'results' existe
    os.makedirs('results', exist_ok=True)
    
    # Nome do arquivo com timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not filename.endswith('.json'):
        filename += '.json'
    filename_with_timestamp = f"results/{timestamp_str}_{filename}"
    
    # Salvar resultados
    with open(filename_with_timestamp, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Resultados salvos em: {filename_with_timestamp}")
    return filename_with_timestamp

def generate_model_comparisons(results_files=None, all_results=None):
    """
    Gera visualizações comparativas entre diferentes modelos testados.
    
    Parâmetros:
    - results_files: Lista de caminhos para arquivos de resultados salvos
    - all_results: Dicionário com todos os resultados já carregados
    
    Retorna:
    - Caminhos dos gráficos gerados
    """
    
    # Criar diretório para gráficos se não existir
    os.makedirs('comparisons', exist_ok=True)
    
    # Preparar os dados dos resultados
    model_results = []
    
    # Carregar resultados de arquivos se fornecidos
    if results_files:
        for file_path in results_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                model_results.append(results)
            except Exception as e:
                print(f"Erro ao carregar arquivo {file_path}: {e}")
    
    # Adicionar resultados já carregados
    if all_results:
        if isinstance(all_results, list):
            model_results.extend(all_results)
        else:
            model_results.append(all_results)
    
    if not model_results:
        print("Nenhum resultado disponível para gerar comparações.")
        return []
    
    # Organizar dados para visualização
    comparison_data = []
    
    for result in model_results:
        # Extrair informações de cada modelo no resultado
        if 'rf' in result:
            # Random Forest
            comparison_data.append({
                'model': 'Random Forest',
                'accuracy': result['rf']['accuracy'],
                'date': result.get('timestamp', 'N/A')
            })
        
        if 'gemini_base' in result and result['gemini_base']:
            # Gemini Base
            comparison_data.append({
                'model': f"Gemini ({result['gemini_base']['approach']})",
                'accuracy': result['gemini_base']['accuracy'],
                'date': result.get('timestamp', 'N/A')
            })
        
        if 'gemini_finetune' in result and result['gemini_finetune']:
            # Gemini Fine-tuned
            comparison_data.append({
                'model': 'Gemini Fine-tuned',
                'accuracy': result['gemini_finetune']['accuracy'],
                'date': result.get('timestamp', 'N/A')
            })
    
    # Criar DataFrame com os dados comparativos
    df_comparison = pd.DataFrame(comparison_data)
    
    # 1. Gráfico de barras comparando acurácia entre modelos
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='accuracy', data=df_comparison, palette='viridis')
    plt.title('Comparação de Acurácia entre Modelos', fontsize=16)
    plt.xlabel('Modelo', fontsize=14)
    plt.ylabel('Acurácia', fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar os valores de acurácia no topo de cada barra
    for i, v in enumerate(df_comparison['accuracy']):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12)
    
    plt.tight_layout()
    accuracy_chart_path = 'comparisons/model_accuracy_comparison.png'
    plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tabela resumo em formato de figura
    fig, ax = plt.subplots(figsize=(10, len(df_comparison) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar dados para a tabela
    table_data = []
    for _, row in df_comparison.iterrows():
        table_data.append([
            row['model'],
            f"{row['accuracy']:.4f}",
            row['date']
        ])
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Modelo', 'Acurácia', 'Data'],
                    loc='center',
                    cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Resumo de Resultados', fontsize=16, pad=20)
    plt.tight_layout()
    
    table_path = 'comparisons/results_summary_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Exibir resultados
    print(f"Gerados arquivos de comparação:")
    print(f"- Gráfico de acurácia: {accuracy_chart_path}")
    print(f"- Tabela resumo: {table_path}")
    
    return [accuracy_chart_path, table_path]

def generate_model_comparisons(results_files=None, all_results=None):
    """
    Gera visualizações comparativas entre diferentes modelos testados.
    
    Parâmetros:
    - results_files: Lista de caminhos para arquivos de resultados salvos
    - all_results: Dicionário com todos os resultados já carregados
    
    Retorna:
    - Caminhos dos gráficos gerados
    """
    
    # Criar diretório para gráficos se não existir
    os.makedirs('comparisons', exist_ok=True)
    
    # Definir estilo dos gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("viridis")
    
    # Lista para armazenar caminhos dos gráficos gerados
    generated_charts = []
    
    # Preparar os dados dos resultados
    model_results = []
    
    # Carregar resultados de arquivos se fornecidos
    if results_files:
        for file_path in results_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                model_results.append(results)
            except Exception as e:
                print(f"Erro ao carregar arquivo {file_path}: {e}")
    
    # Adicionar resultados já carregados
    if all_results:
        if isinstance(all_results, list):
            model_results.extend(all_results)
        else:
            model_results.append(all_results)
    
    if not model_results:
        print("Nenhum resultado disponível para gerar comparações.")
        return []
    
    # Organizar dados para visualização
    comparison_data = []
    
    for result in model_results:
        test_timestamp = result.get('test_timestamp', 'N/A')
        dataset_size = result.get('dataset_size', 'N/A')
        
        # Random Forest
        if 'rf' in result:
            comparison_data.append({
                'model': 'Random Forest',
                'accuracy': result['rf']['accuracy'],
                'timestamp': test_timestamp,
                'dataset_size': dataset_size,
                'type': 'ML Tradicional'
            })
        
        # Gemini Base
        if 'gemini_base' in result and result['gemini_base']:
            approach = result['gemini_base']['approach']
            comparison_data.append({
                'model': f"Gemini ({approach})",
                'accuracy': result['gemini_base']['accuracy'],
                'timestamp': test_timestamp,
                'dataset_size': dataset_size,
                'type': 'LLM'
            })
        
        # Gemini Fine-tuned
        if 'gemini_finetune' in result and result['gemini_finetune']:
            approach = result['gemini_finetune']['approach']
            comparison_data.append({
                'model': f"Gemini Fine-tuned ({approach})",
                'accuracy': result['gemini_finetune']['accuracy'],
                'timestamp': test_timestamp,
                'dataset_size': dataset_size,
                'type': 'LLM Fine-tuned'
            })
    
    # Criar DataFrame com os dados comparativos
    df_comparison = pd.DataFrame(comparison_data)
    
    if len(df_comparison) == 0:
        print("Não há dados suficientes para gerar comparações.")
        return []
    
    print(f"Gerando comparativos para {len(df_comparison)} modelos...")
    
    # 1. Gráfico de barras comparando acurácia entre modelos
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='model', y='accuracy', data=df_comparison, hue='type', palette='viridis')
    plt.title('Comparação de Acurácia entre Modelos', fontsize=16)
    plt.xlabel('Modelo', fontsize=14)
    plt.ylabel('Acurácia', fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Tipo de Modelo')
    
    # Adicionar os valores de acurácia no topo de cada barra
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.4f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', fontsize=10, color='black', 
                   xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    accuracy_chart_path = 'comparisons/model_accuracy_comparison.png'
    plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_charts.append(accuracy_chart_path)
    
    # 2. Gráfico de barras horizontais para destacar a diferença entre RF e outros modelos
    if len(df_comparison) > 1:
        # Obter acurácia do Random Forest
        rf_acc = df_comparison[df_comparison['model'] == 'Random Forest']['accuracy'].values
        
        if len(rf_acc) > 0:
            rf_acc = rf_acc[0]
            
            # Calcular diferença em relação ao Random Forest
            df_comparison['diff_from_rf'] = df_comparison['accuracy'] - rf_acc
            df_comparison['color'] = df_comparison['diff_from_rf'].apply(
                lambda x: 'green' if x > 0 else 'red' if x < 0 else 'gray'
            )
            
            # Filtrar apenas modelos que não são Random Forest
            df_other = df_comparison[df_comparison['model'] != 'Random Forest'].copy()
            
            if len(df_other) > 0:
                plt.figure(figsize=(10, 6))
                
                # Ordenar por diferença
                df_other = df_other.sort_values('diff_from_rf', ascending=False)
                
                # Criar barras
                bars = plt.barh(df_other['model'], df_other['diff_from_rf'], 
                        color=df_other['color'], alpha=0.7)
                
                # Adicionar linha vertical em zero
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                
                plt.title(f'Comparação de Modelos em Relação ao Random Forest (acc: {rf_acc:.4f})', fontsize=14)
                plt.xlabel('Diferença de Acurácia', fontsize=12)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                
                # Adicionar valores nas barras
                for bar in bars:
                    width = bar.get_width()
                    label_x_pos = width if width > 0 else width - 0.02
                    align = 'left' if width > 0 else 'right'
                    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                             f'{width:+.4f}', va='center', ha=align,
                             color='black', fontweight='bold', fontsize=10)
                
                plt.tight_layout()
                rf_comparison_chart = 'comparisons/model_vs_randomforest.png'
                plt.savefig(rf_comparison_chart, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(rf_comparison_chart)
    
    # 3. Feature Importance (se disponível nos resultados)
    for i, result in enumerate(model_results):
        if 'rf' in result and 'feature_importance' in result['rf']:
            feature_importance = result['rf']['feature_importance']
            feature_cols = result['rf'].get('feature_cols', [])
            
            # Se feature_cols não estiver disponível, criar nomes genéricos
            if not feature_cols or len(feature_cols) != len(feature_importance):
                feature_cols = [f"Feature {j+1}" for j in range(len(feature_importance))]
            
            # Criar DataFrame para importância
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': feature_importance
            })
            
            # Ordenar por importância
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Criar gráfico
            plt.figure(figsize=(10, 8))
            bars = plt.barh(importance_df['Feature'], importance_df['Importance'], 
                    color=sns.color_palette("viridis", len(importance_df)))
            
            plt.title('Importância das Features (Random Forest)', fontsize=16)
            plt.xlabel('Importância', fontsize=14)
            plt.tight_layout()
            
            # Adicionar valores nas barras
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{width:.4f}', va='center', ha='left',
                         color='black', fontsize=10)
            
            timestamp = result.get('test_timestamp', f'batch_{i}').replace(":", "-").replace(" ", "_")
            importance_chart = f'comparisons/feature_importance_{timestamp}.png'
            plt.savefig(importance_chart, dpi=300, bbox_inches='tight')
            plt.close()
            generated_charts.append(importance_chart)
    
    # 4. Matriz de Confusão para cada modelo
    for i, result in enumerate(model_results):
        # Para Random Forest
        if 'rf' in result and 'confusion_matrix' in result['rf']:
            cm = result['rf']['confusion_matrix']
            if isinstance(cm, list):
                # Converter para array numpy
                cm = np.array(cm)
                
                # Criar gráfico
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['high', 'low', 'medium'] if cm.shape[0] == 3 else None,
                            yticklabels=['high', 'low', 'medium'] if cm.shape[0] == 3 else None)
                plt.xlabel('Predito')
                plt.ylabel('Real')
                plt.title('Matriz de Confusão - Random Forest')
                plt.tight_layout()
                
                timestamp = result.get('test_timestamp', f'batch_{i}').replace(":", "-").replace(" ", "_")
                cm_chart = f'comparisons/rf_confusion_matrix_{timestamp}.png'
                plt.savefig(cm_chart, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(cm_chart)
        
        # Para Gemini Base
        if 'gemini_base' in result and result['gemini_base'] and 'confusion_matrix' in result['gemini_base']:
            cm = result['gemini_base']['confusion_matrix']
            if isinstance(cm, dict):
                # Converter de formato dict para DataFrame
                cm_df = pd.DataFrame(cm)
                
                # Criar gráfico
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Reds')
                plt.xlabel('Predito')
                plt.ylabel('Real')
                approach = result['gemini_base'].get('approach', 'base')
                plt.title(f'Matriz de Confusão - Gemini ({approach})')
                plt.tight_layout()
                
                timestamp = result.get('test_timestamp', f'batch_{i}').replace(":", "-").replace(" ", "_")
                cm_chart = f'comparisons/gemini_base_confusion_matrix_{timestamp}.png'
                plt.savefig(cm_chart, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(cm_chart)
    
    # 5. Tabela resumo em formato de figura
    fig, ax = plt.subplots(figsize=(12, len(df_comparison) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar dados para a tabela
    table_data = []
    for _, row in df_comparison.iterrows():
        table_data.append([
            row['model'],
            f"{row['accuracy']:.4f}",
            row['type'],
            row['timestamp'],
            row['dataset_size']
        ])
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Modelo', 'Acurácia', 'Tipo', 'Data/Hora', 'Tamanho Dataset'],
                    loc='center',
                    cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Colorir células baseado no tipo de modelo
    for i, row_data in enumerate(table_data):
        model_type = row_data[2]
        color = 'lightblue' if model_type == 'Traditional ML' else 'lightgreen' if model_type == 'LLM' else 'lightyellow'
        for j in range(len(row_data)):
            table[(i+1, j)].set_facecolor(color)
    
    plt.title('Resumo de Resultados dos Modelos', fontsize=16, pad=20)
    plt.tight_layout()
    
    table_path = 'comparisons/results_summary_table.png'
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_charts.append(table_path)
    
    # Exibir resultados
    print(f"Gerados {len(generated_charts)} arquivos de comparação:")
    for chart in generated_charts:
        print(f"- {chart}")
    
    return generated_charts

def main():
    """Função principal do pipeline com interface interativa."""
    np.random.seed(42)
    
    print("\n======================================")
    print("= Wine Quality Classification System =")
    print("======================================")
    
    # Informação sobre o modelo fine-tuned
    if GEMINI_AVAILABLE:
        print("\nINFORMAÇÃO: Este sistema possui um modelo Gemini fine-tuned disponível")
        print(f"ID do modelo: {PROJECT_ID_TUNED}")
        print(f"Caminho: {MODEL_TUNED}")
    
    # Inicialização: variáveis para armazenar resultados e modelos
    df = None
    train_df, val_df, test_df = None, None, None
    rf_results = None
    gemini_base_results = None
    final_test_results = None
    train_file, val_file, test_file = None, None, None
    
    # 1. Carregar dados primeiro (pré-requisito para todas as outras opções)
    print("\nCarregando dataset...")
    for filename in ["wine-quality.csv", "wine.csv", "winequality.csv", "wine_quality.csv"]:
        df = load_and_prepare(filename)
        if df is not None:
            print(f"Arquivo {filename} carregado com sucesso! Formato: {df.shape}")
            # Já dividir os dados para uso posterior
            train_df, val_df, test_df = split_data_for_finetune(df)
            break
    
    if df is None:
        print("ERRO: Não foi possível carregar o dataset. Verifique se o arquivo está disponível.")
        return
    
    # Variável para armazenar todos os resultados dos modelos testados
    all_test_results = {}
    
    # Menu principal
    while True:
        print("\n===== MENU PRINCIPAL =====")
        print("1. Preparar arquivos para fine-tuning do Gemini")
        print("2. Treinar e avaliar modelo Random Forest")
        print("3. Avaliar com Gemini (modo few-shot)")
        print("4. Avaliar com Gemini (modo zero-shot)")
        print("5. Teste final com todos os modelos")
        print("6. Visualizar resultados")
        print("7. Testar modelo com valores personalizados")
        print("8. Ver resumo de resultados")
        print("9. Executar pipeline completo")
        print("10. Salvar resultados e gerar comparativos")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção (0-9): ").strip()
        
        if choice == "0":
            print("\nEncerrando o programa. Até mais!")
            break
            
        elif choice == "1":
            print("\n=== Preparando arquivos para fine-tuning ===")
            train_file = prepare_finetune_data(train_df, output_file="gemini_finetune_train.jsonl")
            val_file = prepare_finetune_data(val_df, output_file="gemini_finetune_validation.jsonl")
            test_file = prepare_finetune_data(test_df, output_file="gemini_finetune_test.jsonl")
            
            print("\nArquivos para fine-tuning do Gemini gerados:")
            print(f"- Treino: {train_file}")
            print(f"- Validação: {val_file}")
            print(f"- Teste: {test_file}")
            
        elif choice == "2":
            print("\n=== Treinando modelo Random Forest ===")
            rf_results = train_random_forest(train_df, val_df)
            print("\nModelo Random Forest treinado e avaliado com sucesso!")
            
        elif choice == "3":
            if not GEMINI_AVAILABLE:
                print("\nAVISO: API do Gemini não configurada. Configure a GOOGLE_API_KEY.")
                continue
                
            print("\n=== Avaliando com Gemini (modo few-shot) ===")
            # Perguntar parâmetros
            try:
                batch_size = int(input("Tamanho do batch (padrão 29): ") or "29")
                pause_seconds = int(input("Segundos de pausa entre batches (padrão 60): ") or "60")
            except ValueError:
                print("Valores inválidos. Usando valores padrão.")
                batch_size, pause_seconds = 29, 60
                
            gemini_base_results = evaluate_with_gemini(
                val_df,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
                use_few_shot=True
            )

        elif choice == "4":
            if not GEMINI_AVAILABLE:
                print("\nAVISO: API do Gemini não configurada. Configure a GOOGLE_API_KEY.")
                continue
                
            print("\n=== Avaliando com Gemini (modo zero-shot) ===")
            # Perguntar parâmetros
            try:
                batch_size = int(input("Tamanho do batch (padrão 29): ") or "29")
                pause_seconds = int(input("Segundos de pausa entre batches (padrão 60): ") or "60")
            except ValueError:
                print("Valores inválidos. Usando valores padrão.")
                batch_size, pause_seconds = 29, 60
                
            gemini_base_results = evaluate_with_gemini(
                val_df,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
                use_few_shot=False
            )
            
        elif choice == "5":
            if rf_results is None:
                print("\nAVISO: É necessário treinar o modelo Random Forest primeiro (opção 2).")
                continue
                
            print("\n=== Realizando teste final nos modelos ===")
            final_test_results = test_final_models(
                test_df,
                rf_results,
                gemini_base_results
            )
            
            # Armazenar resultados para comparativos posteriormente
            all_test_results = final_test_results
            
            print("\nTeste final completado com sucesso!")
            
        elif choice == "6":
            if rf_results is None:
                print("\nAVISO: É necessário treinar o modelo Random Forest primeiro (opção 2).")
                continue
                
            print("\n=== Gerando visualizações dos resultados ===")
            visualize_results(
                rf_results,
                gemini_base_results,
                None,  # gemini_finetune_results
                final_test_results
            )
            print("\nVisualizações salvas no diretório atual.")
            
        elif choice == "7":
            if rf_results is None:
                print("\nAVISO: É necessário treinar o modelo Random Forest primeiro (opção 2).")
                continue
            if not GEMINI_TUNED_AVAILABLE:
                print("\nAVISO: Modelo fine-tunned não configurado." \
                " Configure as variaveis de ambiente PROJECT_ID_TUNED, MODEL_TUNED, LOCATION_TUNED.")
                continue
                
            print("\n=== Teste com Valores Personalizados ===")
            predict_user_input(rf_results['model'], rf_results['feature_cols'])
            
        elif choice == "8":
            print("\n===== RESUMO DE RESULTADOS =====")
            if rf_results:
                print(f"Random Forest (validação): {rf_results['accuracy']:.4f}")
            else:
                print("Random Forest: Não treinado")
                
            if gemini_base_results:
                print(f"Gemini ({gemini_base_results['approach']}): {gemini_base_results['accuracy']:.4f}")
            else:
                print("Gemini: Não avaliado")
                
            if final_test_results:
                print("\nResultados no conjunto de teste:")
                print(f"Random Forest (teste): {final_test_results['rf']['accuracy']:.4f}")
                if final_test_results.get('gemini_base'):
                    approach = final_test_results['gemini_base'].get('approach', 'base')
                    acc = final_test_results['gemini_base'].get('accuracy', 0)
                    print(f"Gemini ({approach}): {acc:.4f}")
            else:
                print("\nTeste final: Não realizado")
                
            if train_file:
                print("\nArquivos para fine-tuning:")
                print(f"- Treino: {train_file}")
                print(f"- Validação: {val_file}")
                print(f"- Teste: {test_file}")
            else:
                print("\nArquivos para fine-tuning: Não gerados")
                
        elif choice == "9":
            print("\n=== Executando Pipeline Completo ===")
            
            # 1. Preparar arquivos para fine-tuning
            print("\nEtapa 1: Preparando arquivos para fine-tuning...")
            train_file = prepare_finetune_data(train_df, output_file="gemini_finetune_train.jsonl")
            val_file = prepare_finetune_data(val_df, output_file="gemini_finetune_validation.jsonl")
            test_file = prepare_finetune_data(test_df, output_file="gemini_finetune_test.jsonl")
            
            # 2. Treinar Random Forest
            print("\nEtapa 2: Treinando modelo Random Forest...")
            rf_results = train_random_forest(train_df, val_df)
            
            # 3. Avaliar com Gemini (se disponível)
            if GEMINI_AVAILABLE:
                print("\nEtapa 3: Avaliando com Gemini (modo few-shot)...")
                gemini_base_results = evaluate_with_gemini(
                    val_df,
                    batch_size=29,
                    pause_seconds=60,
                    use_few_shot=True
                )
            else:
                print("\nEtapa 3: Avaliação com Gemini ignorada (API não configurada).")
            
            # 4. Teste final
            print("\nEtapa 4: Realizando teste final...")
            final_test_results = test_final_models(
                test_df,
                rf_results,
                gemini_base_results
            )
            
            # Armazenar resultados para comparativos
            all_test_results = final_test_results
            
            # 5. Visualizar resultados
            print("\nEtapa 5: Gerando visualizações...")
            visualize_results(
                rf_results,
                gemini_base_results,
                None,  # gemini_finetune_results
                final_test_results
            )
            
            # 6. Salvar resultados
            print("\nEtapa 6: Salvando resultados...")
            save_results(final_test_results, "wine_quality_pipeline_results.json")
            
            # 7. Gerar comparativos
            print("\nEtapa 7: Gerando comparativos...")
            generate_model_comparisons(all_results=final_test_results)
            
            print("\nPipeline completo executado com sucesso!")
            
        elif choice == "10":
            print("\n=== Salvando Resultados e Gerando Comparativos ===")
            
            if not final_test_results:
                print("Não há resultados de teste para salvar. Execute o teste final primeiro (opção 4).")
                continue
            
            # Salvar resultados em arquivo JSON
            results_file = save_results(final_test_results, "wine_quality_model_results.json")
            
            # Gerar comparativos
            comparison_files = generate_model_comparisons(all_results=final_test_results)
            
            print("\nResultados salvos e comparativos gerados com sucesso!")
            
            # Perguntar se deseja procurar e incluir resultados anteriores nos comparativos
            include_previous = input("\nDeseja incluir resultados anteriores nos comparativos? (s/n): ").lower()
            if include_previous.startswith('s'):
                # Procurar arquivos JSON no diretório results
                json_files = glob.glob('results/*.json')
                if json_files:
                    print(f"Encontrados {len(json_files)} arquivos de resultados anteriores.")
                    generate_model_comparisons(results_files=json_files)
                else:
                    print("Nenhum arquivo de resultados anteriores encontrado.")
            
        else:
            print("\nOpção inválida. Por favor, escolha um número entre 0 e 9.")
            
    print("\nPrograma finalizado!")

if __name__ == "__main__":
    main()