# 🍷 Classificação de Qualidade de Vinhos com ML & LLMs
## Guia Completo de Implementação e Fine-tuning

![Classificação de Vinhos](https://img.shields.io/badge/Classifica%C3%A7%C3%A3o-Qualidade%20de%20Vinhos-purple)
![Modelos](https://img.shields.io/badge/Modelos-Random%20Forest%20|%20Gemini%20|%20LLM%20Fine--tuned-blue)
![Status](https://img.shields.io/badge/Status-Ativo-green)

Um sistema completo para classificação de qualidade de vinhos que compara aprendizado de máquina tradicional (Random Forest) com Modelos de Linguagem de Grande Escala (Google Gemini), com recursos de fine-tuning, testes interativos e análises detalhadas de desempenho.

<p align="center">
  <img src="https://storage.googleapis.com/kaggle-datasets-images/7018321/11234696/e509e584ac952d3def6470a2a6817b0e/dataset-cover.png" alt="Sistema de Classificação de Vinhos" width="600"/>
</p>

## 📋 Índice
- [✨ Funcionalidades](#-funcionalidades)
- [🛠️ Instalação e Configuração](#-instalacao-e-configuracao)
- [🚀 Uso do Sistema](#-uso-do-sistema)
- [🔄 Guia de Fine-tuning do Gemini](#-guia-de-fine-tuning-do-gemini)
- [🧪 Testando Diferentes Modelos](#-testando-diferentes-modelos)
- [📈 Exemplos de Visualizações](#-exemplos-de-visualizacoes)
- [🔍 Solução de Problemas](#-solucao-de-problemas)
- [🤝 Contribuindo](#-contribuindo)
- [📄 Licença](#-licenca)
- [🔗 Referências](#-referencias)

## ✨ Funcionalidades <a name="-funcionalidades"/>

- 🤖 **Suporte a Múltiplos Modelos**: Compare Random Forest vs Gemini vs LLM com Fine-tuning
- 📊 **Análises Detalhadas**: Visualize o desempenho dos modelos com métricas abrangentes
- 🔄 **Pipeline de Fine-tuning**: Gere e prepare dados para fine-tuning do Gemini
- 🧪 **Testes Interativos**: Teste modelos com entradas personalizadas ou exemplos pré-definidos
- 📈 **Importância das Características**: Analise quais características do vinho são mais relevantes
- 📱 **Interface Amigável**: Interface baseada em menu para fácil navegação
- 📋 **Relatórios Extensos**: Gere relatórios detalhados de classificação e comparações

## 🛠️ Instalação e Configuraçãos <a name="-instalacao-e-configuracao"/>

### Pré-requisitos

- Python 3.8+
- Conta Google Cloud com acesso à API Gemini
- Arquivo de credenciais de conta de serviço (para acesso ao modelo com fine-tuning)

### Configuração Básica

1. **Clone o repositório**
   ```bash
   git clone https://github.com/SidneyMelo/classificacao-qualidade-vinhos.git
   cd classificacao-qualidade-vinhos
   ```

2. **Crie e ative um ambiente virtual**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure o ambiente**
   
   Crie um arquivo `.env` na raiz do projeto:
   ```
   GOOGLE_API_KEY=sua_chave_api_gemini_aqui
   ```

5. **Configure o dataset**
   
   Coloque seu dataset de qualidade de vinhos (CSV) na raiz do projeto. Nomes de arquivo aceitos:
   - `wine-quality.csv`
   - `wine.csv`
   - `winequality.csv`
   - `wine_quality.csv`

## 🚀 Uso do Sistema <a name="-uso-do-sistema"/>

### Executando a Aplicação

```bash
python main.py
```

### Menu Interativo

A aplicação oferece um menu interativo com as seguintes opções:

1. **Preparar arquivos para fine-tuning do Gemini** - Cria arquivos JSONL para treinamento
2. **Treinar e avaliar modelo Random Forest** - Constrói e valida o modelo RF
3. **Avaliar com Gemini (modo few-shot)** - Testa usando aprendizado few-shot
4. **Teste final com todos os modelos** - Avalia todos os modelos no conjunto de teste
5. **Visualizar resultados** - Gera gráficos e visualizações comparativas
6. **Testar modelo com valores personalizados** - Interface interativa para testes customizados
7. **Ver resumo de resultados** - Mostra visão geral das métricas de desempenho
8. **Executar pipeline completo** - Executa todas as etapas automaticamente
9. **Salvar resultados e gerar comparativos** - Salva resultados em JSON e cria visualizações
0. **Sair** - Encerra o programa

### Testes Personalizados

Ao selecionar a opção 6, você pode:
- Inserir características do vinho manualmente
- Usar exemplos pré-definidos para vinhos de qualidade baixa, média ou alta
- Escolher entre diferentes modelos (Random Forest, variantes do Gemini, modelo com fine-tuning)

### Exemplos de Saída

#### Comparação de Modelos

```
=== COMPARATIVO DE MODELOS ===
Dataset de teste: 200 exemplos
Random Forest: 0.8950
Gemini (few-shot): 0.7750
  Diferença vs. RF: -0.1200
Gemini Fine-tuned (fine-tuned): 0.8400
  Diferença vs. RF: -0.0550

Melhor modelo: Random Forest (Acurácia: 0.8950)
```

#### Importância das Características

O sistema gera visualizações mostrando quais características do vinho têm maior impacto na classificação de qualidade:

```
Importância das Features (Random Forest):
- teor alcoólico: 0.3548
- densidade: 0.2126
- acidez volátil: 0.1247
- cloretos: 0.0873
- ...
```

#### Predição Personalizada

```
Valores usados para predição:
- acidez fixa: 7.4 g/dm³
- açúcar residual: 1.9 g/dm³
- teor alcoólico: 9.4 %
- densidade: 0.9978 g/cm³

Resultados da Predição:
Random Forest prediz: low
Probabilidades por classe:
  - high: 0.1600 (16.0%)
  - low: 0.5700 (57.0%)
  - medium: 0.2700 (27.0%)
Gemini prediz: low
```

## 🔄 Guia de Fine-tuning do Gemini <a name="-guia-de-fine-tuning-do-gemini"/>

O processo de fine-tuning permite melhorar o desempenho do modelo Gemini especificamente para a tarefa de classificação de qualidade de vinhos. Siga os passos abaixo para realizar o fine-tuning e integrar o modelo ao projeto.

### 1. Preparação de Dados para Fine-tuning

1. **Execute o Programa Principal**
   ```bash
   python main.py
   ```

2. **Selecione a Opção 1**
   - No menu interativo, selecione a opção **1: Preparar arquivos para fine-tuning do Gemini**
   - O sistema irá:
     - Processar o dataset de vinhos
     - Criar exemplos formatados para o modelo
     - Gerar um arquivo JSONL com pares de instruções/respostas
     - Salvar o arquivo (geralmente como `wine_tuning_data.jsonl`)

### 2. Acessar e Configurar o Google Cloud

1. **Acessar o Console**
   - Abra o navegador e acesse [https://console.cloud.google.com](https://console.cloud.google.com)
   - Faça login com sua conta Google

2. **Configurar o Ambiente Vertex AI**
   - No menu lateral, navegue até **Vertex AI**
   - Selecione **Vertex AI Studio**
   - Clique na seção **Ajuste**

3. **Criar Modelo Ajustado**
   - Clique em **Criar um modelo ajustado**
   - Preencha as informações:
     - **Nome do modelo**: `gemini-wine-quality-classifier` (ou outro nome descritivo)
     - **Modelo base**: Selecione o modelo Gemini mais adequado
   - Clique em **Continuar**

### 3. Carregar os Dados e Iniciar o Treinamento

1. **Upload do Arquivo JSONL**
   - Localize o arquivo `wine_tuning_data.jsonl` gerado na etapa 1
   - Faça upload deste arquivo
   
2. **Configurar Validação**
   - Clique em **Procurar** para selecionar um bucket para validação
   - Se não tiver um bucket:
     - Clique em **Criar novo bucket**
     - Preencha os campos e confirme a criação

3. **Iniciar o Treinamento**
   - Clique em **Iniciar Ajuste** para começar o processo de fine-tuning
   - Aguarde a conclusão do treinamento (pode levar algumas horas)

### 4. Testar e Obter Credenciais

1. **Acessar o Modelo Ajustado**
   - Volte para **Vertex AI → Ajuste**
   - Localize e clique no nome do seu modelo recém-treinado

2. **Testar o Modelo**
   - Clique em **Testar** 
   - Experimente com exemplos para verificar as classificações
   - Exemplo de prompt para teste:
     ```
     Classifique este vinho como low, medium ou high:
     - acidez fixa: 7.4 g/dm³
     - acidez volátil: 0.7 g/dm³
     - ácido cítrico: 0 g/dm³
     - açúcar residual: 1.9 g/dm³
     - cloretos: 0.076 g/dm³
     - dióxido de enxofre livre: 11 mg/dm³
     - dióxido de enxofre total: 34 mg/dm³
     - densidade: 0.9978 g/cm³
     - pH: 3.51
     - sulfatos: 0.56 g/dm³
     - teor alcoólico: 9.4 %
     ```

3. **Obter Informações para API**
   - Clique em **Receber Código**
   - Anote os valores das seguintes variáveis:
     - `project` (ID do projeto)
     - `location` (região)
     - `model` (ID do modelo/endpoint)

### 5. Configurar Autenticação e Credenciais

1. **Criar Conta de Serviço**
   - Volte para a Dashboard principal
   - Navegue até **IAM e Administrador → Contas de Serviço**
   - Clique em **Criar conta de serviço**
     - Nome: `wine-quality-service-account`
     - Papel: **Agente de Serviço VERTEX AI**
   - Confirme a criação

2. **Gerar Chave de Acesso**
   - Clique na conta de serviço recém-criada
   - Navegue até a aba **Chaves**
   - Clique em **Criar nova chave**
   - Selecione o formato **JSON**
   - O download da chave iniciará automaticamente

3. **Configurar Credenciais no Projeto**
   - Mova o arquivo JSON baixado para a pasta raiz do projeto
   - Renomeie o arquivo para `gen-lang-cred.json`

4. **Atualizar Variáveis de Ambiente**
   - Abra o arquivo `.env` na raiz do projeto
   - Adicione/atualize as seguintes variáveis:
     ```
     GOOGLE_API_KEY=sua_chave_api_gemini_aqui
     PROJECT_ID_TUNED=seu_project_id
     MODEL_TUNED=projects/seu_project_id/locations/sua_location/endpoints/seu_endpoint_id
     LOCATION_TUNED=sua_location
     ```

### 6. Integrar e Testar o Modelo Fine-tuned

1. **Executar o Programa**
   ```bash
   python main.py
   ```

2. **Testar o Modelo Fine-tuned**
   - Selecione a opção **6: Testar modelo com valores personalizados**
   - Escolha o modelo fine-tuned quando solicitado
   - Insira características do vinho ou use exemplos pré-definidos

3. **Avaliar Desempenho**
   - Selecione a opção **4: Teste final com todos os modelos**
   - Compare o desempenho do modelo fine-tuned com os outros modelos
   
4. **Visualizar Resultados**
   - Selecione a opção **5: Visualizar resultados**
   - Explore os gráficos comparativos e as métricas de desempenho

## 📊 Estrutura do Projeto

```
classificacao-qualidade-vinhos/
├── main.py                          # Script principal
├── requirements.txt                 # Dependências
├── .env                             # Variáveis de ambiente (chaves de API)
├── gen-lang-cred.json               # Credenciais da conta de serviço
├── wine-quality.csv                 # Dataset
├── README.md                        # Documentação
├── results/                         # Resultados gerados (JSON)
└── comparisons/                     # Visualizações geradas
```

## 🧪 Testando Diferentes Modelos <a name="-testando-diferentes-modelos"/>

O sistema suporta três tipos de modelos:

1. **Random Forest**: Abordagem tradicional de ML
   - Vantagens: Alto desempenho (geralmente a maior acurácia), rápido, interpretável
   - Desvantagens: Menos flexível para novos tipos de dados

2. **Gemini com Aprendizado Few-shot**: Usando exemplos para guiar o LLM
   - Vantagens: Não requer treinamento específico, adaptável
   - Desvantagens: Geralmente menos preciso que os outros métodos

3. **Gemini com Fine-tuning**: Modelo treinado especificamente para classificação de vinhos
   - Vantagens: Combina o conhecimento do LLM com treinamento específico da tarefa
   - Desvantagens: Requer processo adicional de fine-tuning

Você pode comparar o desempenho de cada um e usar os diferentes modelos para predições através do menu interativo da aplicação.

## 📈 Exemplos de Visualizações <a name="-exemplos-de-visualizacoes"/>

O sistema gera várias visualizações:

- **Gráfico de Comparação de Modelos**: Gráfico de barras comparando acurácia entre modelos
- **Importância das Características**: Gráfico de barras horizontais mostrando a importância das features
- **Matrizes de Confusão**: Mapas de calor mostrando predição vs classes reais
- **Tabela Resumo de Resultados**: Dados tabulares mostrando todas as métricas

## 🔍 Solução de Problemas <a name="-solucao-de-problemas"/>

### Erros de Importação

Se você encontrar erros de importação:

```bash
pip install --upgrade google-generativeai google-cloud-aiplatform
```

### Erros de Autenticação

Certifique-se de que:
- Seu arquivo `.env` contém a chave de API correta
- Seu arquivo `gen-lang-cred.json` está formatado corretamente e possui as permissões adequadas
- A conta de serviço tem acesso ao endpoint do modelo com fine-tuning

### Problemas com Fine-tuning

- **Modelo não aparece na lista:**
  - Verifique se as variáveis de ambiente estão corretas no arquivo `.env`
  - Confirme que o arquivo `gen-lang-cred.json` está na pasta raiz
  - Verifique se a conta de serviço tem as permissões corretas

- **Erros durante o treinamento:**
  - Verifique o formato do arquivo JSONL gerado
  - Aumente o tamanho do dataset de treinamento executando a opção 1 novamente
  - Verifique o histórico de ajustes no console do Google Cloud

- **Predições imprecisas:**
  - Considere retreinar o modelo com mais exemplos
  - Execute a opção 1 novamente para gerar um conjunto mais robusto de dados de treinamento
  - Experimente diferentes configurações de temperatura no código

### Dataset Não Encontrado

O sistema procura por várias variantes de nome de arquivo. Certifique-se de que pelo menos um deles esteja presente na raiz do projeto:
- `wine-quality.csv`
- `wine.csv`
- `winequality.csv`
- `wine_quality.csv`

## 🤝 Contribuindo <a name="-contribuindo"/>

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para a sua funcionalidade (`git checkout -b funcionalidade/recurso-incrivel`)
3. Faça commit das suas alterações (`git commit -m 'Adiciona recurso incrível'`)
4. Faça push para a branch (`git push origin funcionalidade/recurso-incrivel`)
5. Abra um Pull Request

## 📄 Licença <a name="-licenca"/>

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

## 🔗 Referências <a name="-referencias"/>

- [Dataset de Qualidade de Vinhos](https://www.kaggle.com/datasets/sahideseker/wine-quality-classification)
- [Documentação da API Google Gemini](https://ai.google.dev/docs)
- [Documentação de Fine-tuning do Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/quickstart-multimodal)
- [Guia de Autenticação do Google Cloud](https://cloud.google.com/docs/authentication/getting-started)
- [Tutoriais de Vertex AI no Google Cloud](https://cloud.google.com/vertex-ai/docs/tutorials)

---

<p align="center">
  <i>Desenvolvido para explorar o potencial dos Modelos de Linguagem de Grande Escala em tarefas tradicionais de classificação</i>
</p>
