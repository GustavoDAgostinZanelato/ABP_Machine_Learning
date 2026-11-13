# ABP — Machine Learning: Previsão da Qualidade do Vinho

Projeto da Avaliação Baseada em Projeto (ABP) de Machine Learning.
O objetivo é prever a variável `quality` (0–10) de vinhos tintos e brancos
a partir de características físico-químicas, cobrindo todo o fluxo de um
projeto de ML: limpeza e pré-processamento, análise exploratória, regressão
e classificação.

## 📁 Estrutura do projeto

```
ABP_Machine_Learning/
├── .docs/                          # Dados e enunciado (não versionados pelo conteúdo)
│   ├── DescricaoTrabalhoFinal.pdf  # Enunciado do trabalho
│   ├── group_10_winequality.csv    # Dataset original (com ruído/sujo)
│   └── wine_limpo.csv              # Dataset limpo gerado pelo limpezaDados.py
├── limpezaDados.py                 # Etapa 1: limpeza e pré-processamento
├── analiseExploratoria.py          # Etapa 2: análise exploratória
├── regressao.py                    # Parte 1: regressão
├── classificacao.py                # Parte 2: classificação
├── figures/                        # Gráficos gerados pelos scripts
├── results/                        # Resultados e resumos em CSV/Markdown
├── requirements.txt                # Dependências do projeto
├── .gitignore
└── README.md
```

## 🔧 Pré-requisitos

- **Python 3.10+** (recomendado 3.11 ou 3.12)
- **pip** (já vem com o Python)
- **git** (para clonar o repositório)

## 🚀 Como inicializar o projeto

> Os comandos abaixo assumem um terminal com Python disponível no `PATH`.
> No Windows, use o **Git Bash** ou o **PowerShell**.

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/ABP_Machine_Learning.git
cd ABP_Machine_Learning
```

### 2. Criar e ativar um ambiente virtual

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Git Bash / PowerShell):**

```bash
python -m venv venv
source venv/Scripts/activate   # Git Bash
# ou
.\venv\Scripts\Activate.ps1    # PowerShell
```

> Após ativar, o prompt do terminal passará a mostrar `(venv)`.

### 3. Atualizar o pip

```bash
python -m pip install --upgrade pip
```

### 4. Instalar as dependências

```bash
pip install -r requirements.txt
```

As dependências são:

| Pacote        | Uso                                  |
|---------------|--------------------------------------|
| pandas        | Manipulação de dados tabulares       |
| numpy         | Operações numéricas                  |
| matplotlib    | Plotagens básicas                    |
| seaborn       | Visualizações estatísticas (heatmap) |
| scikit-learn  | Modelos, pipelines e métricas de ML  |

### 5. Verificar a instalação

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('Dependências OK')"
```

Se aparecer `Dependências OK`, o ambiente está pronto.

### 6. Executar o pré-processamento

O script `limpezaDados.py` lê o dataset original em `.docs/group_10_winequality.csv`,
aplica a limpeza/normalização e salva o resultado em `.docs/wine_limpo.csv`.

```bash
python limpezaDados.py
```

### 7. Desativar o ambiente (quando terminar)

```bash
deactivate
```

---

## 📋 Etapas do trabalho

### Etapa 1 — Limpeza e pré-processamento (vale para as duas partes)

1. Carregar o dataset e inspecionar estrutura, tipos e valores ausentes.
2. Identificar problemas: escalas inconsistentes, tipos corrompidos,
   valores fora de faixa, duplicatas e registros vazios.
3. **Corrigir ou remover registros inválidos**: `type` corrompido ou vazio,
   `quality` vazio ou fora de 0–10 e a linha vazia. Documentar cada decisão.
4. **Tratar ausentes**: remover ou imputar, justificando. O enunciado pede
   isso "se existirem", e aqui existem.
5. **Tratar duplicatas**: decidir se remove e justificar. Cuidado com o
   vazamento entre treino e teste que as duplicatas causam.
6. **Normalizar as escalas inconsistentes** (ponto crítico). Descubra o padrão
   (por exemplo, colunas multiplicadas por 10³ em parte das linhas) e corrija
   com um critério explícito. Valide comparando as distribuições antes e depois
   com os valores esperados do Wine Quality original.
7. **Detectar outliers** (IQR, z-score ou boxplots) e decidir o que fazer com
   os que sobrarem após a correção de escala. Valores absurdos
   (pH = 114, volatile acidity = 6600) provavelmente são erro e não outlier
   legítimo.
8. **Codificar `type`** (red/white → 0/1 ou one-hot).
9. **Separar treino e teste** antes de qualquer ajuste. Usar `stratify` na
   classificação.
10. **Padronizar com `StandardScaler`**, ajustado só no treino, dentro de um
    `Pipeline`.

### Etapa 2 — Análise exploratória

11. Histogramas das variáveis e de `quality`.
12. Scatter plots das features contra `quality`.
13. Matriz de correlação (heatmap).
14. Comparação entre tinto e branco, se fizer sentido.
15. É recomendável mostrar a EDA antes e depois da limpeza. Isso evidencia o
    problema e justifica o tratamento.

### Etapa 3 — Parte 1: Regressão

16. Treinar pelo menos 3 modelos (por exemplo Regressão Linear/Ridge,
    Random Forest e Gradient Boosting ou SVR).
17. Avaliar com **RMSE, MAE e R²**, usando validação cruzada.
18. Fazer o ajuste de hiperparâmetros com `GridSearchCV` ou
    `RandomizedSearchCV`.
19. Discutir os coeficientes (modelo linear) e a importância das variáveis
    (modelos de árvore), indicando quais mais impactam a qualidade.
20. Comparar os resultados antes e depois do tuning.

#### Como executar a Parte 1

Com o ambiente virtual ativo e a partir da raiz do projeto:

```bash
python regressao.py
```

O script reaproveita a limpeza do `limpezaDados.py`, treina Ridge, Random Forest e
Gradient Boosting com validação cruzada 5-fold, afina os hiperparâmetros com
`GridSearchCV` (critério: RMSE) e gera:

| Arquivo                          | Conteúdo                                                    |
|----------------------------------|-------------------------------------------------------------|
| `figures/07_comparacao_modelos.png` | RMSE, MAE e R² em validação cruzada e no teste, por modelo |
| `figures/08_importancia_features.png` | Coeficientes do Ridge e importância das variáveis nas árvores |
| `results/regressao.csv`          | Métricas de cada modelo antes e depois do tuning            |
| `results/regressao.md`           | Resumo com tabelas e conclusões, pronto para o relatório    |

Tempo de execução: cerca de 1 a 2 minutos.

### Etapa 4 — Parte 2: Classificação

`classificacao.py` completa a Parte 2: compara duas e três faixas de `quality`,
seleciona três classes (0-5, 6 e 7-10), faz divisão estratificada e treina
Regressão Logística, Random Forest e SVM com pesos balanceados. Os modelos são
comparados por F1-macro e balanced accuracy, ajustados com `GridSearchCV` e
`StratifiedKFold` de 5 folds, e analisados por matriz de confusão e importância
de variáveis por permutação.

```bash
python classificacao.py
```

O script gera `results/classificacao.md`, CSVs com distribuição, métricas e
importância, além de figuras para distribuição, comparação dos modelos, matrizes
de confusão e variáveis relevantes (`figures/09` a `figures/13`).
