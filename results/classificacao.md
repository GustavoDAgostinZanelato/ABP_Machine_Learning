# Parte 2 - Classificação

## Etapa 1 - Comparação das discretizações

| Faixa | Classe | Quantidade | Proporção |
|---|---|---:|---:|
| 2 classes | Baixa (0-5) | 2001 | 37.2% |
| 2 classes | Alta (6-10) | 3374 | 62.8% |
| 3 classes | Baixa (0-5) | 2001 | 37.2% |
| 3 classes | Média (6) | 2354 | 43.8% |
| 3 classes | Alta (7-10) | 1020 | 19.0% |

Foi escolhida a discretização em três classes (quality 0-5, 6 e 7-10), conforme a proposta do projeto. A razao entre a maior e a menor classe é 2.31:1, contra 1.69:1 na opção binária. A opção de três classes preserva a categoria intermediária e mantém amostras suficientes em todas as classes para uma divisão estratificada.

## Etapa 2 - Preparação dos dados

Divisão estratificada 80/20 com `random_state=42`. A imputação e a padronização são mantidas dentro do pipeline e ajustadas somente nos dados de treino.

O `LabelEncoder` converte as categorias em códigos inteiros:

| Código | Classe | Treino | Teste |
|---:|---|---:|---:|
| 0 | Alta (7-10) | 816 | 204 |
| 1 | Baixa (0-5) | 1601 | 400 |
| 2 | Média (6) | 1883 | 471 |

Foi usado `class_weight="balanced"` nos três classificadores. Essa ponderação considera a frequência de cada classe durante o treino, sem reamostrar o conjunto de teste.

## Etapa 3 - Modelagem e avaliação

| Modelo | F1 CV padrão | F1 CV ajustado | F1 teste | Balanced accuracy CV padrão | Balanced accuracy CV ajustado | Balanced accuracy teste |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6052 | 0.6052 ± 0.0127 | 0.6223 | 0.6093 | 0.6093 ± 0.0147 | 0.6302 |
| SVM (RBF) | 0.5680 | 0.5747 ± 0.0162 | 0.5624 | 0.6121 | 0.6164 ± 0.0163 | 0.6038 |
| Regressão Logística | 0.5462 | 0.5466 ± 0.0257 | 0.5473 | 0.5876 | 0.5877 ± 0.0289 | 0.5939 |

A seleção de hiperparâmetros usou GridSearchCV com F1-macro e StratifiedKFold de 5 folds (embaralhado, `random_state=42`). A validação cruzada foi realizada somente no conjunto de treino; o conjunto de teste foi reservado para a avaliação final.

### Melhores hiperparâmetros

- **Regressão Logística**: `{'modelo__C': 10}`
- **Random Forest**: `{'modelo__max_depth': None, 'modelo__min_samples_leaf': 1, 'modelo__n_estimators': 300}`
- **SVM (RBF)**: `{'modelo__C': 5, 'modelo__gamma': 'scale'}`

Melhor modelo pela validação cruzada: **Random Forest** (F1-macro 0.6052).

### Matrizes de confusão no teste

Linhas representam a classe real e colunas a classe predita; cada célula mostra contagem e proporção dentro da classe real.

#### Regressão Logística

| Real \ Predita | Alta (7-10) | Baixa (0-5) | Média (6) |
|---|---|---|---|
| Alta (7-10) | 153 (75.0%) | 12 (5.9%) | 39 (19.1%) |
| Baixa (0-5) | 35 (8.8%) | 270 (67.5%) | 95 (23.8%) |
| Média (6) | 169 (35.9%) | 134 (28.5%) | 168 (35.7%) |

Confusão mais frequente: **Média (6)** prevista como **Alta (7-10)** (169 casos; 35.9% da classe real).

#### Random Forest

| Real \ Predita | Alta (7-10) | Baixa (0-5) | Média (6) |
|---|---|---|---|
| Alta (7-10) | 123 (60.3%) | 8 (3.9%) | 73 (35.8%) |
| Baixa (0-5) | 14 (3.5%) | 290 (72.5%) | 96 (24.0%) |
| Média (6) | 95 (20.2%) | 111 (23.6%) | 265 (56.3%) |

Confusão mais frequente: **Média (6)** prevista como **Baixa (0-5)** (111 casos; 23.6% da classe real).

#### SVM (RBF)

| Real \ Predita | Alta (7-10) | Baixa (0-5) | Média (6) |
|---|---|---|---|
| Alta (7-10) | 146 (71.6%) | 11 (5.4%) | 47 (23.0%) |
| Baixa (0-5) | 26 (6.5%) | 288 (72.0%) | 86 (21.5%) |
| Média (6) | 177 (37.6%) | 117 (24.8%) | 177 (37.6%) |

Confusão mais frequente: **Média (6)** prevista como **Alta (7-10)** (177 casos; 37.6% da classe real).

### Relatório por classe - Random Forest

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Alta (7-10) | 0.5302 | 0.6029 | 0.5642 | 204 |
| Baixa (0-5) | 0.7090 | 0.7250 | 0.7169 | 400 |
| Média (6) | 0.6106 | 0.5626 | 0.5856 | 471 |

### Variáveis mais decisivas - Random Forest

Importância por permutação medida no teste: quanto maior a queda do F1-macro ao embaralhar uma variável, maior sua contribuição para as previsões.

| Variável | Redução média do F1-macro | Desvio padrão |
|---|---:|---:|
| alcohol | 0.13870 | 0.00557 |
| volatile acidity | 0.03708 | 0.01141 |
| chlorides | 0.03315 | 0.00995 |
| sulphates | 0.03153 | 0.00557 |
| total sulfur dioxide | 0.03069 | 0.00534 |
| free sulfur dioxide | 0.02646 | 0.00494 |
| density | 0.02056 | 0.00700 |
| fixed acidity | 0.00988 | 0.00391 |
| citric acid | 0.00964 | 0.00742 |
| pH | 0.00845 | 0.00658 |
| residual sugar | 0.00202 | 0.00553 |
| type | -0.00074 | 0.00148 |

Variável mais decisiva: **alcohol** (queda média de F1-macro 0.13870).

A importância por permutação é uma análise descritiva no conjunto de teste; não foi usada para escolher o modelo nem ajustar hiperparâmetros.
