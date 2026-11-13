# Parte 2 - Classificação

## Etapa 1 - Comparação das discretizações

| Faixa | Classe | Quantidade | Proporção |
|---|---|---:|---:|
| 2 classes | Baixa (0-5) | 2001 | 37.2% |
| 2 classes | Alta (6-10) | 3374 | 62.8% |
| 3 classes | Baixa (0-5) | 2001 | 37.2% |
| 3 classes | Média (6) | 2354 | 43.8% |
| 3 classes | Alta (7-10) | 1020 | 19.0% |

Foi escolhida a discretização em três classes (quality 0-5, 6 e 7-10). A nota 6 é a moda do dataset e concentra a maior parte dos vinhos, então tratá-la como classe própria evita misturar vinhos medianos com os de qualidade baixa ou alta. A razão entre a maior e a menor classe é 2.31:1, contra 1.69:1 na opção binária: o desbalanceamento é maior, mas moderado, e todas as classes mantêm amostras suficientes para uma divisão estratificada. O desbalanceamento é tratado com `class_weight="balanced"` em vez de reamostragem, que preserva todos os dados reais e evita vazamento na validação cruzada.

## Etapa 2 - Preparação dos dados

Divisão estratificada 80/20 com `random_state=42`. A imputação e a padronização são mantidas dentro do pipeline e ajustadas somente nos dados de treino.

O `LabelEncoder` converte as categorias em códigos inteiros:

| Código | Classe | Treino | Teste |
|---:|---|---:|---:|
| 0 | Baixa (0-5) | 1601 | 400 |
| 1 | Média (6) | 1883 | 471 |
| 2 | Alta (7-10) | 816 | 204 |

Foi usado `class_weight="balanced"` nos três classificadores. Essa ponderação considera a frequência de cada classe durante o treino, sem reamostrar o conjunto de teste.

## Etapa 3 - Modelagem e avaliação

| Modelo | F1 CV padrão | F1 CV ajustado | F1 teste | Balanced accuracy CV padrão | Balanced accuracy CV ajustado | Balanced accuracy teste |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.5997 | 0.6006 ± 0.0065 | 0.6331 | 0.6068 | 0.6089 ± 0.0078 | 0.6439 |
| SVM (RBF) | 0.5639 | 0.5639 ± 0.0042 | 0.5797 | 0.6111 | 0.6111 ± 0.0056 | 0.6255 |
| Regressão Logística | 0.5457 | 0.5460 ± 0.0157 | 0.5799 | 0.5911 | 0.5913 ± 0.0166 | 0.6199 |

A seleção de hiperparâmetros usou GridSearchCV com F1-macro e StratifiedKFold de 5 folds (embaralhado, `random_state=42`). A validação cruzada foi realizada somente no conjunto de treino; o conjunto de teste foi reservado para a avaliação final.

### Melhores hiperparâmetros

- **Regressão Logística**: `{'modelo__C': 0.1}`
- **Random Forest**: `{'modelo__max_depth': 20, 'modelo__min_samples_leaf': 1, 'modelo__n_estimators': 300}`
- **SVM (RBF)**: `{'modelo__C': 1, 'modelo__gamma': 'scale'}`

Melhor modelo pela validação cruzada: **Random Forest** (F1-macro 0.6006).

### Matrizes de confusão no teste

Linhas representam a classe real e colunas a classe predita; cada célula mostra contagem e proporção dentro da classe real.

#### Regressão Logística

| Real \ Predita | Baixa (0-5) | Média (6) | Alta (7-10) |
|---|---|---|---|
| Baixa (0-5) | 298 (74.5%) | 68 (17.0%) | 34 (8.5%) |
| Média (6) | 132 (28.0%) | 188 (39.9%) | 151 (32.1%) |
| Alta (7-10) | 14 (6.9%) | 44 (21.6%) | 146 (71.6%) |

Confusão mais frequente: **Média (6)** prevista como **Alta (7-10)** (151 casos; 32.1% da classe real).

#### Random Forest

| Real \ Predita | Baixa (0-5) | Média (6) | Alta (7-10) |
|---|---|---|---|
| Baixa (0-5) | 304 (76.0%) | 85 (21.2%) | 11 (2.8%) |
| Média (6) | 128 (27.2%) | 254 (53.9%) | 89 (18.9%) |
| Alta (7-10) | 6 (2.9%) | 69 (33.8%) | 129 (63.2%) |

Confusão mais frequente: **Média (6)** prevista como **Baixa (0-5)** (128 casos; 27.2% da classe real).

#### SVM (RBF)

| Real \ Predita | Baixa (0-5) | Média (6) | Alta (7-10) |
|---|---|---|---|
| Baixa (0-5) | 309 (77.2%) | 59 (14.8%) | 32 (8.0%) |
| Média (6) | 141 (29.9%) | 176 (37.4%) | 154 (32.7%) |
| Alta (7-10) | 15 (7.4%) | 40 (19.6%) | 149 (73.0%) |

Confusão mais frequente: **Média (6)** prevista como **Alta (7-10)** (154 casos; 32.7% da classe real).

### Relatório por classe - Random Forest

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---:|---:|---:|---:|
| Baixa (0-5) | 0.6941 | 0.7600 | 0.7255 | 400 |
| Média (6) | 0.6225 | 0.5393 | 0.5779 | 471 |
| Alta (7-10) | 0.5633 | 0.6324 | 0.5958 | 204 |

### Variáveis mais decisivas - Random Forest

Importância por permutação medida no teste: quanto maior a queda do F1-macro ao embaralhar uma variável, maior sua contribuição para as previsões.

| Variável | Redução média do F1-macro | Desvio padrão |
|---|---:|---:|
| alcohol | 0.14445 | 0.02215 |
| volatile acidity | 0.05382 | 0.00734 |
| pH | 0.03094 | 0.00662 |
| free sulfur dioxide | 0.02039 | 0.00966 |
| total sulfur dioxide | 0.01960 | 0.00536 |
| sulphates | 0.01891 | 0.00560 |
| chlorides | 0.01832 | 0.00534 |
| density | 0.01803 | 0.00595 |
| citric acid | 0.01110 | 0.00674 |
| residual sugar | 0.01012 | 0.00536 |
| fixed acidity | 0.00326 | 0.00622 |
| type | 0.00162 | 0.00186 |

Variável mais decisiva: **alcohol** (queda média de F1-macro 0.14445).

A importância por permutação é uma análise descritiva no conjunto de teste; não foi usada para escolher o modelo nem ajustar hiperparâmetros.
