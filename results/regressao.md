# Etapa 3 - Parte 1: Regressão da qualidade

## Métricas antes e depois do tuning

| etapa | modelo | rmse_cv | rmse_cv_std | mae_cv | r2_cv | rmse_teste | mae_teste | r2_teste |
|---|---|---|---|---|---|---|---|---|
| padrão | Ridge | 0.7364 | 0.0232 | 0.5699 | 0.2973 | - | - | - |
| padrão | Random Forest | 0.6938 | 0.0236 | 0.5348 | 0.3759 | - | - | - |
| padrão | Gradient Boosting | 0.7062 | 0.0239 | 0.5498 | 0.3533 | - | - | - |
| ajustado | Ridge | 0.7364 | 0.0231 | 0.5699 | 0.2974 | 0.7239 | 0.5572 | 0.3206 |
| ajustado | Random Forest | 0.6914 | 0.0236 | 0.5325 | 0.3802 | 0.6807 | 0.5198 | 0.3992 |
| ajustado | Gradient Boosting | 0.7065 | 0.0234 | 0.5489 | 0.3527 | 0.6817 | 0.5300 | 0.3976 |

Validação cruzada 5-fold no treino. As colunas de teste foram calculadas uma única vez, ao final, só para os modelos já ajustados; o modelo escolhido é o de menor RMSE em validação cruzada.

## Melhores hiperparâmetros (GridSearchCV)

- **Ridge**: `{'modelo__alpha': 10}`
- **Random Forest**: `{'modelo__max_depth': None, 'modelo__min_samples_leaf': 1, 'modelo__n_estimators': 300}`
- **Gradient Boosting**: `{'modelo__learning_rate': 0.05, 'modelo__max_depth': 3, 'modelo__n_estimators': 300}`

## Variáveis mais influentes

Coeficientes padronizados do Ridge (sinal: + aumenta a nota, - reduz):

| variável | coeficiente |
|---|---|
| alcohol | 0.3248 |
| volatile acidity | -0.2243 |
| density | -0.1410 |
| residual sugar | 0.1394 |
| free sulfur dioxide | 0.1173 |

Importância das variáveis no Random Forest (top 5):

| variável | importância |
|---|---|
| alcohol | 0.2587 |
| volatile acidity | 0.1124 |
| free sulfur dioxide | 0.0896 |
| total sulfur dioxide | 0.0761 |
| sulphates | 0.0757 |

## Conclusões

- Melhor modelo: Random Forest, com RMSE 0.6914 e R² 0.3802 em validação cruzada (pior: Ridge, RMSE 0.7364).
- Efeito do tuning no RMSE de validação cruzada: Ridge +0.0%; Random Forest +0.3%; Gradient Boosting -0.0%.
- Variável mais influente: alcohol (importância 0.259 no Random Forest); no modelo linear o maior efeito é alcohol (+0.325 por desvio-padrão).
