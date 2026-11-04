# Etapa 1: Limpeza e pré-processamento (vale para as duas partes)

  3. Corrigir ou remover registros inválidos: type corrompido ou vazio, quality vazio ou fora de 0–10, e a linha vazia.
     Documentar cada decisão.
  4. Tratar ausentes: remover ou imputar, justificando. O enunciado pede isso "se existirem", e aqui existem.
  5. Tratar duplicatas: decidir se remove e justificar. Cuidado com o vazamento entre treino e teste que as duplicatas
     causam.
  6. Normalizar as escalas inconsistentes. Este é o ponto crítico. Descubram o padrão (por exemplo, colunas
     multiplicadas por 10³ em parte das linhas) e corrijam com um critério explícito. Validem comparando as
     distribuições antes e depois com os valores esperados do Wine Quality original.
  7. Detectar outliers (IQR, z-score ou boxplots) e decidir o que fazer com os que sobrarem após a correção de escala.
     Os valores absurdos (pH = 114, volatile acidity = 6600) provavelmente são erro e não outlier legítimo.
  8. Codificar type (red/white → 0/1 ou one-hot).
  9. Separar treino e teste antes de qualquer ajuste. Usar stratify na classificação.
  10. Padronizar com StandardScaler, ajustado só no treino, dentro de um Pipeline.

#  Etapa 2: Análise exploratória

  11. Histogramas das variáveis e de quality.
  12. Scatter plots das features contra quality.
  13. Matriz de correlação (heatmap).
  14. Comparação entre tinto e branco, se fizer sentido.
  15. É recomendável mostrar a EDA antes e depois da limpeza. Isso evidencia o problema e justifica o tratamento.

#  Etapa 3: Parte 1, Regressão

  16. Treinar pelo menos 3 modelos (por exemplo Regressão Linear/Ridge, Random Forest e Gradient Boosting ou SVR).
  17. Avaliar com RMSE, MAE e R², usando validação cruzada.
  18. Fazer o ajuste de hiperparâmetros com GridSearchCV ou RandomizedSearchCV.
  19. Discutir os coeficientes (modelo linear) e a importância das variáveis (modelos de árvore), indicando quais mais
      impactam a qualidade.
  20. Comparar os resultados antes e depois do tuning.

#  Etapa 4: Parte 2, Classificação

  21. Criar a variável alvo categórica por discretização de quality.
  22. Testar diferentes ranges (exigência explícita), por exemplo 2 classes (≤5 vs ≥6) e 3 classes (≤5, 6, ≥7). Comparar
      e justificar o escolhido.
  23. Verificar a distribuição das classes e justificar o balanceamento (class_weight, SMOTE ou under/oversampling),
      aplicado só no treino.
  24. Codificar os rótulos (LabelEncoder ou OneHotEncoder, se necessário).
  25. Treinar pelo menos 3 classificadores (por exemplo Regressão Logística, Random Forest e SVM ou Gradient Boosting).
  26. Escolher e justificar a métrica (F1-macro ou balanced accuracy, por causa do desbalanceamento).
  27. Fazer o ajuste de hiperparâmetros com GridSearchCV e StratifiedKFold.
  28. Analisar os erros com a matriz de confusão: quais classes se confundem, qual modelo foi melhor e quais features
      foram decisivas.