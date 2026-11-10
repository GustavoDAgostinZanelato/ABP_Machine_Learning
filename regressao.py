"""Etapa 3 - Parte 1: Regressão da qualidade do vinho.

Responde aos itens 16 a 20 do README:
  16. três modelos: Ridge, Random Forest e Gradient Boosting;
  17. RMSE, MAE e R² com validação cruzada 5-fold;
  18. ajuste de hiperparâmetros com GridSearchCV;
  19. discussão dos coeficientes (linear) e da importância das variáveis (árvores);
  20. comparação dos resultados antes e depois do tuning.

Uso:
    python regressao.py            # a partir da raiz do projeto

Reaproveitamento na Parte 2 (classificação):
    from regressao import avaliar_modelo, ajustar_hiperparametros, tabela_metricas
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')          # gera figuras sem precisar de tela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline

from limpezaDados import SEED, carregar_dados

DIR_FIGURAS = 'figures'
DIR_RESULTADOS = 'results'
ARQ_FIG_COMPARACAO = f'{DIR_FIGURAS}/07_comparacao_modelos.png'
ARQ_FIG_IMPORTANCIA = f'{DIR_FIGURAS}/08_importancia_features.png'
ARQ_CSV = f'{DIR_RESULTADOS}/regressao.csv'
ARQ_MD = f'{DIR_RESULTADOS}/regressao.md'
N_FOLDS = 5

# Nome da última etapa do Pipeline (usado nas grades de hiperparâmetros)
ESTIMADOR = 'modelo'

# Métricas do enunciado: as de erro ficam negativas porque o sklearn sempre maximiza
METRICAS = {
    'rmse': 'neg_root_mean_squared_error',
    'mae': 'neg_mean_absolute_error',
    'r2': 'r2',
}


def modelos_base():
    """Instâncias novas dos 3 modelos, com os hiperparâmetros padrão do sklearn."""
    return {
        'Ridge': Ridge(random_state=SEED),
        'Random Forest': RandomForestRegressor(random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=SEED),
    }


def grades_regressao():
    """Grades pequenas: 4 combinações por modelo (~1 min no total)."""
    return {
        'Ridge': {f'{ESTIMADOR}__alpha': [0.1, 1, 10, 100]},
        'Random Forest': {
            f'{ESTIMADOR}__n_estimators': [300],
            f'{ESTIMADOR}__max_depth': [None, 15],
            f'{ESTIMADOR}__min_samples_leaf': [1, 3],
        },
        'Gradient Boosting': {
            f'{ESTIMADOR}__n_estimators': [300],
            f'{ESTIMADOR}__learning_rate': [0.05, 0.1],
            f'{ESTIMADOR}__max_depth': [2, 3],
        },
    }


def _folds(cv=N_FOLDS):
    """int -> KFold embaralhado e reprodutível; qualquer outro valor passa direto."""
    if isinstance(cv, int):
        return KFold(n_splits=cv, shuffle=True, random_state=SEED)
    return cv


def construir_pipeline(preproc, modelo):
    """Encadeia o pré-processamento de carregar_dados() com o estimador.

    O pré-processamento entra como etapa do Pipeline para que a imputação e a
    padronização sejam ajustadas só nos dados de treino de cada fold (sem vazamento).
    """
    return Pipeline([('preproc', preproc), (ESTIMADOR, modelo)])


def avaliar_modelo(modelo, X_train, y_train, X_test=None, y_test=None, cv=N_FOLDS):
    """Métricas de validação cruzada e, opcionalmente, de teste.

    Devolve rmse_cv, rmse_cv_std, mae_cv, mae_cv_std, r2_cv, r2_cv_std e, se
    X_test for informado, rmse_teste, mae_teste, r2_teste.
    """
    scores = cross_validate(modelo, X_train, y_train, cv=_folds(cv),
                            scoring=METRICAS, n_jobs=-1)

    resultado = {}
    for metrica in METRICAS:
        valores = scores[f'test_{metrica}']
        if metrica != 'r2':
            valores = -valores        # desfaz o sinal negativo das métricas de erro
        resultado[f'{metrica}_cv'] = float(valores.mean())
        resultado[f'{metrica}_cv_std'] = float(valores.std())

    if X_test is not None:
        modelo.fit(X_train, y_train)
        predito = modelo.predict(X_test)
        resultado['rmse_teste'] = float(np.sqrt(mean_squared_error(y_test, predito)))
        resultado['mae_teste'] = float(mean_absolute_error(y_test, predito))
        resultado['r2_teste'] = float(r2_score(y_test, predito))

    return resultado


def ajustar_hiperparametros(pipeline, grade, X_train, y_train, cv=N_FOLDS):
    """GridSearchCV usando RMSE como critério; devolve a busca já ajustada."""
    busca = GridSearchCV(pipeline, grade, cv=_folds(cv),
                         scoring='neg_root_mean_squared_error', n_jobs=-1)
    busca.fit(X_train, y_train)
    return busca


def tabela_metricas(resultados):
    """Converte {modelo: dict de métricas} em DataFrame indexado pelo modelo."""
    tabela = pd.DataFrame(resultados).T
    tabela.index.name = 'modelo'
    return tabela.round(4)


def _nomes_features(pipeline, nomes=None):
    """Nomes das colunas usadas pelo modelo.

    O Pipeline começa pelo ImputadorPorTipo, que não registra ``feature_names_in_``,
    então os nomes normalmente vêm do DataFrame de treino (parâmetro ``nomes``).
    """
    if nomes is not None:
        return list(nomes)
    if hasattr(pipeline, 'feature_names_in_'):
        return list(pipeline.feature_names_in_)
    imputador = pipeline.named_steps['preproc'].named_steps.get('imputer')
    if imputador is not None and hasattr(imputador, 'medianas_'):
        return list(imputador.medianas_.columns)
    return [f'f{i}' for i in range(pipeline.steps[-1][1].n_features_in_)]


def _top(serie, top):
    """As N maiores por valor absoluto, do maior para o menor."""
    return serie.reindex(serie.abs().sort_values(ascending=False).index).head(top)


def plotar_comparacao(tabela, caminho):
    """Barras de RMSE, MAE e R² (validação cruzada x teste) por modelo."""
    titulos = {
        'rmse': 'RMSE (menor é melhor)',
        'mae': 'MAE (menor é melhor)',
        'r2': 'R² (maior é melhor)',
    }
    modelos = list(tabela.index)
    posicoes = np.arange(len(modelos))
    largura = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metrica in zip(axes, titulos):
        ax.bar(posicoes - largura / 2, tabela[f'{metrica}_cv'], largura,
               yerr=tabela[f'{metrica}_cv_std'], capsize=3,
               label='Validação cruzada (treino)', color='tab:blue')
        ax.bar(posicoes + largura / 2, tabela[f'{metrica}_teste'], largura,
               label='Teste', color='tab:orange')
        ax.set_xticks(posicoes)
        ax.set_xticklabels(modelos, rotation=12)
        ax.set_title(titulos[metrica])
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)   # folga para a legenda/barra de erro
    axes[0].legend(fontsize=8)
    fig.suptitle('Regressão - comparação dos modelos após o tuning')
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plotar_importancia(pipelines, caminho, nomes=None, top=10):
    """Coeficientes do modelo linear e importância das variáveis de uma árvore.

    Recebe {nome: pipeline já ajustado} e usa o primeiro modelo com ``coef_`` e o
    primeiro com ``feature_importances_``. Passe o dicionário ordenado pelo RMSE
    para que o painel da direita mostre o melhor modelo de árvore.
    """
    linear = arvore = None
    for nome, pipeline in pipelines.items():
        estimador = pipeline.named_steps[ESTIMADOR]
        if linear is None and hasattr(estimador, 'coef_'):
            linear = (nome, pipeline)
        if arvore is None and hasattr(estimador, 'feature_importances_'):
            arvore = (nome, pipeline)

    paineis = []
    if linear is not None:
        nome, pipeline = linear
        coeficientes = pd.Series(np.ravel(pipeline.named_steps[ESTIMADOR].coef_),
                                 index=_nomes_features(pipeline, nomes))
        paineis.append((f'Coeficientes padronizados - {nome}',
                        _top(coeficientes, top), 'coeficiente (sinal)'))
    if arvore is not None:
        nome, pipeline = arvore
        importancias = pd.Series(pipeline.named_steps[ESTIMADOR].feature_importances_,
                                 index=_nomes_features(pipeline, nomes))
        paineis.append((f'Importância das variáveis - {nome}',
                        _top(importancias, top), 'importância'))
    if not paineis:
        return

    fig, axes = plt.subplots(1, len(paineis), figsize=(6.5 * len(paineis), 4.5))
    for ax, (titulo, serie, rotulo) in zip(np.atleast_1d(axes), paineis):
        valores = serie.iloc[::-1]          # maior valor fica no topo do gráfico
        if 'sinal' in rotulo:
            cores = ['tab:red' if valor < 0 else 'tab:blue' for valor in valores]
        else:
            cores = 'tab:green'
        ax.barh(valores.index, valores.values, color=cores)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel(rotulo)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle('O que mais influencia a nota de qualidade')
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _fmt(valor):
    if isinstance(valor, (int, float, np.floating)) and not pd.isna(valor):
        return f'{valor:.4f}'
    return '-' if pd.isna(valor) else str(valor)


def _tabela_md(tabela, rotulo='modelo'):
    """Tabela markdown a partir de um DataFrame (sem depender de tabulate)."""
    colunas = list(tabela.columns)
    linhas = [f'| {rotulo} | ' + ' | '.join(colunas) + ' |',
              '|' + '---|' * (len(colunas) + 1)]
    for nome, linha in tabela.iterrows():
        linhas.append(f'| {nome} | ' + ' | '.join(_fmt(v) for v in linha) + ' |')
    return '\n'.join(linhas)


def _tabela_etapas_md(tabela):
    """Tabela antes x depois (uma linha por etapa e modelo)."""
    colunas = ['rmse_cv', 'rmse_cv_std', 'mae_cv', 'r2_cv', 'rmse_teste', 'mae_teste', 'r2_teste']
    linhas = ['| etapa | modelo | ' + ' | '.join(colunas) + ' |',
              '|' + '---|' * (len(colunas) + 2)]
    for _, linha in tabela.iterrows():
        celulas = [_fmt(linha[coluna]) for coluna in colunas]
        linhas.append(f'| {linha["etapa"]} | {linha["modelo"]} | ' + ' | '.join(celulas) + ' |')
    return '\n'.join(linhas)


def resumo_markdown(tabela, info, caminho):
    """Escreve o resumo da Parte 1 em markdown, pronto para o relatório."""
    texto = [
        '# Etapa 3 - Parte 1: Regressão da qualidade',
        '',
        '## Métricas antes e depois do tuning',
        '',
        _tabela_etapas_md(tabela),
        '',
        'Validação cruzada 5-fold no treino. As colunas de teste foram calculadas uma '
        'única vez, ao final, só para os modelos já ajustados; o modelo escolhido é o '
        'de menor RMSE em validação cruzada.',
        '',
        '## Melhores hiperparâmetros (GridSearchCV)',
        '',
    ]
    for nome, parametros in info['parametros'].items():
        texto.append(f'- **{nome}**: `{parametros}`')
    texto += ['', '## Variáveis mais influentes', '', info['features'], '',
              '## Conclusões', '']
    texto += [f'- {conclusao}' for conclusao in info['conclusoes']]
    texto.append('')

    with open(caminho, 'w', encoding='utf-8') as arquivo:
        arquivo.write('\n'.join(texto))


def main(verbose=True, salvar=True):
    """Executa a Parte 1 e devolve (tabela antes x depois, buscas do GridSearchCV)."""
    def log(msg=''):
        if verbose:
            print(msg)

    def passo(n, titulo):
        log(f'\n[{n}/6] {titulo}')
        log('-' * 70)

    log('=' * 70)
    log('ETAPA 3 - PARTE 1: REGRESSÃO DA QUALIDADE')
    log('=' * 70)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    # ------------------------------------------------------------------ 1. dados
    passo(1, 'Carregando os dados limpos e o pipeline de pré-processamento')
    X_train, X_test, y_train, y_test, preproc = carregar_dados()
    log(f'Treino: {X_train.shape[0]} linhas x {X_train.shape[1]} colunas | '
        f'Teste: {X_test.shape[0]} linhas')
    log(f'Features: {list(X_train.columns)}')
    log(f'Ausentes a imputar - treino: {int(X_train.isna().sum().sum())} | '
        f'teste: {int(X_test.isna().sum().sum())}')
    log(f'quality - treino: de {y_train.min()} a {y_train.max()} (média {y_train.mean():.2f})')
    log('O conjunto de teste só é usado no passo 5: nenhuma decisão é tomada olhando ele.')

    # ------------------------------------------------- 2. modelos sem tuning
    passo(2, 'Métricas com os hiperparâmetros padrão (validação cruzada 5-fold)')
    modelos = modelos_base()
    resultados_padrao = {}
    for nome, modelo in modelos.items():
        resultado = avaliar_modelo(construir_pipeline(preproc, modelo), X_train, y_train)
        resultados_padrao[nome] = resultado
        log(f'{nome:<18} RMSE {resultado["rmse_cv"]:.4f} (+-{resultado["rmse_cv_std"]:.4f}) | '
            f'MAE {resultado["mae_cv"]:.4f} | R² {resultado["r2_cv"]:.4f}')
    tabela_padrao = tabela_metricas(resultados_padrao)

    # -------------------------------------------------------------- 3. tuning
    passo(3, 'Ajustando hiperparâmetros com GridSearchCV (RMSE como critério)')
    buscas = {}
    resultados_ajustados = {}
    for nome, grade in grades_regressao().items():
        inicio = time.time()
        pipeline = construir_pipeline(preproc, modelos[nome])
        busca = ajustar_hiperparametros(pipeline, grade, X_train, y_train)
        buscas[nome] = busca
        resultado = avaliar_modelo(busca.best_estimator_, X_train, y_train, X_test, y_test)
        resultados_ajustados[nome] = resultado
        log(f'{nome:<18} {len(busca.cv_results_["params"])} combinações em '
            f'{time.time() - inicio:5.1f}s')
        log(f'{" " * 18} melhores parâmetros: {busca.best_params_}')
        log(f'{" " * 18} validação cruzada: RMSE {resultado["rmse_cv"]:.4f} '
            f'(+-{resultado["rmse_cv_std"]:.4f}) | MAE {resultado["mae_cv"]:.4f} | '
            f'R² {resultado["r2_cv"]:.4f}')
        log(f'{" " * 18} teste: RMSE {resultado["rmse_teste"]:.4f} | '
            f'MAE {resultado["mae_teste"]:.4f} | R² {resultado["r2_teste"]:.4f}')
    tabela_ajustada = tabela_metricas(resultados_ajustados)

    # ------------------------------------------------ 4. antes x depois (item 20)
    passo(4, 'Comparando os resultados antes e depois do tuning')
    registros = []
    for nome, resultado in resultados_padrao.items():
        registros.append({
            'etapa': 'padrão',
            'modelo': nome,
            **resultado,
            'rmse_teste': np.nan,
            'mae_teste': np.nan,
            'r2_teste': np.nan,
            'melhores_parametros': '-',
        })
    for nome, resultado in resultados_ajustados.items():
        registros.append({
            'etapa': 'ajustado',
            'modelo': nome,
            **resultado,
            'melhores_parametros': str(buscas[nome].best_params_),
        })
    colunas = ['etapa', 'modelo', 'rmse_cv', 'rmse_cv_std', 'mae_cv', 'mae_cv_std',
               'r2_cv', 'r2_cv_std', 'rmse_teste', 'mae_teste', 'r2_teste',
               'melhores_parametros']
    tabela_etapas = pd.DataFrame(registros)[colunas].round(4)
    log(tabela_etapas.drop(columns='melhores_parametros').to_string(index=False, na_rep='-'))

    ganhos = {}
    for nome, resultado in resultados_padrao.items():
        antes = resultado['rmse_cv']
        depois = resultados_ajustados[nome]['rmse_cv']
        ganhos[nome] = 100 * (antes - depois) / antes
        log(f'{nome:<18} RMSE de validação cruzada {antes:.4f} -> {depois:.4f} '
            f'({ganhos[nome]:+.1f}%)')

    # ------------------------------------------- 5. escolha do modelo e figuras
    passo(5, 'Escolhendo o melhor modelo (menor RMSE em validação cruzada)')
    ordem = tabela_ajustada['rmse_cv'].sort_values()
    melhor = ordem.index[0]
    log(f'Melhor modelo: {melhor}')
    log(f'Validação cruzada: RMSE {tabela_ajustada.loc[melhor, "rmse_cv"]:.4f} | '
        f'MAE {tabela_ajustada.loc[melhor, "mae_cv"]:.4f} | '
        f'R² {tabela_ajustada.loc[melhor, "r2_cv"]:.4f}')
    log(f'Teste: RMSE {tabela_ajustada.loc[melhor, "rmse_teste"]:.4f} | '
        f'MAE {tabela_ajustada.loc[melhor, "mae_teste"]:.4f} | '
        f'R² {tabela_ajustada.loc[melhor, "r2_teste"]:.4f}')

    pipelines_ordenados = {nome: buscas[nome].best_estimator_ for nome in ordem.index}
    nomes_features = list(X_train.columns)
    if salvar:
        plotar_comparacao(tabela_ajustada, ARQ_FIG_COMPARACAO)
        plotar_importancia(pipelines_ordenados, ARQ_FIG_IMPORTANCIA, nomes=nomes_features)
        log(f'Figuras salvas: {ARQ_FIG_COMPARACAO}, {ARQ_FIG_IMPORTANCIA}')

    # ------------------------------------------------------ 6. resumo final
    passo(6, 'Interpretando as variáveis e salvando o resumo')
    linear_nome, linear_pipe = next(
        (nome, pipe) for nome, pipe in pipelines_ordenados.items()
        if hasattr(pipe.named_steps[ESTIMADOR], 'coef_'))
    arvore_nome, arvore_pipe = next(
        (nome, pipe) for nome, pipe in pipelines_ordenados.items()
        if hasattr(pipe.named_steps[ESTIMADOR], 'feature_importances_'))

    coeficientes = pd.Series(np.ravel(linear_pipe.named_steps[ESTIMADOR].coef_),
                             index=_nomes_features(linear_pipe, nomes_features))
    importancias = pd.Series(arvore_pipe.named_steps[ESTIMADOR].feature_importances_,
                             index=_nomes_features(arvore_pipe, nomes_features))
    top_coef = _top(coeficientes, 5)
    top_imp = _top(importancias, 5)

    log(f'Coeficientes padronizados ({linear_nome}) - sinal: + aumenta a nota, - reduz')
    log(top_coef.to_string(float_format=lambda v: f'{v:+.3f}'))
    log(f'\nImportância das variáveis ({arvore_nome})')
    log(top_imp.to_string(float_format=lambda v: f'{v:.3f}'))

    texto_features = (
        f'Coeficientes padronizados do {linear_nome} '
        f'(sinal: + aumenta a nota, - reduz):\n\n'
        + _tabela_md(top_coef.to_frame('coeficiente'), rotulo='variável')
        + f'\n\nImportância das variáveis no {arvore_nome} (top 5):\n\n'
        + _tabela_md(top_imp.to_frame('importância'), rotulo='variável')
    )

    conclusoes = [
        f'Melhor modelo: {melhor}, com RMSE {ordem.iloc[0]:.4f} e R² '
        f'{tabela_ajustada.loc[melhor, "r2_cv"]:.4f} em validação cruzada '
        f'(pior: {ordem.index[-1]}, RMSE {ordem.iloc[-1]:.4f}).',
        'Efeito do tuning no RMSE de validação cruzada: '
        + '; '.join(f'{nome} {ganho:+.1f}%' for nome, ganho in ganhos.items()) + '.',
        f'Variável mais influente: {top_imp.index[0]} '
        f'(importância {top_imp.iloc[0]:.3f} no {arvore_nome}); '
        f'no modelo linear o maior efeito é {top_coef.index[0]} '
        f'({top_coef.iloc[0]:+.3f} por desvio-padrão).',
    ]

    informacoes = {
        'parametros': {nome: str(busca.best_params_) for nome, busca in buscas.items()},
        'features': texto_features,
        'conclusoes': conclusoes,
    }

    if salvar:
        tabela_etapas.to_csv(ARQ_CSV, index=False)
        resumo_markdown(tabela_etapas, informacoes, ARQ_MD)
        log(f'\nArquivos salvos: {ARQ_CSV}, {ARQ_MD}')

    log('')
    log('RESUMO')
    log('-' * 70)
    for conclusao in conclusoes:
        log(f'- {conclusao}')
    log('\n' + '=' * 70 + '\nREGRESSÃO CONCLUÍDA\n' + '=' * 70)

    return tabela_etapas, buscas


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')   # acentos corretos no terminal do Windows
    main(verbose=True, salvar=True)
