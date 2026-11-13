"""Etapas 1 a 3 da Parte 2: classificação da qualidade do vinho.

Compara discretizações da qualidade, prepara uma divisão estratificada e
treina três classificadores com o pré-processamento ajustado apenas no treino.

Uso:
    python classificacao.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from limpezaDados import SEED, carregar_dados
from regressao import construir_pipeline

DIR_FIGURAS = 'figures'
DIR_RESULTADOS = 'results'
ARQ_FIG_FAIXAS = f'{DIR_FIGURAS}/09_faixas_classificacao.png'
ARQ_FIG_DISTRIBUICAO = f'{DIR_FIGURAS}/10_distribuicao_classes.png'
ARQ_FIG_METRICAS = f'{DIR_FIGURAS}/11_metricas_classificacao.png'
ARQ_FIG_CONFUSAO = f'{DIR_FIGURAS}/12_matrizes_confusao.png'
ARQ_FIG_IMPORTANCIA = f'{DIR_FIGURAS}/13_importancia_features_classificacao.png'
ARQ_CSV_DISTRIBUICAO = f'{DIR_RESULTADOS}/classificacao_distribuicao.csv'
ARQ_CSV_METRICAS = f'{DIR_RESULTADOS}/classificacao_metricas.csv'
ARQ_CSV_IMPORTANCIA = f'{DIR_RESULTADOS}/classificacao_importancia.csv'
ARQ_MD = f'{DIR_RESULTADOS}/classificacao.md'
N_FOLDS = 5

FAIXAS = {
    '2 classes': {
        'bins': [0, 5, 10],
        'labels': ['Baixa (0-5)', 'Alta (6-10)'],
    },
    '3 classes': {
        'bins': [0, 5, 6, 10],
        'labels': ['Baixa (0-5)', 'Média (6)', 'Alta (7-10)'],
    },
}
FAIXA_ESCOLHIDA = '3 classes'


def discretizar(quality, definicao):
    """Converte quality em faixas com limites inclusivos."""
    faixa = FAIXAS[definicao]
    return pd.cut(
        quality,
        bins=faixa['bins'],
        labels=faixa['labels'],
        include_lowest=True,
        right=True,
    )


def distribuicao_classes(quality):
    """Monta as contagens e proporções para cada discretização candidata."""
    tabelas = []
    for nome in FAIXAS:
        classes = discretizar(quality, nome)
        contagens = classes.value_counts(sort=False)
        for rotulo, quantidade in contagens.items():
            tabelas.append({
                'faixa': nome,
                'classe': str(rotulo),
                'quantidade': int(quantidade),
                'proporcao': float(quantidade / len(quality)),
            })
    return pd.DataFrame(tabelas)


def plotar_comparacao_faixas(tabela, caminho):
    """Compara as proporções das classes nas discretizações candidatas."""
    faixas = list(FAIXAS)
    fig, axes = plt.subplots(1, len(faixas), figsize=(12, 4))
    for ax, faixa in zip(axes, faixas):
        dados = tabela[tabela['faixa'] == faixa]
        ax.bar(dados['classe'], dados['proporcao'], color='tab:blue')
        ax.set_title(f'Distribuição - {faixa}')
        ax.set_ylabel('Proporção')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Comparação das faixas de quality')
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plotar_distribuicao_treino_teste(y_train, y_test, nomes_classes, caminho):
    """Exibe as contagens por classe após a divisão estratificada."""
    indices = np.arange(len(nomes_classes))
    largura = 0.38
    treino = y_train.value_counts().reindex(indices, fill_value=0)
    teste = y_test.value_counts().reindex(indices, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(indices - largura / 2, treino, largura, label='Treino')
    ax.bar(indices + largura / 2, teste, largura, label='Teste')
    ax.set_xticks(indices)
    ax.set_xticklabels(nomes_classes, rotation=10)
    ax.set_ylabel('Quantidade de amostras')
    ax.set_title('Distribuição das classes - treino e teste')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def modelos_classificacao():
    """Cria três classificadores com pesos balanceados por classe."""
    return {
        'Regressão Logística': LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=SEED,
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced',
            n_estimators=300,
            n_jobs=-1,
            random_state=SEED,
        ),
        'SVM (RBF)': SVC(
            class_weight='balanced',
            kernel='rbf',
        ),
    }


def grades_classificacao():
    """Grades pequenas de hiperparâmetros para os três classificadores."""
    return {
        'Regressão Logística': {'modelo__C': [0.1, 1, 10]},
        'Random Forest': {
            'modelo__n_estimators': [200, 300],
            'modelo__max_depth': [None, 20],
            'modelo__min_samples_leaf': [1, 3],
        },
        'SVM (RBF)': {
            'modelo__C': [0.5, 1, 5],
            'modelo__gamma': ['scale', 'auto'],
        },
    }


def _avaliar_cv(pipeline, X_train, y_train, cv):
    """Retorna F1-macro e balanced accuracy médias e desvios na validação."""
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            'f1_macro': 'f1_macro',
            'balanced_accuracy': 'balanced_accuracy',
        },
        n_jobs=-1,
    )
    return {
        f'{metrica}_cv': float(scores[f'test_{metrica}'].mean())
        for metrica in ('f1_macro', 'balanced_accuracy')
    } | {
        f'{metrica}_cv_std': float(scores[f'test_{metrica}'].std())
        for metrica in ('f1_macro', 'balanced_accuracy')
    }


def plotar_metricas(tabela, caminho):
    """Compara as métricas de validação cruzada e teste após o tuning."""
    metricas = (
        ('f1_macro', 'F1-macro (maior é melhor)'),
        ('balanced_accuracy', 'Balanced accuracy (maior é melhor)'),
    )
    fig, axes = plt.subplots(1, len(metricas), figsize=(12, 4.5))
    posicoes = np.arange(len(tabela))
    largura = 0.38
    for ax, (metrica, titulo) in zip(axes, metricas):
        ax.bar(
            posicoes - largura / 2,
            tabela[f'{metrica}_cv'],
            largura,
            yerr=tabela[f'{metrica}_cv_std'],
            capsize=3,
            label='Validação cruzada (treino)',
        )
        ax.bar(
            posicoes + largura / 2,
            tabela[f'{metrica}_teste'],
            largura,
            label='Teste',
        )
        ax.set_xticks(posicoes)
        ax.set_xticklabels(tabela['modelo'], rotation=15)
        ax.set_ylim(0, 1)
        ax.set_title(titulo)
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Classificação - desempenho dos modelos ajustados')
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plotar_matrizes_confusao(matrizes, nomes_classes, caminho):
    """Plota as matrizes de confusão normalizadas por classe real."""
    fig, axes = plt.subplots(1, len(matrizes), figsize=(15, 4.5))
    cax = fig.add_axes([0.92, 0.22, 0.015, 0.6])   # eixo próprio: a barra não cobre nenhuma matriz
    for ax, (nome, matriz) in zip(axes, matrizes.items()):
        normalizada = matriz / matriz.sum(axis=1, keepdims=True)
        imagem = ax.imshow(normalizada, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(nome)
        ax.set_xticks(range(len(nomes_classes)))
        ax.set_yticks(range(len(nomes_classes)))
        ax.set_xticklabels(nomes_classes, rotation=25, ha='right')
        ax.set_yticklabels(nomes_classes)
        ax.set_xlabel('Classe predita')
        ax.set_ylabel('Classe real')
        for linha in range(len(nomes_classes)):
            for coluna in range(len(nomes_classes)):
                ax.text(
                    coluna,
                    linha,
                    f'{matriz[linha, coluna]}\n{normalizada[linha, coluna]:.0%}',
                    ha='center',
                    va='center',
                    color='white' if normalizada[linha, coluna] > 0.5 else 'black',
                    fontsize=8,
                )
    fig.colorbar(imagem, cax=cax, label='Proporção por classe real')
    fig.suptitle('Matrizes de confusão no conjunto de teste (contagem e proporção)')
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.22, top=0.82, wspace=0.55)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def plotar_importancia(importancias, caminho):
    """Plota as variáveis mais importantes por permutação no modelo escolhido."""
    dados = importancias.sort_values('importancia_media', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        dados['variavel'],
        dados['importancia_media'],
        xerr=dados['importancia_std'],
        color='tab:purple',
        capsize=3,
    )
    ax.set_xlabel('Redução média do F1-macro ao embaralhar a variável')
    ax.set_title('Importância por permutação - melhor modelo')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)


def _par_mais_confundido(matriz, nomes_classes):
    """Retorna o maior erro fora da diagonal e sua proporção na classe real."""
    erros = matriz.copy()
    np.fill_diagonal(erros, 0)
    real, predita = np.unravel_index(np.argmax(erros), erros.shape)
    proporcao = erros[real, predita] / matriz[real].sum()
    return nomes_classes[real], nomes_classes[predita], int(erros[real, predita]), float(proporcao)


def _resumo_markdown(tabela_faixas, distribuicao_treino, distribuicao_teste,
                     nomes_classes, tabela_metricas, matrizes, melhor_modelo,
                     relatorios_classes, importancias, parametros, caminho):
    """Salva a preparação, avaliação, erros e importância em Markdown."""
    contagem_2 = tabela_faixas[tabela_faixas['faixa'] == '2 classes'].set_index('classe')['quantidade']
    contagem_3 = tabela_faixas[tabela_faixas['faixa'] == '3 classes'].set_index('classe')['quantidade']
    razao_2 = contagem_2.max() / contagem_2.min()
    razao_3 = contagem_3.max() / contagem_3.min()
    linhas = [
        '# Parte 2 - Classificação',
        '',
        '## Etapa 1 - Comparação das discretizações',
        '',
        '| Faixa | Classe | Quantidade | Proporção |',
        '|---|---|---:|---:|',
    ]
    for _, linha in tabela_faixas.iterrows():
        linhas.append(
            f'| {linha["faixa"]} | {linha["classe"]} | '
            f'{linha["quantidade"]} | {linha["proporcao"]:.1%} |'
        )
    linhas += [
        '',
        'Foi escolhida a discretização em três classes (quality 0-5, 6 e 7-10). '
        'A nota 6 é a moda do dataset e concentra a maior parte dos vinhos, então '
        'tratá-la como classe própria evita misturar vinhos medianos com os de '
        'qualidade baixa ou alta. '
        f'A razão entre a maior e a menor classe é {razao_3:.2f}:1, contra '
        f'{razao_2:.2f}:1 na opção binária: o desbalanceamento é maior, mas '
        'moderado, e todas as classes mantêm amostras suficientes para uma divisão '
        'estratificada. O desbalanceamento é tratado com `class_weight="balanced"` '
        'em vez de reamostragem, que preserva todos os dados reais e evita '
        'vazamento na validação cruzada.',
        '',
        '## Etapa 2 - Preparação dos dados',
        '',
        f'Divisão estratificada 80/20 com `random_state={SEED}`. '
        'A imputação e a padronização são mantidas dentro do pipeline e ajustadas '
        'somente nos dados de treino.',
        '',
        'O `LabelEncoder` converte as categorias em códigos inteiros:',
        '',
        '| Código | Classe | Treino | Teste |',
        '|---:|---|---:|---:|',
    ]
    for codigo, nome in enumerate(nomes_classes):
        linhas.append(
            f'| {codigo} | {nome} | '
            f'{int(distribuicao_treino.get(codigo, 0))} | '
            f'{int(distribuicao_teste.get(codigo, 0))} |'
        )
    linhas += [
        '',
        'Foi usado `class_weight="balanced"` nos três classificadores. '
        'Essa ponderação considera a frequência de cada classe durante o treino, '
        'sem reamostrar o conjunto de teste.',
        '',
        '## Etapa 3 - Modelagem e avaliação',
        '',
    ]
    linhas += [
        '| Modelo | F1 CV padrão | F1 CV ajustado | F1 teste | '
        'Balanced accuracy CV padrão | Balanced accuracy CV ajustado | '
        'Balanced accuracy teste |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for _, linha in tabela_metricas.iterrows():
        linhas.append(
            f'| {linha["modelo"]} | {linha["f1_macro_cv_padrao"]:.4f} | '
            f'{linha["f1_macro_cv"]:.4f} ± {linha["f1_macro_cv_std"]:.4f} | '
            f'{linha["f1_macro_teste"]:.4f} | '
            f'{linha["balanced_accuracy_cv_padrao"]:.4f} | '
            f'{linha["balanced_accuracy_cv"]:.4f} ± '
            f'{linha["balanced_accuracy_cv_std"]:.4f} | '
            f'{linha["balanced_accuracy_teste"]:.4f} |'
        )
    linhas += [
        '',
        'A seleção de hiperparâmetros usou GridSearchCV com F1-macro e '
        f'StratifiedKFold de {N_FOLDS} folds (embaralhado, `random_state={SEED}`). '
        'A validação cruzada foi realizada somente no conjunto de treino; o conjunto '
        'de teste foi reservado para a avaliação final.',
        '',
        '### Melhores hiperparâmetros',
        '',
    ]
    for nome, valores in parametros.items():
        linhas.append(f'- **{nome}**: `{valores}`')
    linhas += [
        '',
        f'Melhor modelo pela validação cruzada: **{melhor_modelo}** '
        f'(F1-macro {tabela_metricas.loc[tabela_metricas["modelo"] == melhor_modelo, "f1_macro_cv"].iloc[0]:.4f}).',
        '',
        '### Matrizes de confusão no teste',
        '',
        'Linhas representam a classe real e colunas a classe predita; cada célula '
        'mostra contagem e proporção dentro da classe real.',
        '',
    ]
    for nome, matriz in matrizes.items():
        linhas += [
            f'#### {nome}',
            '',
            '| Real \\ Predita | ' + ' | '.join(nomes_classes) + ' |',
            '|' + '---|' * (len(nomes_classes) + 1),
        ]
        for indice, classe in enumerate(nomes_classes):
            celulas = ' | '.join(
                f'{matriz[indice, coluna]} '
                f'({matriz[indice, coluna] / matriz[indice].sum():.1%})'
                for coluna in range(len(nomes_classes))
            )
            linhas.append(f'| {classe} | {celulas} |')
        real, predita, quantidade, proporcao = _par_mais_confundido(matriz, nomes_classes)
        linhas += [
            '',
            f'Confusão mais frequente: **{real}** prevista como **{predita}** '
            f'({quantidade} casos; {proporcao:.1%} da classe real).',
            '',
        ]
    linhas += [
        f'### Relatório por classe - {melhor_modelo}',
        '',
        '| Classe | Precisão | Recall | F1-score | Suporte |',
        '|---|---:|---:|---:|---:|',
    ]
    for classe in nomes_classes:
        metricas_classe = relatorios_classes[classe]
        linhas.append(
            f'| {classe} | {metricas_classe["precision"]:.4f} | '
            f'{metricas_classe["recall"]:.4f} | {metricas_classe["f1-score"]:.4f} | '
            f'{int(metricas_classe["support"])} |'
        )
    linhas += [
        '',
        f'### Variáveis mais decisivas - {melhor_modelo}',
        '',
        'Importância por permutação medida no teste: quanto maior a queda do '
        'F1-macro ao embaralhar uma variável, maior sua contribuição para as previsões.',
        '',
        '| Variável | Redução média do F1-macro | Desvio padrão |',
        '|---|---:|---:|',
    ]
    for _, linha in importancias.sort_values('importancia_media', ascending=False).iterrows():
        linhas.append(
            f'| {linha["variavel"]} | {linha["importancia_media"]:.5f} | '
            f'{linha["importancia_std"]:.5f} |'
        )
    if not importancias.empty:
        linhas += [
            '',
            f'Variável mais decisiva: **{importancias.iloc[0]["variavel"]}** '
            f'(queda média de F1-macro {importancias.iloc[0]["importancia_media"]:.5f}).',
        ]
    linhas += [
        '',
        'A importância por permutação é uma análise descritiva no conjunto de teste; '
        'não foi usada para escolher o modelo nem ajustar hiperparâmetros.',
        '',
    ]
    with open(caminho, 'w', encoding='utf-8') as arquivo:
        arquivo.write('\n'.join(linhas))


def main(verbose=True, salvar=True):
    """Compara faixas, prepara, ajusta e avalia os classificadores."""
    def log(mensagem=''):
        if verbose:
            print(mensagem)

    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(DIR_RESULTADOS, exist_ok=True)

    X_train_original, X_test_original, y_train_original, y_test_original, preproc = (
        carregar_dados()
    )
    X = pd.concat([X_train_original, X_test_original]).sort_index()
    quality = pd.concat([y_train_original, y_test_original]).sort_index()
    tabela = distribuicao_classes(quality)

    log('=' * 70)
    log('PARTE 2 - CLASSIFICAÇÃO: ETAPAS 1 A 3')
    log('=' * 70)
    log('\n[1/3] Comparando faixas de qualidade')
    log(tabela.to_string(index=False, formatters={'proporcao': '{:.1%}'.format}))
    log(f'\nFaixa escolhida: {FAIXA_ESCOLHIDA}')

    alvo_texto = discretizar(quality, FAIXA_ESCOLHIDA)
    codificador = LabelEncoder()
    # classes_ na ordem natural (Baixa < Média < Alta); o fit alfabético a embaralharia
    codificador.classes_ = np.array(FAIXAS[FAIXA_ESCOLHIDA]['labels'])
    alvo = pd.Series(
        codificador.transform(alvo_texto.astype(str)),
        index=alvo_texto.index,
        name='classe',
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        alvo,
        test_size=0.2,
        random_state=SEED,
        stratify=alvo,
    )
    distribuicao_treino = y_train.value_counts().sort_index()
    distribuicao_teste = y_test.value_counts().sort_index()
    nomes_classes = list(codificador.classes_)

    log('\n[2/3] Preparando os dados')
    log(f'Treino: {len(X_train)} | teste: {len(X_test)}')
    log('Codificacao: ' + ', '.join(
        f'{codigo}={nome}' for codigo, nome in enumerate(nomes_classes)
    ))
    log('Classes no treino: ' + str(distribuicao_treino.to_dict()))
    log('Classes no teste: ' + str(distribuicao_teste.to_dict()))
    log('Balanceamento: class_weight="balanced"; sem reamostragem do teste.')

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    modelos = modelos_classificacao()
    resultados = []
    buscas = {}
    matrizes = {}
    pipelines = {}
    log(f'\n[3/3] Validando, ajustando e avaliando com StratifiedKFold {N_FOLDS}-fold')
    for nome, modelo in modelos.items():
        log(f'\n{nome}: avaliação inicial e GridSearchCV')
        base = construir_pipeline(preproc, modelo)
        metricas_base = _avaliar_cv(base, X_train, y_train, cv)
        busca = GridSearchCV(
            construir_pipeline(preproc, modelo),
            grades_classificacao()[nome],
            scoring='f1_macro',
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        busca.fit(X_train, y_train)
        metricas_ajustadas = _avaliar_cv(busca.best_estimator_, X_train, y_train, cv)
        predito = busca.best_estimator_.predict(X_test)
        metricas_teste = {
            'f1_macro_teste': float(f1_score(y_test, predito, average='macro')),
            'balanced_accuracy_teste': float(balanced_accuracy_score(y_test, predito)),
        }
        resultados.append({
            'modelo': nome,
            **{f'{chave}_padrao': valor for chave, valor in metricas_base.items()},
            **metricas_ajustadas,
            **metricas_teste,
            'melhores_parametros': str(busca.best_params_),
        })
        buscas[nome] = busca
        pipelines[nome] = busca.best_estimator_
        matrizes[nome] = confusion_matrix(
            y_test, predito, labels=np.arange(len(nomes_classes))
        )
        log(f'  melhores parâmetros: {busca.best_params_}')
        log(f'  F1-macro CV: {metricas_ajustadas["f1_macro_cv"]:.4f} | '
            f'teste: {metricas_teste["f1_macro_teste"]:.4f}')
        log(f'  Balanced accuracy CV: {metricas_ajustadas["balanced_accuracy_cv"]:.4f} | '
            f'teste: {metricas_teste["balanced_accuracy_teste"]:.4f}')

    tabela_metricas = pd.DataFrame(resultados).sort_values(
        'f1_macro_cv', ascending=False
    ).reset_index(drop=True)
    melhor_modelo = tabela_metricas.iloc[0]['modelo']
    melhor_pipeline = pipelines[melhor_modelo]
    melhor_predito = melhor_pipeline.predict(X_test)
    nomes_features = list(X_train.columns)
    permutacao = permutation_importance(
        melhor_pipeline,
        X_test,
        y_test,
        scoring='f1_macro',
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )
    importancias = pd.DataFrame({
        'variavel': nomes_features,
        'importancia_media': permutacao.importances_mean,
        'importancia_std': permutacao.importances_std,
    }).sort_values('importancia_media', ascending=False).reset_index(drop=True)
    relatorios_classes = pd.DataFrame(
        classification_report(
            y_test,
            melhor_predito,
            labels=np.arange(len(nomes_classes)),
            target_names=nomes_classes,
            output_dict=True,
            zero_division=0,
        )
    ).T.loc[nomes_classes].to_dict(orient='index')

    log('\nResumo dos modelos ajustados:')
    log(tabela_metricas[
        ['modelo', 'f1_macro_cv', 'f1_macro_teste',
         'balanced_accuracy_cv', 'balanced_accuracy_teste']
    ].to_string(index=False, float_format=lambda valor: f'{valor:.4f}'))
    log(f'\nMelhor modelo em validação cruzada: {melhor_modelo}')
    log('Variáveis mais decisivas: ' + ', '.join(importancias.head(5)['variavel']))

    if salvar:
        plotar_comparacao_faixas(tabela, ARQ_FIG_FAIXAS)
        plotar_distribuicao_treino_teste(
            y_train, y_test, nomes_classes, ARQ_FIG_DISTRIBUICAO
        )
        plotar_metricas(tabela_metricas, ARQ_FIG_METRICAS)
        plotar_matrizes_confusao(matrizes, nomes_classes, ARQ_FIG_CONFUSAO)
        plotar_importancia(importancias.head(10), ARQ_FIG_IMPORTANCIA)
        tabela.to_csv(ARQ_CSV_DISTRIBUICAO, index=False)
        tabela_metricas.to_csv(ARQ_CSV_METRICAS, index=False)
        importancias.to_csv(ARQ_CSV_IMPORTANCIA, index=False)
        _resumo_markdown(
            tabela,
            distribuicao_treino,
            distribuicao_teste,
            nomes_classes,
            tabela_metricas,
            matrizes,
            melhor_modelo,
            relatorios_classes,
            importancias,
            {nome: busca.best_params_ for nome, busca in buscas.items()},
            ARQ_MD,
        )
        log(f'\nResultados salvos: {ARQ_CSV_DISTRIBUICAO}, {ARQ_CSV_METRICAS}, '
            f'{ARQ_CSV_IMPORTANCIA}, {ARQ_MD}')
        log(f'Figuras salvas: {ARQ_FIG_FAIXAS}, {ARQ_FIG_DISTRIBUICAO}, '
            f'{ARQ_FIG_METRICAS}, {ARQ_FIG_CONFUSAO}, {ARQ_FIG_IMPORTANCIA}')
    log('\nCLASSIFICAÇÃO CONCLUÍDA')
    return tabela_metricas, buscas


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main(verbose=True, salvar=True)
