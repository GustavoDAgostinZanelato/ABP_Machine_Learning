"""Etapa 1 - Limpeza e pré-processamento (vale para as Partes 1 e 2).

Uso nas outras etapas:
    from dataCleaning import carregar_dados
    X_train, X_test, y_train, y_test, preproc = carregar_dados()

Execução direta (`python dataCleaning.py`) roda com logs detalhados e salva os CSVs.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARQUIVO_BRUTO = '.docs/group_10_winequality.csv'
ARQUIVO_LIMPO = '.docs/wine_limpo.csv'
ARQUIVO_TREINO = '.docs/wine_treino.csv'
ARQUIVO_TESTE = '.docs/wine_teste.csv'
SEED = 42

# Faixa físico-química plausível de cada variável
LIMITES = {
    'fixed acidity': (3.5, 16),  'volatile acidity': (0.07, 1.7), 'citric acid': (0, 1.7),
    'residual sugar': (0.5, 70), 'chlorides': (0.008, 0.7),
    'free sulfur dioxide': (1, 300), 'total sulfur dioxide': (5, 450),
    'density': (0.985, 1.04),    'pH': (2.7, 4.1),
    'sulphates': (0.2, 2.1),     'alcohol': (8, 15),
}
# Potências de 10 permitidas por coluna (as demais só aceitam o valor original)
EXPOENTES = {
    'volatile acidity': [0, -3],
    'chlorides': [0, -3],
    'density': [0, -1, -2, -3],
    'alcohol': list(range(0, -15, -1)),
}
FEATURES = list(LIMITES)


class ImputadorPorTipo(BaseEstimator, TransformerMixin):
    """Imputa ausentes com a mediana de cada tipo de vinho (tinto/branco).

    Tinto e branco têm perfis muito diferentes (ex.: volatile acidity 0,53 vs 0,28);
    a mediana global puxaria os tintos para baixo. As medianas são aprendidas no fit,
    ou seja, só com o treino.
    """

    def fit(self, X, y=None):
        self.medianas_ = X.groupby('type').median()
        self.mediana_global_ = X.median()
        return self

    def transform(self, X):
        X = X.copy()
        for tipo, medianas in self.medianas_.iterrows():
            linhas = X['type'] == tipo
            X.loc[linhas] = X.loc[linhas].fillna(medianas)
        return X.fillna(self.mediana_global_)   # tipo não visto no treino (não ocorre aqui)


def parse_alcohol(v):
    """alcohol veio como texto: remove os pontos extras ('10.033.333' -> 10033333)."""
    if pd.isna(v):
        return np.nan
    s = str(v)
    return float(s) if s.count('.') <= 1 else float(s.replace('.', ''))


def corrigir(v, lo, hi, exps):
    """Testa as potências de 10 permitidas e devolve o primeiro valor dentro da faixa."""
    if pd.isna(v):
        return np.nan
    for k in exps:
        w = v * 10.0**k
        if lo <= w <= hi:
            return w
    return np.nan   # valor impossível -> NaN


def carregar_dados(verbose=False, salvar=False):
    """Executa toda a limpeza e devolve X_train, X_test, y_train, y_test e o pipeline."""
    def log(msg=''):
        if verbose:
            print(msg)

    def passo(n, total, titulo):
        log(f'\n[{n}/{total}] {titulo}')
        log('-' * 70)

    TOTAL = 10

    # ---------------------------------------------------------------- 1. diagnóstico
    log('=' * 70)
    log('ETAPA 1 - LIMPEZA E PRÉ-PROCESSAMENTO')
    log('=' * 70)
    passo(1, TOTAL, f'Carregando e diagnosticando o dataset bruto ({ARQUIVO_BRUTO})')
    df = pd.read_csv(ARQUIVO_BRUTO, sep=',')
    n_bruto = len(df)
    log(f'Dimensão: {df.shape[0]} linhas x {df.shape[1]} colunas')
    resumo = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'ausentes': df.isna().sum(),
        'min': pd.to_numeric(df.drop(columns='type').stack(), errors='coerce')
                 .groupby(level=1).min(),
        'max': pd.to_numeric(df.drop(columns='type').stack(), errors='coerce')
                 .groupby(level=1).max(),
    })
    log(resumo.to_string(float_format=lambda x: f'{x:,.3f}'))
    log(f'\nColunas lidas como texto: {[c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]}')
    log(f'Valores de type: {df["type"].value_counts(dropna=False).rename(lambda v: str(v)[:12]).to_dict()}')
    log(f'Valores de quality: {df["quality"].value_counts(dropna=False).sort_index().to_dict()}')
    log(f'Linhas duplicadas: {df.duplicated().sum()}')

    # ---------------------------------------------------------------- 2. alcohol
    passo(2, TOTAL, 'Convertendo "alcohol" de texto para número')
    n_texto = (df['alcohol'].astype(str).str.count(r'\.') > 1).sum()
    df['alcohol'] = df['alcohol'].map(parse_alcohol)
    log(f'{n_texto} valores com pontos extras (ex.: "100.333.333") foram reinterpretados')

    # ---------------------------------------------------------------- 3. escala
    passo(3, TOTAL, 'Corrigindo escalas inconsistentes (potências de 10)')
    log(f'{"coluna":<22}{"corrigidos":>11}{"invalidos->NaN":>16}   expoentes testados')
    for col, (lo, hi) in LIMITES.items():
        antes = df[col]
        depois = antes.map(lambda v: corrigir(v, lo, hi, EXPOENTES.get(col, [0])))
        corrigidos = (antes.notna() & depois.notna() & ((antes - depois).abs() > 1e-12)).sum()
        invalidos = (antes.notna() & depois.isna()).sum()
        df[col] = depois
        exps = EXPOENTES.get(col, [0])
        exps_txt = f'{exps}' if len(exps) <= 4 else f'0 a {exps[-1]}'
        log(f'{col:<22}{corrigidos:>11}{invalidos:>16}   {exps_txt}  faixa [{lo}, {hi}]')

    # ---------------------------------------------------------------- 4. inválidos
    passo(4, TOTAL, 'Removendo registros inválidos')
    n0 = len(df)
    df['type'] = df['type'].map({'red': 0, 'white': 1})   # corrompido/vazio vira NaN
    df = df[df['type'].notna()]
    log(f'type corrompido ou vazio : {n0 - len(df):>4} linha(s) removida(s)')

    n0 = len(df)
    df = df[df['quality'].between(0, 10)]   # NaN e valores como 230 ficam de fora
    log(f'quality vazia ou fora de 0-10: {n0 - len(df):>4} linha(s) removida(s)')

    # volatile acidity e density têm muitos ausentes: são imputadas no Pipeline
    imputaveis = ['volatile acidity', 'density']
    n0 = len(df)
    df = df.dropna(subset=[c for c in FEATURES if c not in imputaveis])
    log(f'NaN em outras features   : {n0 - len(df):>4} linha(s) removida(s) '
        f'(valor ausente ou impossível)')
    log(f'Ausentes mantidos para imputação (por tipo, só no treino): '
        f'{df[imputaveis].isna().sum().to_dict()}')

    # ---------------------------------------------------------------- 5. duplicatas
    passo(5, TOTAL, 'Removendo duplicatas')
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log(f'{n0 - len(df)} duplicata(s) removida(s). Feito ANTES do split, para que a mesma '
        f'linha não caia em treino e teste (vazamento).')

    # ---------------------------------------------------------------- 6. codificação
    passo(6, TOTAL, 'Codificando "type" e validando')
    df['type'] = df['type'].astype(int)
    df['quality'] = df['quality'].astype(int)
    assert df['type'].isin([0, 1]).all()
    assert df['quality'].between(0, 10).all()
    log(f'type: red -> 0, white -> 1  | contagem: {df["type"].value_counts().sort_index().to_dict()}')
    log(f'Validações OK (type em {{0,1}}, quality em 0-10)')

    # ---------------------------------------------------------------- 7. outliers
    passo(7, TOTAL, 'Detectando outliers (IQR 1.5x) - apenas relatório')
    q1, q3 = df[FEATURES].quantile(0.25), df[FEATURES].quantile(0.75)
    iqr = q3 - q1
    mask_out = (df[FEATURES] < q1 - 1.5 * iqr) | (df[FEATURES] > q3 + 1.5 * iqr)
    for col, n in mask_out.sum().items():
        log(f'{col:<22}{n:>5} ({n / len(df):5.1%})')
    log('Mantidos: os valores absurdos já viraram NaN na etapa 3; o que resta é variação '
        'legítima do vinho.')

    # ---------------------------------------------------------------- 8. arredondamento
    passo(8, TOTAL, 'Arredondando para 4 casas decimais')
    df[FEATURES] = df[FEATURES].round(4)
    log('Remove lixo de ponto flutuante da correção de escala (0.07100000000000001 -> 0.071).')

    # ---------------------------------------------------------------- 9. salvar
    passo(9, TOTAL, 'Resumo da limpeza')
    log(f'Linhas: {n_bruto} (bruto) -> {len(df)} (limpo)  | removidas: {n_bruto - len(df)} '
        f'({(n_bruto - len(df)) / n_bruto:.1%})')
    X = df.drop(columns='quality')
    y = df['quality']

    # ---------------------------------------------------------------- 10. split + pipeline
    passo(10, TOTAL, 'Separando treino/teste e montando o Pipeline')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    log(f'Split 80/20, random_state={SEED}, stratify=quality: '
        f'treino={len(X_train)}  teste={len(X_test)}')
    preproc = Pipeline([
        ('imputer', ImputadorPorTipo()),
        ('scaler', StandardScaler()),
    ])
    log('Pipeline: ImputadorPorTipo (mediana por type) -> StandardScaler. '
        'Deve ser ajustado (fit) SÓ no treino, dentro do Pipeline do modelo.')

    if salvar:
        df.to_csv(ARQUIVO_LIMPO, index=False)
        X_train.assign(quality=y_train).to_csv(ARQUIVO_TREINO, index=False)
        X_test.assign(quality=y_test).to_csv(ARQUIVO_TESTE, index=False)
        log(f'\nArquivos salvos: {ARQUIVO_LIMPO}, {ARQUIVO_TREINO}, {ARQUIVO_TESTE}')
    log('\n' + '=' * 70 + '\nLIMPEZA CONCLUÍDA\n' + '=' * 70)

    return X_train, X_test, y_train, y_test, preproc


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')   # acentos corretos no terminal do Windows
    carregar_dados(verbose=True, salvar=True)
