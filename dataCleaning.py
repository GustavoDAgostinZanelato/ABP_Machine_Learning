import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv('.docs/group_10_winequality.csv', sep=',')
df = df_raw.copy()
print(df.shape)
df.info()

print(df.isna().sum())
print(df['type'].value_counts())
print(df['quality'].value_counts().sort_index())
print('Duplicatas:', df.duplicated().sum())
df.describe().T

# alcohol veio como texto: remove os pontos extras e converte para número
def parse_alcohol(v):
    if pd.isna(v): return np.nan
    s = str(v)
    return float(s) if s.count('.') <= 1 else float(s.replace('.', ''))
df['alcohol'] = df['alcohol'].map(parse_alcohol)

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

def corrigir(v, lo, hi, exps):
    if pd.isna(v): return np.nan
    for k in exps:
        w = v * 10.0**k
        if lo <= w <= hi:
            return w
    return np.nan   # valor impossível -> NaN (será imputado depois)

for col, (lo, hi) in LIMITES.items():
    df[col] = df[col].map(lambda v: corrigir(v, lo, hi, EXPOENTES.get(col, [0])))

df.describe().T   # confira: chlorides ~0.05, density ~0.995, etc.

# Linhas com NaN espalhado (linha vazia etc.) são removidas.
# volatile acidity e density ficam com NaN e serão imputadas no Pipeline.
cols_ok = [c for c in df.columns if c not in ('volatile acidity', 'density')]
df = df.dropna(subset=cols_ok)

df = df[df['quality'].between(0, 10)]   # garantia extra contra o 230
df = df.drop_duplicates().reset_index(drop=True)
print(df.shape, df.isna().sum()[lambda s: s > 0].to_dict())


df = df[df['type'].isin(['red', 'white'])].copy()

# Codifica as categorias
df['type'] = df['type'].map({'red': 0, 'white': 1}).astype(int)

# Validação antes de salvar
assert df['type'].isin([0, 1]).all()
assert df['quality'].between(0, 10).all()
print(df.shape, df['type'].value_counts().to_dict())

df.to_csv('.docs/wine_limpo2.csv', index=False)

X = df.drop(columns='quality')
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Imputer e scaler são ajustados SÓ no treino (evita vazamento)
preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])