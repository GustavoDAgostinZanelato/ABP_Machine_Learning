import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

os.makedirs('figures', exist_ok=True)

# Antes: dado bruto (alcohol só convertido para número, sem nenhuma correção de escala)
raw = pd.read_csv('.docs/group_10_winequality.csv', sep=',')
raw['alcohol'] = pd.to_numeric(raw['alcohol'], errors='coerce')
# Depois: saída do dataCleaning.py
clean = pd.read_csv('.docs/wine_limpo.csv')

FEATURES = [c for c in clean.columns if c not in ('type', 'quality')]


def histogramas(df, titulo, arquivo, log=False):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, col in zip(axes.flat, FEATURES + ['quality']):
        sns.histplot(df[col].dropna(), bins=40, ax=ax, log_scale=log and col != 'quality')
        ax.set_title(col)
        ax.set_xlabel('')
        if log and col != 'quality':
            # só potências de 10 rotuladas, sem os rótulos menores sobrepostos
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat[len(FEATURES) + 1:]:
        ax.axis('off')
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(f'figures/{arquivo}', dpi=120)
    plt.close(fig)


# 11. Histogramas (escala log no bruto, senão os valores absurdos esmagam o gráfico)
histogramas(raw, 'Histogramas - ANTES da limpeza (eixo x em escala log)', '01_hist_antes.png', log=True)
histogramas(clean, 'Histogramas - DEPOIS da limpeza', '02_hist_depois.png')

# 15. Boxplots antes x depois, com cada variável na sua própria escala
fig, axes = plt.subplots(2, len(FEATURES), figsize=(24, 7))
for j, col in enumerate(FEATURES):
    sns.boxplot(y=raw[col].dropna(), ax=axes[0, j])
    sns.boxplot(y=clean[col], ax=axes[1, j])
    axes[0, j].set_title(col, fontsize=9)
    for i in (0, 1):
        axes[i, j].set_ylabel('antes' if (j == 0 and i == 0) else 'depois' if j == 0 else '')
fig.tight_layout()
fig.savefig('figures/03_boxplots_antes_depois.png', dpi=120)
plt.close(fig)

# 12. Scatter plots das features contra quality (com jitter, pois quality é inteira)
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
rng = np.random.default_rng(42)
for ax, col in zip(axes.flat, FEATURES):
    jitter = rng.uniform(-0.2, 0.2, len(clean))
    ax.scatter(clean[col], clean['quality'] + jitter, s=4, alpha=0.3,
               c=clean['type'].map({0: 'firebrick', 1: 'goldenrod'}))
    ax.set_xlabel(col)
    ax.set_ylabel('quality')
for ax in axes.flat[len(FEATURES):]:
    ax.axis('off')
fig.suptitle('Features x quality (vermelho = tinto, amarelo = branco)')
fig.tight_layout()
fig.savefig('figures/04_scatter_quality.png', dpi=120)
plt.close(fig)

# 13. Matriz de correlação antes x depois
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, df, t in ((axes[0], raw, 'ANTES'), (axes[1], clean, 'DEPOIS')):
    cols = FEATURES + ['quality']
    sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar=False)
    ax.set_title(f'Correlação - {t} da limpeza')
fig.tight_layout()
fig.savefig('figures/05_correlacao.png', dpi=120)
plt.close(fig)

# 14. Tinto x branco
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
rotulos = clean['type'].map({0: 'tinto', 1: 'branco'})
for ax, col in zip(axes.flat, FEATURES + ['quality']):
    sns.boxplot(x=rotulos, y=clean[col], ax=ax)
    ax.set_title(col)
    ax.set_xlabel('')
for ax in axes.flat[len(FEATURES) + 1:]:
    ax.axis('off')
fig.suptitle('Tinto x branco (dados limpos)')
fig.tight_layout()
fig.savefig('figures/06_tinto_vs_branco.png', dpi=120)
plt.close(fig)

# Resumos numéricos para o relatório
print('Correlação com quality (limpo):')
print(clean[FEATURES].corrwith(clean['quality']).sort_values(key=abs, ascending=False).round(3))
print('\nMédias por tipo (0 = tinto, 1 = branco):')
print(clean.groupby('type')[FEATURES + ['quality']].mean().T.round(3))
print('\nDistribuição de quality (limpo):')
print(clean['quality'].value_counts().sort_index())
