#########################################################################################
# PROJEÇÃO DO MERCADO DE TELEFONIA MÓVEL NO BRASIL ATÉ 2030 COM ÊNFASE NA TECNOLOGIA 5G
# MBA USP/ESALQ - 2025
# DATA SCIENCE & ANALYTICS
# FRANCISCO ANDERSON MARQUES DA SILVA
##########################################################################################


#%% IMPORTANDO BIBLIOTECAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
from pmdarima import auto_arima
import matplotlib.dates as mdates
from arch import arch_model
from scipy import stats
from scipy.stats import kstest
from scipy.stats import shapiro
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

#%% In[1]: PIVOTANDO BASE

df_anatel = pd.read_csv("BASE_ANATEL.CSV", sep=";")

df_pivot = pd.melt(
    df_anatel,
    id_vars=["IBGE", "MUNICIPIO", "UF", "EMPRESA","TECNOLOGIA"],  # colunas que ficam fixas
    var_name="DATA",                # nome da coluna que vai receber os nomes das colunas originais
    value_name="DEVICES"              # nome da coluna que vai receber os valores
)

# Salvar CSV completo sem limite de linhas
df_pivot.to_csv("BASE_PIVOT.csv", index=False)

print(df_pivot.head())

#%% In[2]: LEITURA DA BASE

# Para abrir o CSV criado da base
df_anatel = pd.read_csv("BASE_PIVOT.CSV", sep=",") 

df_anatel["DATA"] = pd.to_datetime(df_anatel["DATA"], errors='coerce')#convertendo texto para data 
df_anatel["TECNOLOGIA"]= np.where(df_anatel["TECNOLOGIA"] == "5G", "5G", "DEMAIS TEC")
df_anatel = df_anatel[df_anatel['TECNOLOGIA'] == "5G"]

#%%
MAPEAMENTO_REGIOES = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 
    'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'DF': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

#%%
df_anatel['REGIAO'] = df_anatel['UF'].map(MAPEAMENTO_REGIOES)
print(df_anatel.head())

#%% In[3]: ANÁLISE BRASIL
#Base agrupada em nivel Brasil, sem aberturas por UF, Cidade, Empresa

df_brasil = (df_anatel.groupby(["DATA"])["DEVICES"]
            .sum()
            .reset_index()
            .sort_values("DATA", ascending=True))

print(df_brasil.head(10))

# Base regional Norte
df_regiao_NO = ( 
    df_anatel[df_anatel['REGIAO'] == "Norte"]
    .groupby("DATA")["DEVICES"]
    .sum()
    .reset_index()
    .sort_values("DATA", ascending=True)
)

# Nordeste
df_regiao_NE = ( 
    df_anatel[df_anatel['REGIAO'] == "Nordeste"]
    .groupby("DATA")["DEVICES"]
    .sum()
    .reset_index()
    .sort_values("DATA", ascending=True)
)

# Sudeste
df_regiao_SE = ( 
    df_anatel[df_anatel['REGIAO'] == "Sudeste"]
    .groupby("DATA")["DEVICES"]
    .sum()
    .reset_index()
    .sort_values("DATA", ascending=True)
)

# Sul
df_regiao_SU = ( 
    df_anatel[df_anatel['REGIAO'] == "Sul"]
    .groupby("DATA")["DEVICES"]
    .sum()
    .reset_index()
    .sort_values("DATA", ascending=True)
)

# Centro-Oeste
df_regiao_CO = ( 
    df_anatel[df_anatel['REGIAO'] == "Centro-Oeste"]
    .groupby("DATA")["DEVICES"]
    .sum()
    .reset_index()
    .sort_values("DATA", ascending=True)
)


# Criando serie Temporal
serie_brasil=pd.Series(df_brasil["DEVICES"].values, index=df_brasil["DATA"])
# Norte
serie_NO = pd.Series(df_regiao_NO["DEVICES"].values, index=df_regiao_NO["DATA"])

# Nordeste
serie_NE = pd.Series(df_regiao_NE["DEVICES"].values, index=df_regiao_NE["DATA"])

# Sudeste
serie_SE = pd.Series(df_regiao_SE["DEVICES"].values, index=df_regiao_SE["DATA"])

# Sul
serie_SU = pd.Series(df_regiao_SU["DEVICES"].values, index=df_regiao_SU["DATA"])

# Centro-Oeste
serie_CO = pd.Series(df_regiao_CO["DEVICES"].values, index=df_regiao_CO["DATA"])

#%% Gráficos


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Lista com séries, títulos e cores
series = [
    (serie_brasil, "5G Brasil", "black"),
    (serie_NO, "Norte", "green"),
    (serie_NE, "Nordeste", "orange"),
    (serie_SE, "Sudeste", "blue"),
    (serie_SU, "Sul", "red"),
    (serie_CO, "Centro-Oeste", "purple")
]

for i, (serie, titulo, cor) in enumerate(series):
    axes[i].plot(serie.index, serie.values, label=titulo, color=cor, linewidth=2)

    # Título da região
    axes[i].set_title(titulo, fontsize=12, fontweight="bold")

    # Rótulos dos eixos
    axes[i].set_xlabel("Data", fontsize=10)
    axes[i].set_ylabel("Devices 5G", fontsize=10)

    # Formatar datas no eixo X para mm/aa
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    # Remove grid
    axes[i].grid(False)

    # Remove margens extras
    axes[i].margins(x=0, y=0)

    # Deixa só os eixos da esquerda e de baixo visíveis
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    # Legenda discreta
    axes[i].legend(loc="upper left", fontsize=8, frameon=False)

plt.tight_layout()
plt.show()


#%% In[4]: ANÁLISE BRASIL
# Dividindo a base em treino e teste

treino_brasil = serie_brasil[:'2024-06-01']
teste_brasil = serie_brasil ['2024-07-01':]

print(f"Comprimento da série de teste: {len(teste_brasil)}")

# Norte
treino_NO = serie_NO[:'2024-06-01']
teste_NO = serie_NO['2024-07-01':]
print(f"Norte - Comprimento da série de teste: {len(teste_NO)}")

# Nordeste
treino_NE = serie_NE[:'2024-06-01']
teste_NE = serie_NE['2024-07-01':]
print(f"Nordeste - Comprimento da série de teste: {len(teste_NE)}")

# Sudeste
treino_SE = serie_SE[:'2024-06-01']
teste_SE = serie_SE['2024-07-01':]
print(f"Sudeste - Comprimento da série de teste: {len(teste_SE)}")

# Sul
treino_SU = serie_SU[:'2024-06-01']
teste_SU = serie_SU['2024-07-01':]
print(f"Sul - Comprimento da série de teste: {len(teste_SU)}")

# Centro-Oeste
treino_CO = serie_CO[:'2024-06-01']
teste_CO = serie_CO['2024-07-01':]
print(f"Centro-Oeste - Comprimento da série de teste: {len(teste_CO)}")


#%%

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Lista com séries, treino, teste, título e cor da série principal
series_info = [
    (serie_brasil, treino_brasil, teste_brasil, "5G Brasil", "black"),
    (serie_NO, treino_NO, teste_NO, "Norte", "green"),
    (serie_NE, treino_NE, teste_NE, "Nordeste", "orange"),
    (serie_SE, treino_SE, teste_SE, "Sudeste", "blue"),
    (serie_SU, treino_SU, teste_SU, "Sul", "red"),
    (serie_CO, treino_CO, teste_CO, "Centro-Oeste", "purple")
]

for i, (serie, treino, teste, titulo, cor) in enumerate(series_info):
    # Série treino
    axes[i].plot(treino.index, treino.values, label="Treino", color=cor, linewidth=2)

    # Série teste
    axes[i].plot(teste.index, teste.values, label="Teste", color="grey", linewidth=2, linestyle="--")

    # Título e labels
    axes[i].set_title(titulo, fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Data", fontsize=10)
    axes[i].set_ylabel("Devices 5G", fontsize=10)

    # Formatar eixo X para mm/aa
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    # Remove grid
    axes[i].grid(False)

    # Remove margens extras
    axes[i].margins(x=0, y=0)

    # Deixa só os eixos da esquerda e de baixo visíveis
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    # Legenda discreta
    axes[i].legend(loc="upper left", fontsize=8, frameon=False)

plt.tight_layout()
plt.show()


#%% In[5]: ANÁLISE BRASIL
# Testando a autocorrelação dos dados ACF e PACF

fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(treino_brasil, lags=20, ax=axes[0])
plot_pacf(treino_brasil, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Brasil - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Norte
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_NO, lags=20, ax=axes[0])
plot_pacf(treino_NO, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Norte - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Nordeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_NE, lags=20, ax=axes[0])
plot_pacf(treino_NE, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Nordeste - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Sudeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_SE, lags=20, ax=axes[0])
plot_pacf(treino_SE, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Sudeste - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Sul
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_SU, lags=20, ax=axes[0])
plot_pacf(treino_SU, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Sul - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Centro-Oeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_CO, lags=20, ax=axes[0])
plot_pacf(treino_CO, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Centro-Oeste - ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()


#%% In[6]: ANÁLISE BRASIL
#Testando Estacionaridade ADF

result = adfuller(treino_brasil.dropna())
print(f'Resultado do Teste ADF: p-valor = {result[1]}')
if result[1] < 0.05:
    print("A série é estacionária.")
else:
    print("A série não é estacionária.")

# Norte
result = adfuller(treino_NO.dropna())
print("Norte - Teste ADF")
print(f"p-valor = {result[1]}")
if result[1] < 0.05:
    print("A série é estacionária.\n")
else:
    print("A série não é estacionária.\n")

# Nordeste
result = adfuller(treino_NE.dropna())
print("Nordeste - Teste ADF")
print(f"p-valor = {result[1]}")
if result[1] < 0.05:
    print("A série é estacionária.\n")
else:
    print("A série não é estacionária.\n")

# Sudeste
result = adfuller(treino_SE.dropna())
print("Sudeste - Teste ADF")
print(f"p-valor = {result[1]}")
if result[1] < 0.05:
    print("A série é estacionária.\n")
else:
    print("A série não é estacionária.\n")

# Sul
result = adfuller(treino_SU.dropna())
print("Sul - Teste ADF")
print(f"p-valor = {result[1]}")
if result[1] < 0.05:
    print("A série é estacionária.\n")
else:
    print("A série não é estacionária.\n")

# Centro-Oeste
result = adfuller(treino_CO.dropna())
print("Centro-Oeste - Teste ADF")
print(f"p-valor = {result[1]}")
if result[1] < 0.05:
    print("A série é estacionária.\n")
else:
    print("A série não é estacionária.\n")


#%% In[7]: ANÁLISE BRASIL
# Função: Verificando quantos niveis de Diferenciação serão necessários

def verificar_differenciacao(serie, nome):
    # Usar a função ndiffs do pmdarima
    d = pm.arima.ndiffs(serie, test='adf')  # Teste de Dickey-Fuller aumentado
    print(f"A série {nome} precisa de {d} diferenciação(ões) para ser estacionária.")
    return d


#%% In[8]: ANÁLISE BRASIL
# Verificando quantos niveis de Diferenciação serão necessários 

verificar_differenciacao(treino_brasil, "Brasil 5G - Treinamento")

# Norte
verificar_differenciacao(treino_NO, "Norte - Treinamento")

# Nordeste
verificar_differenciacao(treino_NE, "Nordeste - Treinamento")

# Sudeste
verificar_differenciacao(treino_SE, "Sudeste - Treinamento")

# Sul
verificar_differenciacao(treino_SU, "Sul - Treinamento")

# Centro-Oeste
verificar_differenciacao(treino_CO, "Centro-Oeste - Treinamento")

#%% In[9]: ANÁLISE BRASIL
# Primeira Diferenciação

treino_brasil_diff = treino_brasil.diff().dropna()
result_diff = adfuller(treino_brasil_diff)
print("Brasil - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença")
    

# Norte
treino_NO_diff = treino_NO.diff().dropna()
result_diff = adfuller(treino_NO_diff)

print("Norte - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!\n")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença\n")

# Nordeste
treino_NE_diff = treino_NE.diff().dropna()
result_diff = adfuller(treino_NE_diff)
print("Nordeste - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!\n")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença\n")

# Sudeste
treino_SE_diff = treino_SE.diff().dropna()
result_diff = adfuller(treino_SE_diff)
print("Sudeste - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!\n")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença\n")

# Sul
treino_SU_diff = treino_SU.diff().dropna()
result_diff = adfuller(treino_SU_diff)
print("Sul - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!\n")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença\n")

# Centro-Oeste
treino_CO_diff = treino_CO.diff().dropna()
result_diff = adfuller(treino_CO_diff)
print("Centro-Oeste - 1ª diferença")
print(f'ADF após 1ª diferença: p-valor = {result_diff[1]:.6f}')

if result_diff[1] < 0.05:
    print("Série diferenciada é estacionária!\n")
else:
    print("Ainda não estacionária, pode precisar de 2ª diferença\n")

    
#%% In[10]: ANÁLISE BRASIL
# Segunda Diferenciação

treino_brasil_diff2 = treino_brasil.diff().diff().dropna()

result_diff2 = adfuller(treino_brasil_diff2)
print("Brasil - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}')

# Norte
treino_NO_diff2 = treino_NO.diff().diff().dropna()
result_diff2 = adfuller(treino_NO_diff2)
print("Norte - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}\n')

# Nordeste
treino_NE_diff2 = treino_NE.diff().diff().dropna()
result_diff2 = adfuller(treino_NE_diff2)
print("Nordeste - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}\n')

# Sudeste
treino_SE_diff2 = treino_SE.diff().diff().dropna()
result_diff2 = adfuller(treino_SE_diff2)
print("Sudeste - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}\n')

# Sul
treino_SU_diff2 = treino_SU.diff().diff().dropna()
result_diff2 = adfuller(treino_SU_diff2)
print("Sul - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}\n')

# Centro-Oeste
treino_CO_diff2 = treino_CO.diff().diff().dropna()
result_diff2 = adfuller(treino_CO_diff2)
print("Centro-Oeste - 2ª diferença")
print(f'ADF após 2ª diferença: p-valor = {result_diff2[1]:.6f}\n')


#%% In[11]: ANÁLISE BRASIL
# Testando a autocorrelação dos dados ACF e PACF da diferenciação

fig, axes = plt.subplots(1, 2, figsize=(16,4))
plot_acf(treino_brasil_diff2, lags=20, ax=axes[0])
plot_pacf(treino_brasil_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Brasil - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Norte
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_NO_diff2, lags=20, ax=axes[0])
plot_pacf(treino_NO_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Norte - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Nordeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_NE_diff2, lags=20, ax=axes[0])
plot_pacf(treino_NE_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Nordeste - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()
# Sudeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_SE_diff2, lags=20, ax=axes[0])
plot_pacf(treino_SE_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Sudeste - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Sul
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_SU_diff2, lags=20, ax=axes[0])
plot_pacf(treino_SU_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Sul - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()

# Centro-Oeste
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(treino_CO_diff2, lags=20, ax=axes[0])
plot_pacf(treino_CO_diff2, lags=10, ax=axes[1], method='ywm')
plt.suptitle("Centro-Oeste - 2ª diferença ACF e PACF")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.show()


#%% In[14]: ANÁLISE BRASIL
# Função: Modelo Melhor Arima

def melhor_arima(data, max_p=3, max_q=3):
    best_aic = float('inf')
    best_params = None
    best_model = None
    
    # Testar diferentes combinações
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(data, order=(p, 2, q))  # d=2 fixo
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_params = (p, 2, q)
                    best_model = fitted
                    
                print(f'ARIMA{(p,2,q)} - AIC: {fitted.aic:.2f}')
                
            except:
                continue
                
    return best_model, best_params, best_aic
#%% In[15]: ANÁLISE BRASIL
# Modelo Melhor Arima

best_model, best_params, best_aic = melhor_arima(serie_brasil)
print("Brasil")
print(f'\nMelhor modelo: ARIMA{best_params} - AIC: {best_aic:.2f}')

# Norte
best_model_NO, best_params_NO, best_aic_NO = melhor_arima(serie_NO)
print("Norte")
print(f'Melhor modelo: ARIMA{best_params_NO} - AIC: {best_aic_NO:.2f}\n')

# Nordeste
best_model_NE, best_params_NE, best_aic_NE = melhor_arima(serie_NE)
print("Nordeste")
print(f'Melhor modelo: ARIMA{best_params_NE} - AIC: {best_aic_NE:.2f}\n')

# Sudeste
best_model_SE, best_params_SE, best_aic_SE = melhor_arima(serie_SE)
print("Sudeste")
print(f'Melhor modelo: ARIMA{best_params_SE} - AIC: {best_aic_SE:.2f}\n')

# Sul
best_model_SU, best_params_SU, best_aic_SU = melhor_arima(serie_SU)
print("Sul")
print(f'Melhor modelo: ARIMA{best_params_SU} - AIC: {best_aic_SU:.2f}\n')

# Centro-Oeste
best_model_CO, best_params_CO, best_aic_CO = melhor_arima(serie_CO)
print("Centro-Oeste")
print(f'Melhor modelo: ARIMA{best_params_CO} - AIC: {best_aic_CO:.2f}\n')

#%% In[16]: ANÁLISE BRASIL
# Projeção até dezembro 2030 (58 períodos)
forecast_steps = 66  # último dado real JUNHO 2025

#%% In[17]: ANÁLISE BRASIL
# Fazer previsão
forecast = best_model.forecast(steps=forecast_steps)
forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()

# Norte
forecast_NO = best_model_NO.forecast(steps=forecast_steps)
forecast_ci_NO = best_model_NO.get_forecast(steps=forecast_steps).conf_int()

# Nordeste
forecast_NE = best_model_NE.forecast(steps=forecast_steps)
forecast_ci_NE = best_model_NE.get_forecast(steps=forecast_steps).conf_int()

# Sudeste
forecast_SE = best_model_SE.forecast(steps=forecast_steps)
forecast_ci_SE = best_model_SE.get_forecast(steps=forecast_steps).conf_int()

# Sul
forecast_SU = best_model_SU.forecast(steps=forecast_steps)
forecast_ci_SU = best_model_SU.get_forecast(steps=forecast_steps).conf_int()

# Centro-Oeste
forecast_CO = best_model_CO.forecast(steps=forecast_steps)
forecast_ci_CO = best_model_CO.get_forecast(steps=forecast_steps).conf_int()

#%% In[18]: ANÁLISE BRASIL
# Criar índice temporal para as projeções

last_date_br = serie_brasil.index[-1]  # última data dos dados
forecast_dates_br = pd.date_range(start=last_date_br + pd.DateOffset(months=1), 
                              periods=forecast_steps, freq='MS')

# Norte
last_date_NO = serie_NO.index[-1]
forecast_dates_NO = pd.date_range(start=last_date_NO + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='MS')

# Nordeste
last_date_NE = serie_NE.index[-1]
forecast_dates_NE = pd.date_range(start=last_date_NE + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='MS')

# Sudeste
last_date_SE = serie_SE.index[-1]
forecast_dates_SE = pd.date_range(start=last_date_SE + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='MS')

# Sul
last_date_SU = serie_SU.index[-1]
forecast_dates_SU = pd.date_range(start=last_date_SU + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='MS')

# Centro-Oeste
last_date_CO = serie_CO.index[-1]
forecast_dates_CO = pd.date_range(start=last_date_CO + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='MS')

#%% In[19]: ANÁLISE BRASIL
# Organizar resultados
forecast_df = pd.DataFrame({
    'forecast': forecast.values,
    'lower_ci': forecast_ci.iloc[:, 0].values,
    'upper_ci': forecast_ci.iloc[:, 1].values
}, index=forecast_dates_br)

# Norte
forecast_df_NO = pd.DataFrame({
    'forecast': forecast_NO.values,
    'lower_ci': forecast_ci_NO.iloc[:, 0].values,
    'upper_ci': forecast_ci_NO.iloc[:, 1].values
}, index=forecast_dates_NO)

# Nordeste
forecast_df_NE = pd.DataFrame({
    'forecast': forecast_NE.values,
    'lower_ci': forecast_ci_NE.iloc[:, 0].values,
    'upper_ci': forecast_ci_NE.iloc[:, 1].values
}, index=forecast_dates_NE)

# Sudeste
forecast_df_SE = pd.DataFrame({
    'forecast': forecast_SE.values,
    'lower_ci': forecast_ci_SE.iloc[:, 0].values,
    'upper_ci': forecast_ci_SE.iloc[:, 1].values
}, index=forecast_dates_SE)

# Sul
forecast_df_SU = pd.DataFrame({
    'forecast': forecast_SU.values,
    'lower_ci': forecast_ci_SU.iloc[:, 0].values,
    'upper_ci': forecast_ci_SU.iloc[:, 1].values
}, index=forecast_dates_SU)

# Centro-Oeste
forecast_df_CO = pd.DataFrame({
    'forecast': forecast_CO.values,
    'lower_ci': forecast_ci_CO.iloc[:, 0].values,
    'upper_ci': forecast_ci_CO.iloc[:, 1].values
}, index=forecast_dates_CO)

#%%


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Lista com DataFrames de forecast, títulos e cores da série principal
forecast_info = [
    (forecast_df, "Brasil 5G", "black"),
    (forecast_df_NO, "Norte", "green"),
    (forecast_df_NE, "Nordeste", "orange"),
    (forecast_df_SE, "Sudeste", "blue"),
    (forecast_df_SU, "Sul", "red"),
    (forecast_df_CO, "Centro-Oeste", "purple")
]

for i, (df_forecast, titulo, cor) in enumerate(forecast_info):
    # Forecast principal
    axes[i].plot(df_forecast.index, df_forecast['forecast'], label="Forecast", color=cor, linewidth=2)

    # Intervalo de confiança
    axes[i].plot(df_forecast.index, df_forecast['lower_ci'], label="Lower CI", color="grey", linestyle="--")
    axes[i].plot(df_forecast.index, df_forecast['upper_ci'], label="Upper CI", color="grey", linestyle="--")

    # Título e labels
    axes[i].set_title(titulo, fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Data", fontsize=10)
    axes[i].set_ylabel("Devices 5G", fontsize=10)

    # Formatar eixo X para mm/aa
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    # Remover grid e margens
    axes[i].grid(False)
    axes[i].margins(x=0, y=0)

    # Deixar só os eixos de baixo e da esquerda
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    # Legenda discreta
    axes[i].legend(loc="upper left", fontsize=8, frameon=False)

plt.tight_layout()
plt.show()



#%% In[20]: ANÁLISE BRASIL
# Validação dos residuos

residuos = serie_brasil - best_model.fittedvalues
print(residuos.head())

# Norte
residuos_NO = serie_NO - best_model_NO.fittedvalues
print("Norte - Resíduos")
print(residuos_NO.head(), "\n")

# Nordeste
residuos_NE = serie_NE - best_model_NE.fittedvalues
print("Nordeste - Resíduos")
print(residuos_NE.head(), "\n")

# Sudeste
residuos_SE = serie_SE - best_model_SE.fittedvalues
print("Sudeste - Resíduos")
print(residuos_SE.head(), "\n")

# Sul
residuos_SU = serie_SU - best_model_SU.fittedvalues
print("Sul - Resíduos")
print(residuos_SU.head(), "\n")

# Centro-Oeste
residuos_CO = serie_CO - best_model_CO.fittedvalues
print("Centro-Oeste - Resíduos")
print(residuos_CO.head(), "\n")

#%% In[21]: ANÁLISE BRASIL
# Ljung, Autocorrelação

ljung_box = acorr_ljungbox(residuos, lags=10)
print(f"P-valor Ljung-Box: {ljung_box['lb_pvalue'].iloc[-1]:.4f}")

if ljung_box['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)")

# Norte
ljung_box_NO = acorr_ljungbox(residuos_NO, lags=10)
print("Norte - Ljung-Box")
print(f"P-valor Ljung-Box: {ljung_box_NO['lb_pvalue'].iloc[-1]:.4f}")
if ljung_box_NO['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)\n")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)\n")

# Nordeste
ljung_box_NE = acorr_ljungbox(residuos_NE, lags=10)
print("Nordeste - Ljung-Box")
print(f"P-valor Ljung-Box: {ljung_box_NE['lb_pvalue'].iloc[-1]:.4f}")
if ljung_box_NE['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)\n")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)\n")

# Sudeste
ljung_box_SE = acorr_ljungbox(residuos_SE, lags=10)
print("Sudeste - Ljung-Box")
print(f"P-valor Ljung-Box: {ljung_box_SE['lb_pvalue'].iloc[-1]:.4f}")
if ljung_box_SE['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)\n")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)\n")

# Sul
ljung_box_SU = acorr_ljungbox(residuos_SU, lags=10)
print("Sul - Ljung-Box")
print(f"P-valor Ljung-Box: {ljung_box_SU['lb_pvalue'].iloc[-1]:.4f}")
if ljung_box_SU['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)\n")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)\n")

# Centro-Oeste
ljung_box_CO = acorr_ljungbox(residuos_CO, lags=10)
print("Centro-Oeste - Ljung-Box")
print(f"P-valor Ljung-Box: {ljung_box_CO['lb_pvalue'].iloc[-1]:.4f}")
if ljung_box_CO['lb_pvalue'].iloc[-1] > 0.05:
    print("Resíduos são ruído branco (modelo OK)\n")
else:
    print("Resíduos têm autocorrelação (modelo inadequado)\n")

#%% In[22]: ANÁLISE BRASIL
# Shapiro, Normalidade (distribuição normal) rigoroso

# H0: resíduos são normais
shapiro_stat, shapiro_p = shapiro(residuos.dropna())
print(f"P-valor normalidade: {shapiro_p:.6f}")

if shapiro_p > 0.05:
    print("Resíduos são normais")
else:
    print("Resíduos não são normais (não impede previsão)")

# Norte
shapiro_stat_NO, shapiro_p_NO = shapiro(residuos_NO.dropna())
print("Norte - Shapiro-Wilk")
print(f"P-valor normalidade: {shapiro_p_NO:.6f}")
if shapiro_p_NO > 0.05:
    print("Resíduos são normais\n")
else:
    print("Resíduos não são normais (não impede previsão)\n")

# Nordeste
shapiro_stat_NE, shapiro_p_NE = shapiro(residuos_NE.dropna())
print("Nordeste - Shapiro-Wilk")
print(f"P-valor normalidade: {shapiro_p_NE:.6f}")
if shapiro_p_NE > 0.05:
    print("Resíduos são normais\n")
else:
    print("Resíduos não são normais (não impede previsão)\n")

# Sudeste
shapiro_stat_SE, shapiro_p_SE = shapiro(residuos_SE.dropna())
print("Sudeste - Shapiro-Wilk")
print(f"P-valor normalidade: {shapiro_p_SE:.6f}")
if shapiro_p_SE > 0.05:
    print("Resíduos são normais\n")
else:
    print("Resíduos não são normais (não impede previsão)\n")

# Sul
shapiro_stat_SU, shapiro_p_SU = shapiro(residuos_SU.dropna())
print("Sul - Shapiro-Wilk")
print(f"P-valor normalidade: {shapiro_p_SU:.6f}")
if shapiro_p_SU > 0.05:
    print("Resíduos são normais\n")
else:
    print("Resíduos não são normais (não impede previsão)\n")

# Centro-Oeste
shapiro_stat_CO, shapiro_p_CO = shapiro(residuos_CO.dropna())
print("Centro-Oeste - Shapiro-Wilk")
print(f"P-valor normalidade: {shapiro_p_CO:.6f}")
if shapiro_p_CO > 0.05:
    print("Resíduos são normais\n")
else:
    print("Resíduos não são normais (não impede previsão)\n")

#%% In[23]: ANÁLISE BRASIL
# Kolmogorov-Smirnov, Normalidade (amostras), menos rigoroso

ks_stat, p_value = kstest(residuos, 'norm', 
                          args=(np.mean(residuos), 
                                np.std(residuos)))
print(f'Teste de Kolmogorov-Smirnov para normalidade: p-valor = {p_value}')
if p_value > 0.01:
    print("Os resíduos seguem uma distribuição normal.")
else:
    print("Os resíduos não seguem uma distribuição normal.")

# Norte
ks_stat_NO, p_value_NO = kstest(residuos_NO.dropna(), 'norm',
                                args=(np.mean(residuos_NO.dropna()), np.std(residuos_NO.dropna())))
print("Norte - Kolmogorov-Smirnov")
print(f'p-valor = {p_value_NO:.6f}')
if p_value_NO > 0.01:
    print("Os resíduos seguem uma distribuição normal.\n")
else:
    print("Os resíduos não seguem uma distribuição normal.\n")

# Nordeste
ks_stat_NE, p_value_NE = kstest(residuos_NE.dropna(), 'norm',
                                args=(np.mean(residuos_NE.dropna()), np.std(residuos_NE.dropna())))
print("Nordeste - Kolmogorov-Smirnov")
print(f'p-valor = {p_value_NE:.6f}')
if p_value_NE > 0.01:
    print("Os resíduos seguem uma distribuição normal.\n")
else:
    print("Os resíduos não seguem uma distribuição normal.\n")

# Sudeste
ks_stat_SE, p_value_SE = kstest(residuos_SE.dropna(), 'norm',
                                args=(np.mean(residuos_SE.dropna()), np.std(residuos_SE.dropna())))
print("Sudeste - Kolmogorov-Smirnov")
print(f'p-valor = {p_value_SE:.6f}')
if p_value_SE > 0.01:
    print("Os resíduos seguem uma distribuição normal.\n")
else:
    print("Os resíduos não seguem uma distribuição normal.\n")

# Sul
ks_stat_SU, p_value_SU = kstest(residuos_SU.dropna(), 'norm',
                                args=(np.mean(residuos_SU.dropna()), np.std(residuos_SU.dropna())))
print("Sul - Kolmogorov-Smirnov")
print(f'p-valor = {p_value_SU:.6f}')
if p_value_SU > 0.01:
    print("Os resíduos seguem uma distribuição normal.\n")
else:
    print("Os resíduos não seguem uma distribuição normal.\n")

# Centro-Oeste
ks_stat_CO, p_value_CO = kstest(residuos_CO.dropna(), 'norm',
                                args=(np.mean(residuos_CO.dropna()), np.std(residuos_CO.dropna())))
print("Centro-Oeste - Kolmogorov-Smirnov")
print(f'p-valor = {p_value_CO:.6f}')
if p_value_CO > 0.01:
    print("Os resíduos seguem uma distribuição normal.\n")
else:
    print("Os resíduos não seguem uma distribuição normal.\n")


#%% In[24]: ANÁLISE BRASIL
# Previsão no período de teste
forecast_teste_steps = len(teste_brasil)  # número de períodos do teste

# Refazer o modelo apenas com dados de treino
modelo_treino = ARIMA(treino_brasil, order=best_params).fit()
forecast_teste = modelo_treino.forecast(steps=forecast_teste_steps)

# Criar DataFrame com as datas do teste
forecast_teste_df = pd.DataFrame({
    'real': teste_brasil.values,
    'forecast': forecast_teste.values
}, index=teste_brasil.index)

print("Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df)

# Norte
forecast_teste_steps_NO = len(teste_NO)
modelo_treino_NO = ARIMA(treino_NO, order=best_params_NO).fit()
forecast_teste_NO = modelo_treino_NO.forecast(steps=forecast_teste_steps_NO)
forecast_teste_df_NO = pd.DataFrame({
    'real': teste_NO.values,
    'forecast': forecast_teste_NO.values
}, index=teste_NO.index)
print("Norte - Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df_NO, "\n")

# Nordeste
forecast_teste_steps_NE = len(teste_NE)
modelo_treino_NE = ARIMA(treino_NE, order=best_params_NE).fit()
forecast_teste_NE = modelo_treino_NE.forecast(steps=forecast_teste_steps_NE)
forecast_teste_df_NE = pd.DataFrame({
    'real': teste_NE.values,
    'forecast': forecast_teste_NE.values
}, index=teste_NE.index)
print("Nordeste - Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df_NE, "\n")

# Sudeste
forecast_teste_steps_SE = len(teste_SE)
modelo_treino_SE = ARIMA(treino_SE, order=best_params_SE).fit()
forecast_teste_SE = modelo_treino_SE.forecast(steps=forecast_teste_steps_SE)
forecast_teste_df_SE = pd.DataFrame({
    'real': teste_SE.values,
    'forecast': forecast_teste_SE.values
}, index=teste_SE.index)
print("Sudeste - Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df_SE, "\n")

# Sul
forecast_teste_steps_SU = len(teste_SU)
modelo_treino_SU = ARIMA(treino_SU, order=best_params_SU).fit()
forecast_teste_SU = modelo_treino_SU.forecast(steps=forecast_teste_steps_SU)
forecast_teste_df_SU = pd.DataFrame({
    'real': teste_SU.values,
    'forecast': forecast_teste_SU.values
}, index=teste_SU.index)
print("Sul - Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df_SU, "\n")

# Centro-Oeste
forecast_teste_steps_CO = len(teste_CO)
modelo_treino_CO = ARIMA(treino_CO, order=best_params_CO).fit()
forecast_teste_CO = modelo_treino_CO.forecast(steps=forecast_teste_steps_CO)
forecast_teste_df_CO = pd.DataFrame({
    'real': teste_CO.values,
    'forecast': forecast_teste_CO.values
}, index=teste_CO.index)
print("Centro-Oeste - Comparação Real vs Forecast (período de teste):")
print(forecast_teste_df_CO, "\n")

#%% In[25]: ANÁLISE BRASIL
# Calcular MAPE para o período de teste
mape_teste = mean_absolute_percentage_error(forecast_teste_df['real'], 
                                          forecast_teste_df['forecast'])
print(f"Brasil - MAPE do modelo no período de teste: {mape_teste:.2%}")

# Norte
mape_teste_NO = mean_absolute_percentage_error(forecast_teste_df_NO['real'], forecast_teste_df_NO['forecast'])
print(f"Norte - MAPE do modelo no período de teste: {mape_teste_NO:.2%}")

# Nordeste
mape_teste_NE = mean_absolute_percentage_error(forecast_teste_df_NE['real'], forecast_teste_df_NE['forecast'])
print(f"Nordeste - MAPE do modelo no período de teste: {mape_teste_NE:.2%}")

# Sudeste
mape_teste_SE = mean_absolute_percentage_error(forecast_teste_df_SE['real'], forecast_teste_df_SE['forecast'])
print(f"Sudeste - MAPE do modelo no período de teste: {mape_teste_SE:.2%}")

# Sul
mape_teste_SU = mean_absolute_percentage_error(forecast_teste_df_SU['real'], forecast_teste_df_SU['forecast'])
print(f"Sul - MAPE do modelo no período de teste: {mape_teste_SU:.2%}")

# Centro-Oeste
mape_teste_CO = mean_absolute_percentage_error(forecast_teste_df_CO['real'], forecast_teste_df_CO['forecast'])
print(f"Centro-Oeste - MAPE do modelo no período de teste: {mape_teste_CO:.2%}")


#%% In[26]: ANÁLISE BRASIL
# Calcular outras métricas também
mae_teste = mean_absolute_error(forecast_teste_df['real'], 
                               forecast_teste_df['forecast'])
rmse_teste = np.sqrt(mean_squared_error(forecast_teste_df['real'], 
                                       forecast_teste_df['forecast']))

print(f"MAE (Mean Absolute Error): {mae_teste:,.0f}")
print(f"RMSE (Root Mean Square Error): {rmse_teste:,.0f}")

# Norte
mae_teste_NO = mean_absolute_error(forecast_teste_df_NO['real'], 
                                   forecast_teste_df_NO['forecast'])
rmse_teste_NO = np.sqrt(mean_squared_error(forecast_teste_df_NO['real'], 
                                           forecast_teste_df_NO['forecast']))
print("Norte - Métricas de Forecast")
print(f"MAE: {mae_teste_NO:,.0f}")
print(f"RMSE: {rmse_teste_NO:,.0f}\n")

# Nordeste
mae_teste_NE = mean_absolute_error(forecast_teste_df_NE['real'], 
                                   forecast_teste_df_NE['forecast'])
rmse_teste_NE = np.sqrt(mean_squared_error(forecast_teste_df_NE['real'], 
                                           forecast_teste_df_NE['forecast']))
print("Nordeste - Métricas de Forecast")
print(f"MAE: {mae_teste_NE:,.0f}")
print(f"RMSE: {rmse_teste_NE:,.0f}\n")

# Sudeste
mae_teste_SE = mean_absolute_error(forecast_teste_df_SE['real'], 
                                   forecast_teste_df_SE['forecast'])
rmse_teste_SE = np.sqrt(mean_squared_error(forecast_teste_df_SE['real'], 
                                           forecast_teste_df_SE['forecast']))
print("Sudeste - Métricas de Forecast")
print(f"MAE: {mae_teste_SE:,.0f}")
print(f"RMSE: {rmse_teste_SE:,.0f}\n")

# Sul
mae_teste_SU = mean_absolute_error(forecast_teste_df_SU['real'], 
                                   forecast_teste_df_SU['forecast'])
rmse_teste_SU = np.sqrt(mean_squared_error(forecast_teste_df_SU['real'], 
                                           forecast_teste_df_SU['forecast']))
print("Sul - Métricas de Forecast")
print(f"MAE: {mae_teste_SU:,.0f}")
print(f"RMSE: {rmse_teste_SU:,.0f}\n")

# Centro-Oeste
mae_teste_CO = mean_absolute_error(forecast_teste_df_CO['real'], 
                                   forecast_teste_df_CO['forecast'])
rmse_teste_CO = np.sqrt(mean_squared_error(forecast_teste_df_CO['real'], 
                                           forecast_teste_df_CO['forecast']))
print("Centro-Oeste - Métricas de Forecast")
print(f"MAE: {mae_teste_CO:,.0f}")
print(f"RMSE: {rmse_teste_CO:,.0f}\n")


#%% In[27]: ANÁLISE BRASIL
# Visualizar a comparação

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Lista com séries e DataFrames de forecast de teste, títulos
series_info = [
    (serie_brasil, treino_brasil, teste_brasil, forecast_teste_df, "Brasil 5G"),
    (serie_NO, treino_NO, teste_NO, forecast_teste_df_NO, "Norte"),
    (serie_NE, treino_NE, teste_NE, forecast_teste_df_NE, "Nordeste"),
    (serie_SE, treino_SE, teste_SE, forecast_teste_df_SE, "Sudeste"),
    (serie_SU, treino_SU, teste_SU, forecast_teste_df_SU, "Sul"),
    (serie_CO, treino_CO, teste_CO, forecast_teste_df_CO, "Centro-Oeste")
]

for i, (serie, treino, teste, forecast_df_teste, titulo) in enumerate(series_info):
    # Série completa
    axes[i].plot(serie.index, serie.values, label="Série Completa", color="black", alpha=0.7)

    # Treino
    axes[i].plot(treino.index, treino.values, label="Treino", color="blue", linewidth=1.5)

    # Teste real
    axes[i].plot(teste.index, teste.values, label="Teste Real", color="green", linewidth=2)

    # Forecast do teste
    axes[i].plot(
        forecast_df_teste.index, 
        forecast_df_teste["forecast"], 
        label="Forecast Teste", 
        color="red", 
        linewidth=2, 
        linestyle="--"
    )

    # Título e labels
    axes[i].set_title(titulo, fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Data", fontsize=10)
    axes[i].set_ylabel("Devices 5G", fontsize=10)

    # Formatar eixo X para mm/aa
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))

    # Remover grid e margens
    axes[i].grid(False)
    axes[i].margins(x=0, y=0)

    # Deixar só os eixos de baixo e da esquerda
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    # Legenda discreta
    axes[i].legend(loc="upper left", fontsize=8, frameon=False)

plt.tight_layout()
plt.show()

#%% In[28]: ANÁLISE BRASIL
# Se quiser unir TODA a série (treino + teste) com forecast completo para visualização
# Verificar se há sobreposição de índices

print("Última data do fittedvalues:", best_model.fittedvalues.index[-1])
print("Primeira data do teste:", teste_brasil.index[0])

# Criar série completa de forecast (in-sample + out-of-sample)
# Garantir que não há duplicação de índices
forecast_completo = pd.concat([
    best_model.fittedvalues,
    pd.Series(forecast_teste.values, index=teste_brasil.index)
])

# Verificar se há índices duplicados
if forecast_completo.index.duplicated().any():
   forecast_completo = forecast_completo[~forecast_completo.index.duplicated(keep='last')]

# Alinhar os índices - usar apenas as datas que existem em ambas as séries
indices_comuns = serie_brasil.index.intersection(forecast_completo.index)

# DataFrame com série real completa vs forecast completo
serie_completa_df = pd.DataFrame({
    'real': serie_brasil.loc[indices_comuns],
    'forecast': forecast_completo.loc[indices_comuns]
})

print("\nSérie completa - Real vs Forecast:")
print(serie_completa_df.tail(10))
print(f"\nTamanho da série real: {len(serie_brasil)}")
print(f"Tamanho da série forecast: {len(forecast_completo)}")
print(f"Tamanho da série combinada: {len(serie_completa_df)}")

# ----------------- Norte -----------------
print("Norte")
print("Última data do fittedvalues:", best_model_NO.fittedvalues.index[-1])
print("Primeira data do teste:", teste_NO.index[0])

forecast_completo_NO = pd.concat([
    best_model_NO.fittedvalues,
    pd.Series(best_model_NO.forecast(steps=len(teste_NO)), index=teste_NO.index)
])

if forecast_completo_NO.index.duplicated().any():
    forecast_completo_NO = forecast_completo_NO[~forecast_completo_NO.index.duplicated(keep='last')]

indices_comuns_NO = serie_NO.index.intersection(forecast_completo_NO.index)

serie_completa_df_NO = pd.DataFrame({
    'real': serie_NO.loc[indices_comuns_NO],
    'forecast': forecast_completo_NO.loc[indices_comuns_NO]
})

print("\nSérie completa - Real vs Forecast (últimos 10):")
print(serie_completa_df_NO.tail(10))
print(f"Tamanho da série real: {len(serie_NO)}")
print(f"Tamanho da série forecast: {len(forecast_completo_NO)}")
print(f"Tamanho da série combinada: {len(serie_completa_df_NO)}\n")

# ----------------- Nordeste -----------------
print("Nordeste")
print("Última data do fittedvalues:", best_model_NE.fittedvalues.index[-1])
print("Primeira data do teste:", teste_NE.index[0])

forecast_completo_NE = pd.concat([
    best_model_NE.fittedvalues,
    pd.Series(best_model_NE.forecast(steps=len(teste_NE)), index=teste_NE.index)
])

if forecast_completo_NE.index.duplicated().any():
    forecast_completo_NE = forecast_completo_NE[~forecast_completo_NE.index.duplicated(keep='last')]

indices_comuns_NE = serie_NE.index.intersection(forecast_completo_NE.index)

serie_completa_df_NE = pd.DataFrame({
    'real': serie_NE.loc[indices_comuns_NE],
    'forecast': forecast_completo_NE.loc[indices_comuns_NE]
})

print("\nSérie completa - Real vs Forecast (últimos 10):")
print(serie_completa_df_NE.tail(10))
print(f"Tamanho da série real: {len(serie_NE)}")
print(f"Tamanho da série forecast: {len(forecast_completo_NE)}")
print(f"Tamanho da série combinada: {len(serie_completa_df_NE)}\n")

# ----------------- Sudeste -----------------
print("Sudeste")
print("Última data do fittedvalues:", best_model_SE.fittedvalues.index[-1])
print("Primeira data do teste:", teste_SE.index[0])

forecast_completo_SE = pd.concat([
    best_model_SE.fittedvalues,
    pd.Series(best_model_SE.forecast(steps=len(teste_SE)), index=teste_SE.index)
])

if forecast_completo_SE.index.duplicated().any():
    forecast_completo_SE = forecast_completo_SE[~forecast_completo_SE.index.duplicated(keep='last')]

indices_comuns_SE = serie_SE.index.intersection(forecast_completo_SE.index)

serie_completa_df_SE = pd.DataFrame({
    'real': serie_SE.loc[indices_comuns_SE],
    'forecast': forecast_completo_SE.loc[indices_comuns_SE]
})

print("\nSérie completa - Real vs Forecast (últimos 10):")
print(serie_completa_df_SE.tail(10))
print(f"Tamanho da série real: {len(serie_SE)}")
print(f"Tamanho da série forecast: {len(forecast_completo_SE)}")
print(f"Tamanho da série combinada: {len(serie_completa_df_SE)}\n")

# ----------------- Sul -----------------
print("Sul")
print("Última data do fittedvalues:", best_model_SU.fittedvalues.index[-1])
print("Primeira data do teste:", teste_SU.index[0])

forecast_completo_SU = pd.concat([
    best_model_SU.fittedvalues,
    pd.Series(best_model_SU.forecast(steps=len(teste_SU)), index=teste_SU.index)
])

if forecast_completo_SU.index.duplicated().any():
    forecast_completo_SU = forecast_completo_SU[~forecast_completo_SU.index.duplicated(keep='last')]

indices_comuns_SU = serie_SU.index.intersection(forecast_completo_SU.index)

serie_completa_df_SU = pd.DataFrame({
    'real': serie_SU.loc[indices_comuns_SU],
    'forecast': forecast_completo_SU.loc[indices_comuns_SU]
})

print("\nSérie completa - Real vs Forecast (últimos 10):")
print(serie_completa_df_SU.tail(10))
print(f"Tamanho da série real: {len(serie_SU)}")
print(f"Tamanho da série forecast: {len(forecast_completo_SU)}")
print(f"Tamanho da série combinada: {len(serie_completa_df_SU)}\n")

# ----------------- Centro-Oeste -----------------
print("Centro-Oeste")
print("Última data do fittedvalues:", best_model_CO.fittedvalues.index[-1])
print("Primeira data do teste:", teste_CO.index[0])

forecast_completo_CO = pd.concat([
    best_model_CO.fittedvalues,
    pd.Series(best_model_CO.forecast(steps=len(teste_CO)), index=teste_CO.index)
])

if forecast_completo_CO.index.duplicated().any():
    forecast_completo_CO = forecast_completo_CO[~forecast_completo_CO.index.duplicated(keep='last')]

indices_comuns_CO = serie_CO.index.intersection(forecast_completo_CO.index)

serie_completa_df_CO = pd.DataFrame({
    'real': serie_CO.loc[indices_comuns_CO],
    'forecast': forecast_completo_CO.loc[indices_comuns_CO]
})

print("\nSérie completa - Real vs Forecast (últimos 10):")
print(serie_completa_df_CO.tail(10))
print(f"Tamanho da série real: {len(serie_CO)}")
print(f"Tamanho da série forecast: {len(forecast_completo_CO)}")
print(f"Tamanho da série combinada: {len(serie_completa_df_CO)}\n")


#%% In[29]: ANÁLISE BRASIL
# MAPE apenas para dados out-of-sample (mais importante)
print(f"\n{'='*50}")
print(f"MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste:.2%}")
print(f"MAE: {mae_teste:,.0f} devices")
print(f"RMSE: {rmse_teste:,.0f} devices")
print(f"Período de teste: {teste_brasil.index[0].strftime('%Y-%m')} a {teste_brasil.index[-1].strftime('%Y-%m')}")

# ----------------- Norte -----------------
print(f"\n{'='*50}")
print("Norte - MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste_NO:.2%}")
print(f"MAE: {mae_teste_NO:,.0f} devices")
print(f"RMSE: {rmse_teste_NO:,.0f} devices")
print(f"Período de teste: {teste_NO.index[0].strftime('%Y-%m')} a {teste_NO.index[-1].strftime('%Y-%m')}")

# ----------------- Nordeste -----------------
print(f"\n{'='*50}")
print("Nordeste - MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste_NE:.2%}")
print(f"MAE: {mae_teste_NE:,.0f} devices")
print(f"RMSE: {rmse_teste_NE:,.0f} devices")
print(f"Período de teste: {teste_NE.index[0].strftime('%Y-%m')} a {teste_NE.index[-1].strftime('%Y-%m')}")

# ----------------- Sudeste -----------------
print(f"\n{'='*50}")
print("Sudeste - MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste_SE:.2%}")
print(f"MAE: {mae_teste_SE:,.0f} devices")
print(f"RMSE: {rmse_teste_SE:,.0f} devices")
print(f"Período de teste: {teste_SE.index[0].strftime('%Y-%m')} a {teste_SE.index[-1].strftime('%Y-%m')}")

# ----------------- Sul -----------------
print(f"\n{'='*50}")
print("Sul - MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste_SU:.2%}")
print(f"MAE: {mae_teste_SU:,.0f} devices")
print(f"RMSE: {rmse_teste_SU:,.0f} devices")
print(f"Período de teste: {teste_SU.index[0].strftime('%Y-%m')} a {teste_SU.index[-1].strftime('%Y-%m')}")

# ----------------- Centro-Oeste -----------------
print(f"\n{'='*50}")
print("Centro-Oeste - MÉTRICAS DE VALIDAÇÃO (PERÍODO DE TESTE)")
print(f"{'='*50}")
print(f"MAPE: {mape_teste_CO:.2%}")
print(f"MAE: {mae_teste_CO:,.0f} devices")
print(f"RMSE: {rmse_teste_CO:,.0f} devices")
print(f"Período de teste: {teste_CO.index[0].strftime('%Y-%m')} a {teste_CO.index[-1].strftime('%Y-%m')}")

#%% In[30]: ANÁLISE BRASIL

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# Lista de séries, forecasts de teste, projeções e títulos
series_info = [
    (serie_brasil, forecast_teste_df, forecast_df, mape_teste, "Brasil", 'red', 'orange'),
    (serie_NO, forecast_teste_df_NO, forecast_df_NO, mape_teste_NO, "Norte", 'blue', 'cyan'),
    (serie_NE, forecast_teste_df_NE, forecast_df_NE, mape_teste_NE, "Nordeste", 'green', 'lime'),
    (serie_SE, forecast_teste_df_SE, forecast_df_SE, mape_teste_SE, "Sudeste", 'purple', 'violet'),
    (serie_SU, forecast_teste_df_SU, forecast_df_SU, mape_teste_SU, "Sul", 'brown', 'gold'),
    (serie_CO, forecast_teste_df_CO, forecast_df_CO, mape_teste_CO, "Centro-Oeste", 'pink', 'gold')
]

for i, (serie, forecast_teste, forecast_futuro, mape, titulo, cor_forecast, cor_proj) in enumerate(series_info):
    # Série histórica
    axes[i].plot(serie, label='Dados Reais', color='black', linewidth=2)
    
    # Forecast no período de teste
    axes[i].plot(forecast_teste.index, forecast_teste['forecast'], 
                 label=f'Forecast Teste (MAPE: {mape:.1%})',
                 color=cor_forecast, linewidth=2, linestyle='--')
    
    # Projeção futura
    axes[i].plot(forecast_futuro.index, forecast_futuro['forecast'], 
                 label='Projeção até 2030', color=cor_proj, linewidth=2)
    
    # Intervalo de confiança
    axes[i].fill_between(forecast_futuro.index,
                         forecast_futuro['lower_ci'],
                         forecast_futuro['upper_ci'],
                         color='lightgray', alpha=0.3, label='IC 95%')
    
    axes[i].set_title(f"{titulo}", fontsize=12)
    axes[i].set_xlabel('Data')
    axes[i].set_ylabel('Devices 5G')
    axes[i].legend(fontsize=8)
    
     
    # Formatar eixo x como mm/yy
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    fig.autofmt_xdate(rotation=45)  # Rotaciona para melhor visualização

# Remover grid e margens
    axes[i].grid(False)
    axes[i].margins(x=0, y=0)

    # Deixar só os eixos de baixo e da esquerda
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

    # Legenda discreta
    axes[i].legend(loc="upper left", fontsize=8, frameon=False)

plt.tight_layout()
plt.show()

#%% In[31]: ANÁLISE BRASIL
# Lista de regiões e seus DataFrames
regioes = [
    ("Brasil", df_brasil, forecast_df),
    ("Norte", df_regiao_NO, forecast_df_NO),
    ("Nordeste", df_regiao_NE, forecast_df_NE),
    ("Sudeste", df_regiao_SE, forecast_df_SE),
    ("Sul", df_regiao_SU, forecast_df_SU),
    ("Centro-Oeste", df_regiao_CO, forecast_df_CO)
]

# Lista para armazenar DataFrames long
dfs_long = []

for nome, df_real, df_forecast in regioes:
    # Dados reais
    temp_real = df_real[['DATA','DEVICES']].copy()
    temp_real = temp_real.rename(columns={'DEVICES':'valores'})
    temp_real['REGIAO'] = nome
    temp_real['tipo'] = 'real'
    
    # Dados forecast
    temp_forecast = df_forecast[['forecast']].copy()
    temp_forecast = temp_forecast.reset_index().rename(columns={'index':'DATA','forecast':'valores'})
    temp_forecast['REGIAO'] = nome
    temp_forecast['tipo'] = 'forecast'
    
    # Concatenar real + forecast
    df_regiao_long = pd.concat([temp_real, temp_forecast], ignore_index=True)
    dfs_long.append(df_regiao_long)

# Concatenar todas as regiões
df_long = pd.concat(dfs_long, ignore_index=True)
df_long = df_long.sort_values(['REGIAO','DATA']).reset_index(drop=True)

print("DataFrame em formato long (cada linha = data + região + valor + tipo):")
print(df_long.head(10))
print(df_long.tail(10))
print(f"\nTotal de registros: {len(df_long)}")
print(f"Registros por tipo:")
print(df_long['tipo'].value_counts())
print(f"Registros por região:")
print(df_long['REGIAO'].value_counts())

#%% In[31]: ANÁLISE BRASIL
# Valores finais de cada ano

df_brasil_anual = df_long[df_long['DATA'].dt.month == 12]

print("="*60)
print("DADOS DE DEZEMBRO - REAL E FORECAST ATÉ 2030")
print("="*60)
print(df_brasil_anual)
df_brasil_anual.to_csv("BASE_DEZ2030.csv", 
                       index=False,
                       sep=';',          # Ponto-e-vírgula separa colunas
                       decimal=',')      # Vírgula como separador decimal

print("CÓDIGO ENCERRADO")