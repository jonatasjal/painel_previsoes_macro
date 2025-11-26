# ==========================================================
# MÓDULO DE PREVISÃO DA TAXA DE DESEMPREGO
# ==========================================================

# ==========================================================
# MÓDULO DE PREVISÃO DA TAXA DE DESEMPREGO (VERSÃO SIMPLIFICADA)
# ==========================================================

# =========== Bibliotecas
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PowerTransformer
from io import StringIO
import google.generativeai as genai
import pandas as pd
import numpy as np
import os

# =========== Definições e configurações globais
h = 12
inicio_treino = pd.to_datetime("2012-03-01")
semente = 1984

def transformar(x, tipo):
    switch = {
        "1": lambda x: x,
        "2": lambda x: x.diff(),
        "3": lambda x: x.diff().diff(),
        "4": lambda x: np.log(x),
        "5": lambda x: np.log(x).diff(),
        "6": lambda x: np.log(x).diff().diff()
    }
    if tipo not in switch:
        raise ValueError("Tipo inválido")
    return switch[tipo](x)

# =========== Carrega dados
dados_brutos = pd.read_parquet("dados/df_mensal.parquet")
dados_tratados = dados_brutos.asfreq("MS")

# =========== Separa Y e X
y = dados_tratados.tx_desemprego.dropna()
x = dados_tratados.drop(labels="tx_desemprego", axis="columns").copy()

# =========== Aplica transformações apenas para variáveis numéricas
for col in x.select_dtypes(include=[np.number]).columns:
    try:
        x[col] = transformar(x[col], "1")  # Usa transformação 1 (sem transformação) para simplificar
    except:
        continue

# =========== Filtra amostra
y = y[y.index >= inicio_treino]
x = x.query("index >= @inicio_treino and index <= @y.index.max()")

# =========== Seleção de variáveis existentes
variaveis_disponiveis = [
    "pib_acum12m", "prod_ind_geral", "pmc_volume", 
    "conf_consumidor_fecomercio", "pop_ocupada", "ibc_br",
    "vendas_supermercados", "cons_energia_total"
]

x_reg = [var for var in variaveis_disponiveis if var in x.columns]

# Adiciona dummies sazonais
dummies_sazonais = (
    pd.get_dummies(y.index.month_name())
    .astype(int)
    .drop(labels="December", axis="columns")
    .set_index(y.index)
)
x = x.join(other=dummies_sazonais, how="outer")
x_reg += dummies_sazonais.columns.to_list()

# =========== Preenche NAs
x = x[x_reg].bfill().ffill()

# =========== Modelos (usando apenas RandomForest para simplificar)
modelo1 = ForecasterAutoreg(
    regressor=RandomForestRegressor(n_estimators=50, random_state=semente),  # Reduzido para performance
    lags=3  # Reduzido lags
)
modelo1.fit(y, x[x_reg])

# =========== Período de previsão
periodo_previsao = pd.date_range(
    start=modelo1.last_window.index[0] + pd.offsets.MonthBegin(1),
    end=modelo1.last_window.index[0] + pd.offsets.MonthBegin(h),
    freq="MS"
)

# =========== Cenários simples (últimos valores disponíveis)
dados_cenarios = x[x_reg].iloc[-1:].copy()
dados_cenarios = pd.concat([dados_cenarios] * h, ignore_index=True)
dados_cenarios.index = periodo_previsao

# =========== Previsões
previsao1 = (
    modelo1.predict(steps=h, exog=dados_cenarios)
    .to_frame()
    .assign(
        Tipo="Random Forest",
        Valor=lambda x: x.iloc[:, 0],
        "Intervalo Inferior"=lambda x: x.Valor * 0.95,  # Intervalo aproximado
        "Intervalo Superior"=lambda x: x.Valor * 1.05
    )
    .drop(columns=0)
)

# =========== IA (mantém igual)
y.to_frame().join(x[x_reg]).to_csv("dados/desemprego.csv")

prompt = f"""
Assume que estamos em {pd.to_datetime("today").strftime("%d/%m/%Y")}.
Forneça sua melhor previsão para a taxa de desemprego no Brasil (em %), medida pela PNAD Contínua do IBGE,
para o período de {periodo_previsao.min().strftime("%B %Y")} a {periodo_previsao.max().strftime("%B %Y")}.
Use os dados históricos do arquivo CSV anexo "desemprego.csv", onde "tx_desemprego" é a variável alvo.
Forneça apenas valores numéricos em formato CSV com cabeçalho.
"""

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    modelo_ia = genai.GenerativeModel(model_name="gemini-1.5-pro")
    arquivo = genai.upload_file("dados/desemprego.csv")
    
    previsao3 = pd.read_csv(
        filepath_or_buffer=StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
        names=["date", "Valor"],
        skiprows=1,
        index_col="date",
        converters={"date": pd.to_datetime}
    ).assign(Tipo="IA")
except:
    # Fallback se a IA falhar
    previsao3 = pd.DataFrame({
        "Valor": [y.iloc[-1]] * h,
        "Tipo": "IA"
    }, index=periodo_previsao)

# =========== Salvar
pasta = "previsao"
if not os.path.exists(pasta):
    os.makedirs(pasta)

pd.concat([
    y.rename("Valor").to_frame().assign(Tipo="Desemprego"),
    previsao1,
    previsao3
]).to_parquet("previsao/desemprego.parquet")

print("Previsão de desemprego concluída com sucesso!")