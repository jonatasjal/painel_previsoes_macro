# ==========================================================
# MÓDULO DE PREVISÃO DA PRODUÇÃO INDUSTRIAL
# ==========================================================

# =========== Bibliotecas
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.preprocessing import PowerTransformer
from io import StringIO
import google.generativeai as genai
import pandas as pd
import numpy as np
import os

# =========== Definições e configurações globais
h = 12
inicio_treino = pd.to_datetime("2004-01-01")
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

# =========== Metadados
metadados = (
    pd.read_excel(
        io="https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name="Metadados",
        dtype=str,
        index_col="Identificador"
    )
    .filter(["Transformação"])
)

# =========== Carrega dados
dados_brutos = pd.read_parquet("dados/df_mensal.parquet")
dados_tratados = dados_brutos.asfreq("MS")

# =========== Separa Y e X
y = dados_tratados.prod_ind_geral.dropna()
x = dados_tratados.drop(labels="prod_ind_geral", axis="columns").copy()

# =========== Aplica transformações
for col in x.columns:
    x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

# =========== Filtra amostra
y = y[y.index >= inicio_treino]
x = x.query("index >= @inicio_treino and index <= @y.index.max()")

# =========== Trata NAs
prop_na = x.isnull().sum() / y.shape[0]
x = x.drop(labels=prop_na[prop_na >= 0.2].index.to_list(), axis="columns")
x = x.bfill().ffill()

# =========== Dummies sazonais
dummies_sazonais = (
    pd.get_dummies(y.index.month_name())
    .astype(int)
    .drop(labels="December", axis="columns")
    .set_index(y.index)
)
x = x.join(other=dummies_sazonais, how="outer")

# =========== Seleção de variáveis
x_reg = [
    "uci_ind_fgv",
    "expec_ipca_top5_curto_prazo",
    "cambio",
    "ibc_br",
    "conf_ind_cni"
] + dummies_sazonais.columns.to_list()

# =========== Modelos
modelo1 = ForecasterAutoreg(
    regressor=Lasso(random_state=semente),
    lags=3,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo1.fit(y, x[x_reg])

modelo2 = ForecasterAutoreg(
    regressor=BayesianRidge(),
    lags=3,
    transformer_y=PowerTransformer(),
    transformer_exog=PowerTransformer()
)
modelo2.fit(y, x[x_reg])

# =========== Período de previsão
periodo_previsao = pd.date_range(
    start=modelo1.last_window.index[0] + pd.offsets.MonthBegin(1),
    end=modelo1.last_window.index[0] + pd.offsets.MonthBegin(h),
    freq="MS"
)

# =========== Cenários
dados_cenarios = {}
for var in x_reg:
    if var in x.columns:
        if var in dummies_sazonais.columns:
            # Para dummies sazonais, usar padrão mensal
            dados_cenario = pd.get_dummies(periodo_previsao.month_name())[var].values
        else:
            # Para outras variáveis, usar mediana histórica por mês
            dados_cenario = (
                x[[var]]
                .dropna()
                .query("index >= @inicio_treino")
                .assign(mes=lambda x: x.index.month_name())
                .groupby("mes")[var]
                .median()
                .reindex(periodo_previsao.month_name())
                .values
            )
        dados_cenarios[var] = dados_cenario

dados_cenarios = pd.DataFrame(dados_cenarios, index=periodo_previsao)

# =========== Previsões
previsao1 = (
    modelo1.predict_interval(steps=h, exog=dados_cenarios, n_boot=5000, random_state=semente)
    .assign(Tipo="Lasso")
    .rename(columns={"pred": "Valor", "lower_bound": "Intervalo Inferior", "upper_bound": "Intervalo Superior"})
)

previsao2 = (
    modelo2.predict_interval(steps=h, exog=dados_cenarios, n_boot=5000, random_state=semente)
    .assign(Tipo="Bayesian Ridge")
    .rename(columns={"pred": "Valor", "lower_bound": "Intervalo Inferior", "upper_bound": "Intervalo Superior"})
)

# =========== IA
y.to_frame().join(x[x_reg]).to_csv("dados/producao_industrial.csv")

prompt = f"""
Assume que estamos em {pd.to_datetime("today").strftime("%d/%m/%Y")}.
Forneça sua melhor previsão para a produção industrial geral no Brasil (variação % mensal), medida pelo IBGE,
para o período de {periodo_previsao.min().strftime("%B %Y")} a {periodo_previsao.max().strftime("%B %Y")}.
Use os dados históricos do arquivo CSV anexo "producao_industrial.csv", onde "prod_ind_geral" é a variável alvo.
Forneça apenas valores numéricos em formato CSV com cabeçalho.
"""

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
modelo_ia = genai.GenerativeModel(model_name="gemini-1.5-pro")
arquivo = genai.upload_file("dados/producao_industrial.csv")

previsao3 = pd.read_csv(
    filepath_or_buffer=StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
    names=["date", "Valor"],
    skiprows=1,
    index_col="date",
    converters={"date": pd.to_datetime}
).assign(Tipo="IA")

# =========== Salvar
pasta = "previsao"
if not os.path.exists(pasta):
    os.makedirs(pasta)
pd.concat([
    y.rename("Valor").to_frame().assign(Tipo="Producao Industrial"),
    previsao1,
    previsao2,
    previsao3
]).to_parquet("previsao/producao_industrial.parquet")