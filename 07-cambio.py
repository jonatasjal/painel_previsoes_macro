# Bibliotecas
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.preprocessing import PowerTransformer
from io import StringIO
import google.generativeai as genai
import pandas as pd
import numpy as np
import os


# Definições e configurações globais
h = 12 # horizonte de previsão
inicio_treino = pd.to_datetime("2004-01-01") # amostra inicial de treinamento
semente = 1984 # semente para reprodução


# Função para transformar dados, conforme definido nos metadados
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


# Planilha de metadados
metadados = (
    pd.read_excel(
        io = "https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name = "Metadados",
        dtype = str,
        index_col = "Identificador"
        )
    .filter(["Transformação"])
)

# Importa dados online
dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")
dados_brutos_t = pd.read_parquet("dados/df_trimestral.parquet")
dados_brutos_a = pd.read_parquet("dados/df_anual.parquet")

# Converte frequência
dados_tratados = (
    dados_brutos_m
    .asfreq("MS")
    .join(
        other = dados_brutos_a.asfreq("MS").ffill(),
        how = "outer"
        )
    .join(
        other = (
            dados_brutos_t
            .filter(["us_gdp", "pib"])
            .dropna()
            .assign(us_gdp = lambda x: ((x.us_gdp.rolling(4).mean() / x.us_gdp.rolling(4).mean().shift(4)) - 1) * 100)
            .asfreq("MS")
            .ffill()
        ),
        how = "outer"
    )
    .rename_axis("data", axis = "index")
)

# Separa Y
y = dados_tratados.cambio.dropna()

# Separa X
x = dados_tratados.drop(labels = "cambio", axis = "columns").copy()

# Computa transformações
for col in x.drop(labels = ["saldo_caged_antigo", "saldo_caged_novo"], axis = "columns").columns.to_list():
  x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

# Filtra amostra
y = y[y.index >= inicio_treino]
x_alem_de_y = x.query("index >= @y.index.max()")
x = x.query("index >= @inicio_treino and index <= @y.index.max()")

# Conta por coluna proporção de NAs em relação ao nº de obs. do IPCA
prop_na = x.isnull().sum() / y.shape[0]

# Remove variáveis que possuem mais de 20% de NAs
x = x.drop(labels = prop_na[prop_na >= 0.2].index.to_list(), axis = "columns")

# Preenche NAs restantes com a vizinhança
x = x.bfill().ffill()


# Seleção final de variáveis
x_reg = [
    "selic",
    "expec_cambio",
    "ic_br_agro",
    "cotacao_petroleo_fmi"
    ]
# + 1 lag


# Reestima os 2 melhores modelos com amostra completa
modelo1 = ForecasterAutoreg(
    regressor = BayesianRidge(),
    lags = 1,
    transformer_y = PowerTransformer(),
    transformer_exog = PowerTransformer()
    )
modelo1.fit(y, x[x_reg])

modelo2 = ForecasterAutoreg(
    regressor = HuberRegressor(),
    lags = 1,
    transformer_y = PowerTransformer(),
    transformer_exog = PowerTransformer()
    )
modelo2.fit(y, x[x_reg])

# Período de previsão fora da amostra
periodo_previsao = pd.date_range(
    start = modelo1.last_window.index[0] + pd.offsets.MonthBegin(1),
    end = modelo1.last_window.index[0] + pd.offsets.MonthBegin(h),
    freq = "MS"
    )

# Coleta dados de expectativas da Selic (selic)
dados_focus_selic = (
    pd.read_csv(
        filepath_or_buffer = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTop5Selic?$filter=Data%20ge%20'{modelo1.last_window.index[0].strftime('%Y-%m-%d')}'%20and%20tipoCalculo%20eq%20'C'&$format=text/csv",
        decimal = ",",
        converters = {
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format = "%m/%Y")
            }
        ))

# Constrói cenário para expectativas de juros (selic)
dados_cenario_selic = (
    dados_focus_selic
    .query("Data == Data.max()")
    .rename(columns = {"mediana": "selic"})
    .head(12)
    .filter(["selic"])
    .set_index(periodo_previsao)
)

# Coleta dados de expectativas do câmbio (expec_cambio)
dados_focus_cambio = (
    pd.read_csv(
        filepath_or_buffer = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?$filter=Indicador%20eq%20'C%C3%A2mbio'%20and%20baseCalculo%20eq%200%20and%20Data%20ge%20'{modelo1.last_window.index[0].strftime('%Y-%m-%d')}'&$format=text/csv",
        decimal = ",",
        converters = {
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format = "%m/%Y")
            }
        ))

# Data do relatório Focus usada para construir cenário para câmbio
data_focus_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .Data
    .value_counts()
    .to_frame()
    .reset_index()
    .query("count == @h")
    .query("Data == Data.max()")
    .Data
    .to_list()[0]
)

# Constrói cenário para câmbio (expec_cambio)
dados_cenario_cambio = (
    dados_focus_cambio
    .query("DataReferencia in @periodo_previsao or DataReferencia == @modelo1.last_window.index[0]")
    .query("Data == @data_focus_cambio")
    .sort_values(by = "DataReferencia")
    .set_index("DataReferencia")
    .filter(["Mediana"])
    .rename(columns = {"Mediana": "expec_cambio"})
    .dropna()
)

# Constrói cenário para commodities (ic_br_agro)
dados_cenario_ic_br_agro = (
    x
    .filter(["ic_br_agro"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes = lambda x: x.index.month_name())
    .groupby(["mes"], as_index = False)
    .ic_br_agro
    .median()
    .set_index("mes")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes = lambda x: x.data.dt.month_name())
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("mes")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)

# Constrói cenário para Petróleo (cotacao_petroleo_fmi)
dados_cenario_cotacao_petroleo_fmi = (
    x
    .filter(["cotacao_petroleo_fmi"])
    .dropna()
    .query("index >= @inicio_treino")
    .assign(mes = lambda x: x.index.month_name())
    .groupby(["mes"], as_index = False)
    .cotacao_petroleo_fmi
    .median()
    .set_index("mes")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(mes = lambda x: x.data.dt.month_name())
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("mes")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)

# Junta cenários e gera dummies sazonais
dados_cenarios = (
    dados_cenario_selic
    .join(
        other = [
            dados_cenario_cambio,
            dados_cenario_ic_br_agro,
            dados_cenario_cotacao_petroleo_fmi
            ],
        how = "outer"
        )
)

# Produz previsões
previsao1 = (
   modelo1.predict_interval(
      steps = h,
      exog = dados_cenarios,
      n_boot = 5000,
      random_state = semente
      )
    .assign(Tipo = "Bayesian Ridge")
    .rename(
       columns = {
          "pred": "Valor", 
          "lower_bound": "Intervalo Inferior", 
          "upper_bound": "Intervalo Superior"
          }
    )
)

previsao2 = (
   modelo2.predict_interval(
      steps = h,
      exog = dados_cenarios,
      n_boot = 5000,
      random_state = semente
      )
    .assign(Tipo = "Huber")
    .rename(
       columns = {
          "pred": "Valor", 
          "lower_bound": "Intervalo Inferior", 
          "upper_bound": "Intervalo Superior"
          }
    )
)

y.to_frame().join(x[x_reg]).to_csv("dados/cambio.csv")
prompt = f"""
Assume that you are in {pd.to_datetime("today").strftime("%B %d, %Y")}. 
Please give me your best forecast of Exchange Rate for Brazil, measured in BRL/USD 
and published by Banco Central do Brasil, for {periodo_previsao.min().strftime("%B %Y")} 
to {periodo_previsao.max().strftime("%B %Y")}. Use the historical Exchange Rate 
data from the attached CSV file named "cambio.csv", where "cambio" is the target 
column, "data" is the date column and the others are exogenous variables. 
Please give me numeric values for these forecasts, in a CSV like format with 
a header, and nothing more. Do not use any information that was not available 
to you as of {pd.to_datetime("today").strftime("%B %d, %Y")} to formulate these 
forecasts.
"""

genai.configure(api_key = os.environ["GEMINI_API_KEY"])
modelo_ia = genai.GenerativeModel(model_name = "gemini-1.5-pro")
arquivo = genai.upload_file("dados/cambio.csv")
previsao3 = pd.read_csv(
    filepath_or_buffer = StringIO(modelo_ia.generate_content([prompt, arquivo]).text),
    names = ["date", "Valor"],
    skiprows = 1,
    index_col = "date",
    converters = {"date": pd.to_datetime}
    ).assign(Tipo = "IA")

# Salvar previsões
pasta = "previsao"
if not os.path.exists(pasta):
  os.makedirs(pasta)
  
pd.concat(
    [y.rename("Valor").to_frame().assign(Tipo = "Câmbio"),
    previsao1,
    previsao2,
    previsao3
    ]).to_parquet("previsao/cambio.parquet")
