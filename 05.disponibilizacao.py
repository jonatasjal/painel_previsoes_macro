# ==========================================================
# MÓDULO DE SALVAMENTO: ORGANIZA E SERIALIZA DADOS POR FREQUÊNCIA
# Objetivo: juntar os DataFrames tratados por frequência (diária,
# mensal, trimestral, anual), filtrar horizonte mínimo e salvar em
# arquivos parquet dentro da pasta 'dados'.
# ==========================================================


# Cria pasta dados se não existir
pasta = "dados"
if not os.path.exists(pasta):
  os.makedirs(pasta)   # garante diretório de saída para os arquivos


# =========================
# Série: Diária
# =========================
# - Junta séries diárias do BCB/SGS, IPEA e FRED usando outer join
# - Normaliza índice de data e filtra a partir de 2000-01-01
# - Salva resultado em parquet
df_diaria = (
    df_tratado_bcb_sgs["Diária"]
    .join(
        other=df_tratado_ipeadata["Diária"].reset_index().assign(
            # formata corretamente as datas vindas do IPEA e garante timestamp
            data=lambda x: pd.to_datetime(x['data'].dt.strftime("%Y-%m-%d"))
        ).set_index("data"),
        how="outer"
    )
    .join(other=df_tratado_fred["Diária"], how="outer")  # junta com FRED (diário)
    .reset_index()
    .assign(data=lambda x: pd.to_datetime(x['data']))      # assegura tipo datetime
    .query("data >= @pd.to_datetime('2000-01-01')")        # filtra janela mínima
    .set_index('data')                                     # usa data como índice
)
df_diaria.to_parquet(f"{pasta}/df_diaria.parquet")        # salva parquet


# =========================
# Série: Mensal
# =========================
# - Monta lista de DataFrames mensais de diversas fontes e faz outer join
# - Filtra a partir de 2000-01-01 e força tipo float para compatibilidade
# - Salva em parquet
temp_lista = [
    df_tratado_bcb_sgs["Mensal"],
    df_tratado_bcb_odata_mensal,
    df_tratado_ipeadata["Mensal"],
    df_tratado_ibge_sidra["Mensal"],
    df_tratado_fred["Mensal"]
]

df_mensal = (
  temp_lista[0]
  .join(other = temp_lista[1:], how = "outer")            # junta todas as fontes mensais
  .query("index >= @pd.to_datetime('2000-01-01')")        # filtra por data mínima
  .astype(float)                                          # uniformiza tipo numérico
)
df_mensal.to_parquet(f"{pasta}/df_mensal.parquet")        # salva parquet


# =========================
# Série: Trimestral
# =========================
# - Agrupa DataFrames trimestrais (inclui PIB do BCB/ODATA e IFI)
# - Filtra a partir de 2000-01-01, força float e garante índice datetime
# - Salva em parquet
temp_lista = [
    df_tratado_bcb_sgs["Trimestral"],
    df_tratado_bcb_odata_pib.set_index("data"),
    df_tratado_ibge_sidra["Trimestral"],
    df_tratado_fred["Trimestral"],
    df_tratado_ifi
]

df_trimestral = (
  temp_lista[0]
  .join(other = temp_lista[1:], how = "outer")            # junta todas as fontes trimestrais
  .query("index >= @pd.to_datetime('2000-01-01')")        # filtra por data mínima
  .astype(float)                                          # uniformiza tipo numérico
)
df_trimestral.index = pd.to_datetime(df_trimestral.index) # garante índice datetime
df_trimestral.to_parquet(f"{pasta}/df_trimestral.parquet")# salva parquet


# =========================
# Série: Anual
# =========================
# - Seleciona séries anuais do BCB/SGS, filtra e força tipo float
# - Salva em parquet
df_anual = (
  df_tratado_bcb_sgs["Anual"]
  .query("index >= @pd.to_datetime('2000-01-01')")        # filtra por data mínima
  .astype(float)                                          # uniformiza tipo numérico
)
df_anual.to_parquet(f"{pasta}/df_anual.parquet")          # salva parquet
