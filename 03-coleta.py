# ==========================================================
# COLETA AUTOMÁTICA DE DADOS DE MÚLTIPLAS FONTES ECONÔMICAS
# Usa metadados para identificar quais APIs chamar e organiza
# todos os dados brutos em dicionários por frequência.
# ==========================================================


# ---- Lê planilha de metadados que orienta toda a coleta ----
# df_metadados = pd.read_excel(
#     io = "https://docs.google.com/spreadsheets/d/1Dy317e6l_TbKlR7rokg-IWLdg0K42xQyH9yFkNnKgYw/edit?usp=sharing",
#     sheet_name = "Metadados"
# )

file_id = "1Dy317e6l_TbKlR7rokg-IWLdg0K42xQyH9yFkNnKgYw"
export_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

df_metadados = pd.read_excel(
    io = export_url,
    sheet_name = "Metadados",
    dtype = str
)


# ==========================================================
# COLETA: Banco Central / SGS
# ==========================================================

# Filtra apenas séries cuja fonte é BCB/SGS e coleta via API
input_bcb_sgs = (
    df_metadados
    .query("Fonte == 'BCB/SGS' and `Forma de Coleta` == 'API'")
    .reset_index(drop=True)
)

# Armazena dados brutos separados por frequência
df_bruto_bcb_sgs = {"Diária": [], "Mensal": [], "Trimestral": [], "Anual": []}

# Loop pelas séries listadas na planilha e coleta cada uma
for serie in input_bcb_sgs.index:
    ser = input_bcb_sgs.iloc[serie]

    # Chama função de coleta da API/SGS
    df_temp = coleta_bcb_sgs(
        codigo = ser["Input de Coleta"],
        nome = ser["Identificador"], 
        freq = ser["Frequência"]
    )

    # Armazena na frequência correta
    df_bruto_bcb_sgs[ser["Frequência"]].append(df_temp)


# ==========================================================
# COLETA: Banco Central / ODATA
# ==========================================================

# Filtra séries da API ODATA do BCB
input_bcb_odata = (
    df_metadados
    .query("Fonte == 'BCB/ODATA' and `Forma de Coleta` == 'API'")
    .reset_index(drop=True)
)

df_bruto_bcb_odata = []

# Faz a coleta de cada série
for serie in input_bcb_odata.index:
    ser = input_bcb_odata.iloc[serie]

    df_temp = coleta_bcb_odata(
        codigo = ser["Input de Coleta"],
        nome = ser["Identificador"]
    )

    df_bruto_bcb_odata.append(df_temp)


# ==========================================================
# COLETA: IPEADATA
# ==========================================================

# Filtra séries que vêm da API IPEADATA
input_ipeadata = (
    df_metadados
    .query("Fonte == 'IPEADATA' and `Forma de Coleta` == 'API'")
    .reset_index(drop=True)
)

df_bruto_ipeadata = {"Diária": [], "Mensal": []}

# Loop de coleta
for serie in input_ipeadata.index:
    ser = input_ipeadata.iloc[serie]

    df_temp = coleta_ipeadata(
        codigo = ser["Input de Coleta"],
        nome = ser["Identificador"]
    )

    df_bruto_ipeadata[ser["Frequência"]].append(df_temp)


# ==========================================================
# COLETA: IBGE / SIDRA
# ==========================================================

# Filtra séries da API SIDRA (IBGE)
input_sidra = (
    df_metadados
    .query("Fonte == 'IBGE/SIDRA' and `Forma de Coleta` == 'API'")
    .reset_index(drop=True)
)

df_bruto_ibge_sidra = {"Mensal": [], "Trimestral": []}

# Loop por cada série definida na planilha
for serie in input_sidra.index:
    ser = input_sidra.iloc[serie]

    df_temp = coleta_ibge_sidra(
        codigo = ser["Input de Coleta"],
        nome = ser["Identificador"]
    )

    df_bruto_ibge_sidra[ser["Frequência"]].append(df_temp)


# ==========================================================
# COLETA: FRED (Federal Reserve)
# ==========================================================

# Filtra séries que vêm da API FRED
input_fred = (
    df_metadados
    .query("Fonte == 'FRED' and `Forma de Coleta` == 'API'")
    .reset_index(drop=True)
)

df_bruto_fred = {"Diária": [], "Mensal": [], "Trimestral": []}

# Loop coletando cada série
for serie in input_fred.index:
    ser = input_fred.iloc[serie]

    df_temp = coleta_fred(
        codigo = ser["Input de Coleta"],
        nome = ser["Identificador"]
    )

    df_bruto_fred[ser["Frequência"]].append(df_temp)


# ==========================================================
# COLETA: IFI (Hiato do Produto via Excel)
# ==========================================================

# Filtra linha correspondente à série do IFI
input_ifi = (
    df_metadados
    .query("Fonte == 'IFI'")
    .reset_index(drop=True)
)

# Coleta única planilha fornecida pela IFI
df_bruto_ifi = coleta_ifi(
    input_ifi["Input de Coleta"][0],
    input_ifi["Identificador"][0]
)
