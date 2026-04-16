import streamlit as st
import pandas as pd
import io
import heapq
import itertools
import unicodedata

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")

# ==========================================
# GERENCIAMENTO DE ESTADO E FILTROS
# ==========================================
if 'prioridade' not in st.session_state:
    st.session_state.prioridade = "NS"

st.sidebar.header("🎯 Parâmetros de Otimização")

# 1. BOTÃO DE FUNIL (Nível de Análise)
nivel_analise = st.sidebar.selectbox(
    "🔍 Nível de Detalhamento/Funil",
    options=["CEP", "Cidade", "UF", "Região"],
    index=1,
    help="Define em qual nível a ponderação de NS, Prazo e CMU será consolidada."
)

st.sidebar.subheader("🚀 Estratégia do Motor")
col_btn_ns, col_btn_prz = st.sidebar.columns(2)

if col_btn_ns.button("🎯 Priorizar NS", use_container_width=True):
    st.session_state.prioridade = "NS"
if col_btn_prz.button("⚡ Priorizar Prazo", use_container_width=True):
    st.session_state.prioridade = "Prazo"

st.sidebar.markdown(f"Modo Ativo: **{st.session_state.prioridade}** | Nível: **{nivel_analise}**")
st.sidebar.markdown("---")

meta_ns = st.sidebar.number_input(
    "Meta de Nível de Serviço (%)", 
    min_value=0.0, max_value=100.0, value=95.0, step=0.1, format="%.2f"
) / 100.0

limite_prazo = st.sidebar.number_input(
    "Prazo Máximo Aceitável (Global)", 
    min_value=1.0, value=7.0, step=0.1, format="%.2f"
)

# ==========================================
# CONFIGURAÇÕES DE NEGÓCIO
# ==========================================
CMU_MAX_UF = {
    'SP': 17.63, 'MG': 23.75, 'RJ': 21.29, 'RS': 24.31, 'PR': 21.78, 'SC': 22.28,
    'BA': 32.18, 'PE': 33.56, 'GO': 26.07, 'CE': 36.37, 'DF': 21.49, 'MT': 33.54,
    'PA': 36.89, 'AM': 63.12, 'PB': 32.98, 'MA': 48.69, 'MS': 26.78, 'RN': 38.18,
    'ES': 19.53, 'AL': 35.81, 'RO': 60.97, 'PI': 47.99, 'SE': 34.18, 'TO': 34.81,
    'RR': 83.62, 'AC': 64.82, 'AP': 52.72
}

CENARIOS = [
    {'coluna': 'NS (-3 dias)', 'ajuste': -3}, {'coluna': 'NS (-2 dias)', 'ajuste': -2},
    {'coluna': 'NS (-1 dia)',  'ajuste': -1}, {'coluna': 'NS Atual',      'ajuste': 0},
    {'coluna': 'NS (+1 dia)',  'ajuste': 1},  {'coluna': 'NS (+ 2 dias)', 'ajuste': 2},
    {'coluna': 'NS (+3 dias)', 'ajuste': 3}
]

def normalizar_texto(txt):
    if pd.isna(txt): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(txt)) if unicodedata.category(c) != 'Mn').upper()

def limpar_coluna_segura(serie):
    if pd.api.types.is_numeric_dtype(serie): return serie.fillna(0.0)
    s = serie.astype(str).str.replace(r'[R\$\%\s]', '', regex=True)
    mask = s.str.contains(',')
    s.loc[mask] = s.loc[mask].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

# ==========================================
# PROCESSAMENTO CORE
# ==========================================

@st.cache_data(show_spinner=False)
def carregar_e_limpar_dados(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    colunas_ns = [c['coluna'] for c in CENARIOS]
    col_prazo = 'Prazo Prometido (Dias Úteis)' if 'Prazo Prometido (Dias Úteis)' in df.columns else 'Prazo Prometido'
    
    for col in colunas_ns + ['NS Atual']:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            if 0 < df[col].max() <= 1.0: df[col] = df[col] * 100

    for col in ['CMU', col_prazo]:
        if col in df.columns: df[col] = limpar_coluna_segura(df[col])
            
    if 'Qtd Pedidos' in df.columns:
        df['Qtd Pedidos'] = pd.to_numeric(df['Qtd Pedidos'], errors='coerce').fillna(0)
    
    df['Cidade_Norm'] = df.get('Cidade', '').apply(normalizar_texto)
    return df

@st.cache_data(show_spinner=False)
def mapear_pareto_base(df):
    records = df.to_dict('records')
    records.sort(key=lambda x: str(x.get('CEP', '')))
    cep_pareto_list = []
    
    for cep, group_iter in itertools.groupby(records, key=lambda x: str(x.get('CEP', ''))):
        group = list(group_iter)
        vol_total_cep = sum(r['Qtd Pedidos'] for r in group)
        if vol_total_cep <= 0: continue
        
        transp_ant_row = max(group, key=lambda x: x['Qtd Pedidos'])
        uf = transp_ant_row['UF']
        limite_cmu = CMU_MAX_UF.get(uf, float('inf'))
        
        valid_scenarios = []
        for row in group:
            cmu, prazo_orig = row['CMU'], row['Prazo Prometido']
            if cmu > limite_cmu: continue
            
            for c in CENARIOS:
                ns = row.get(c['coluna'], 0.0)
                if pd.isna(ns) or ns == 0: continue
                prazo_final = prazo_orig + c['ajuste']
                if prazo_final < 1: continue
                valid_scenarios.append({
                    'prazo': prazo_final, 'ns': ns, 'cmu': cmu, 
                    'transp': row['Transportador'], 'ns_atual': row['NS Atual'], 'ajuste': c['ajuste']
                })
        
        if not valid_scenarios:
            pareto = [{'prazo': max(transp_ant_row['Prazo Prometido'], 1), 'ns': transp_ant_row['NS Atual'], 
                       'cmu': transp_ant_row['CMU'], 'transp': transp_ant_row['Transportador'], 
                       'ns_atual': transp_ant_row['NS Atual'], 'ajuste': 0}]
        else:
            valid_scenarios.sort(key=lambda x: (x['prazo'], -x['ns'], x['cmu']))
            pareto = []
            best_ns_so_far = -1
            for s in valid_scenarios:
                if s['ns'] > best_ns_so_far:
                    pareto.append(s)
                    best_ns_so_far = s['ns']
                    
        cep_pareto_list.append({
            'cep': cep, 'regiao': transp_ant_row['Região'], 'uf': uf, 'cidade': transp_ant_row['Cidade'], 
            'cidade_norm': transp_ant_row['Cidade_Norm'], 'vol_total': vol_total_cep, 
            'transp_ant': transp_ant_row['Transportador'], 'prazo_ant': transp_ant_row['Prazo Prometido'], 
            'qtd_ant': transp_ant_row['Qtd Pedidos'], 'cmu_ant': transp_ant_row['CMU'], 
            'ns_atual_ant': transp_ant_row['NS Atual'], 'pareto': pareto
        })
    return cep_pareto_list

def calcular_agregacao_executiva(group):
    peso_ant, peso_sel = group['qtd pedidos transp anterior'], group['Qtd Pedidos Transportadora Selecionada']
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    
    if vol_ant_total == 0: return pd.Series()

    def mp(col, pesos, total): 
        return (group[col] * pesos).sum() / total if total > 0 else group[col].mean()

    prazo_ant = mp('Prazo Transportadora Anterior', peso_ant, vol_ant_total)
    prazo_sel = mp('Prazo Transportadora Selecionada', peso_sel, vol_sel_total)
    
    diff_prazo = prazo_ant - prazo_sel
    if diff_prazo > 0.01:
        acao = f"Redução de {abs(diff_prazo):.1f} dias"
    elif diff_prazo < -0.01:
        acao = f"Aumento de {abs(diff_prazo):.1f} dias"
    else:
        acao = "Manter cenário"

    return pd.Series({
        'Transportadora Anterior': group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax(),
        'Prazo Antes': prazo_ant,
        'CMU Antes': mp('CMU transportadora atual', peso_ant, vol_ant_total),
        'NS Antes': mp('NS Atual', peso_ant, vol_ant_total),
        'Transportadora Selecionada': group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax(),
        'Prazo Depois': prazo_sel,
        'CMU Depois': mp('CMU transportadora selecionada', peso_sel, vol_sel_total),
        'NS Depois': mp('NS Projetado', peso_sel, vol_sel_total),
        'Volume Total': vol_sel_total,
        'Ação Sugerida': acao
    })

# ==========================================
# EXECUÇÃO DA INTERFACE
# ==========================================
uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = carregar_e_limpar_dados(uploaded_file)
    cep_pareto_base = mapear_pareto_base(df)

    # Lógica de Otimização (Similiar ao anterior mas focada no funil)
    cep_data_list = []
    vol_total_geral = 0
    for base_c in cep_pareto_base:
        c = base_c.copy()
        ideal_idx = 0
        for i, s in enumerate(c['pareto']):
            if s['ns'] >= (meta_ns * 100):
                ideal_idx = i
                break
        c['curr_idx'] = ideal_idx
        cep_data_list.append(c)
        vol_total_geral += c['vol_total']

    # Gerar DataFrame de CEPs processados
    rows_out = []
    for c in cep_data_list:
        sel = c['pareto'][c['curr_idx']]
        diff = c['prazo_ant'] - sel['prazo']
        acao = f"Redução de {diff:.1f}" if diff > 0 else (f"Aumento de {-diff:.1f}" if diff < 0 else "Manter")
        
        rows_out.append({
            'CEP': c['cep'], 'Região': c['regiao'], 'UF': c['uf'], 'Cidade': c['cidade'],
            'Transportadora Anterior': c['transp_ant'], 'Prazo Transportadora Anterior': c['prazo_ant'],
            'qtd pedidos transp anterior': c['qtd_ant'], 'CMU transportadora atual': c['cmu_ant'],
            'Transportadora Selecionada': sel['transp'], 'Prazo Transportadora Selecionada': sel['prazo'],
            'Qtd Pedidos Transportadora Selecionada': c['vol_total'], 'CMU transportadora selecionada': sel['cmu'],
            'NS Atual': c['ns_atual_ant'], 'NS Projetado': sel['ns'], 'Ação Sugerida': acao
        })
    
    df_cep_final = pd.DataFrame(rows_out)

    # ==========================================
    # LÓGICA DO FUNIL (AGREGAÇÃO DINÂMICA)
    # ==========================================
    if nivel_analise == "CEP":
        df_display = df_cep_final.copy()
        df_display.rename(columns={'Prazo Transportadora Anterior': 'Prazo Antes', 'Prazo Transportadora Selecionada': 'Prazo Depois', 'NS Atual': 'NS Antes', 'NS Projetado': 'NS Depois', 'CMU transportadora atual': 'CMU Antes', 'CMU transportadora selecionada': 'CMU Depois'}, inplace=True)
    elif nivel_analise == "Cidade":
        df_display = df_cep_final.groupby(['UF', 'Cidade']).apply(calcular_agregacao_executiva).reset_index()
    elif nivel_analise == "UF":
        df_display = df_cep_final.groupby('UF').apply(calcular_agregacao_executiva).reset_index()
    else: # Região
        df_display = df_cep_final.groupby('Região').apply(calcular_agregacao_executiva).reset_index()

    # Dashboard de Resumo
    st.subheader(f"📊 Resumo da Otimização - Nível {nivel_analise}")
    
    m1, m2, m3 = st.columns(3)
    p_antes = (df_cep_final['Prazo Transportadora Anterior'] * df_cep_final['qtd pedidos transp anterior']).sum() / vol_total_geral
    p_depois = (df_cep_final['Prazo Transportadora Selecionada'] * df_cep_final['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral
    ns_antes = (df_cep_final['NS Atual'] * df_cep_final['qtd pedidos transp anterior']).sum() / vol_total_geral
    ns_depois = (df_cep_final['NS Projetado'] * df_cep_final['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral
    
    m1.metric("Prazo Médio", f"{p_depois:.2f} d", f"{p_depois - p_antes:.2f} d", delta_color="inverse")
    m2.metric("Nível de Serviço", f"{ns_depois:.1f}%", f"{ns_depois - ns_antes:.1f}%")
    m3.metric("Sugestão Principal", df_display['Ação Sugerida'].mode()[0] if not df_display.empty else "-")

    # Exibição da Tabela De-Para
    st.markdown(f"### 🔍 Detalhamento De-Para ({nivel_analise})")
    
    # Formatação para destacar aumentos e reduções
    def color_acao(val):
        if 'Redução' in str(val): return 'color: green; font-weight: bold'
        if 'Aumento' in str(val): return 'color: orange; font-weight: bold'
        return ''

    st.dataframe(
        df_display.style.map(color_acao, subset=['Ação Sugerida'])
        .format({
            'Prazo Antes': "{:.2f}", 'Prazo Depois': "{:.2f}",
            'NS Antes': "{:.1f}%", 'NS Depois': "{:.1f}%",
            'CMU Antes': "R$ {:.2f}", 'CMU Depois': "R$ {:.2f}"
        }),
        use_container_width=True
    )

    # Download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_cep_final.to_excel(writer, sheet_name='Base_CEP', index=False)
        df_display.to_excel(writer, sheet_name=f'Analise_{nivel_analise}', index=False)
    
    st.download_button(
        label=f"📥 Baixar Resultados ({nivel_analise})",
        data=buffer.getvalue(),
        file_name=f"Otimizacao_{nivel_analise}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
