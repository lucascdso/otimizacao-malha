import streamlit as st
import pandas as pd
import io
import heapq
import itertools
import unicodedata

# Configuração da página
st.set_page_config(page_title="Otimização de Malha V2", layout="wide")
st.title("📦 Otimização de Transportes e Gestão de Performance")

# ==========================================
# GERENCIAMENTO DE ESTADO E PARÂMETROS
# ==========================================
if 'prioridade' not in st.session_state:
    st.session_state.prioridade = "NS"

st.sidebar.header("🎯 Parâmetros de Otimização")

# --- NOVO: Seletor de Granularidade ---
nivel_analise = st.sidebar.selectbox(
    "Nível de Granularidade da Análise",
    options=["CEP", "Cidade", "UF", "Região"],
    index=1  # Padrão: Cidade
)

st.sidebar.subheader("🚀 Estratégia do Motor")
col_btn_ns, col_btn_prz = st.sidebar.columns(2)

if col_btn_ns.button("🎯 Priorizar NS", use_container_width=True):
    st.session_state.prioridade = "NS"
if col_btn_prz.button("⚡ Priorizar Prazo", use_container_width=True):
    st.session_state.prioridade = "Prazo"

meta_ns = st.sidebar.number_input(
    "Meta de Nível de Serviço (%)", 
    min_value=0.0, max_value=100.0, value=95.0, step=0.1
) / 100.0

limite_prazo = st.sidebar.number_input(
    "Prazo Máximo Aceitável (Média)", 
    min_value=1.0, value=7.0, step=0.1
)

# ==========================================
# CONFIGURAÇÕES E UTILITÁRIOS
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
    s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

# ==========================================
# MOTOR DE PROCESSAMENTO AGRUPADO
# ==========================================
@st.cache_data(show_spinner=False)
def processar_otimizacao(uploaded_file, nivel):
    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    
    # 1. Limpeza de colunas
    colunas_ns = [c['coluna'] for c in CENARIOS]
    for col in colunas_ns:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            if df[col].max() <= 1.0: df[col] *= 100
    
    df['CMU'] = limpar_coluna_segura(df['CMU'])
    df['Prazo Prometido'] = limpar_coluna_segura(df['Prazo Prometido'])
    df['Qtd Pedidos'] = pd.to_numeric(df['Qtd Pedidos'], errors='coerce').fillna(0)
    df['Cidade_Norm'] = df['Cidade'].apply(normalizar_texto)

    # 2. Definição da chave de agrupamento
    if nivel == "CEP": chave = ['CEP']
    elif nivel == "Cidade": chave = ['UF', 'Cidade']
    elif nivel == "UF": chave = ['UF']
    else: chave = ['Região']

    # 3. Agrupamento por Nível + Transportador (Ponderação)
    def ponderar(x):
        d = {}
        vol = x['Qtd Pedidos'].sum()
        d['vol_total'] = vol
        d['CMU'] = (x['CMU'] * x['Qtd Pedidos']).sum() / vol if vol > 0 else 0
        d['Prazo Prometido'] = (x['Prazo Prometido'] * x['Qtd Pedidos']).sum() / vol if vol > 0 else 0
        for c in colunas_ns:
            d[c] = (x[c] * x['Qtd Pedidos']).sum() / vol if vol > 0 else 0
        return pd.Series(d)

    df_grouped = df.groupby(chave + ['Transportador']).apply(ponderar).reset_index()

    # 4. Mapeamento de Pareto por Localidade
    resultados_pareto = []
    for loc, group in df_grouped.groupby(chave):
        loc_info = loc if isinstance(loc, tuple) else (loc,)
        uf_ref = group['UF'].iloc[0] if 'UF' in group.columns else "N/A"
        limite_cmu = CMU_MAX_UF.get(uf_ref, 999)
        
        # Identificar transportadora atual (maior volume original)
        transp_atual_row = group.loc[group['vol_total'].idxmax()]
        
        valid_options = []
        for _, row in group.iterrows():
            if row['CMU'] > limite_cmu * 1.2: continue # Margem de segurança CMU
            
            for c in CENARIOS:
                ns_val = row[c['coluna']]
                if ns_val <= 0: continue
                
                valid_options.append({
                    'transp': row['Transportador'],
                    'prazo': max(row['Prazo Prometido'] + c['ajuste'], 1),
                    'ns': ns_val,
                    'cmu': row['CMU'],
                    'ns_max_possivel': row['NS (+3 dias)'] # Para lógica de bloqueio
                })
        
        if not valid_options: continue
        
        # Criar Pareto: ordenar por prazo, depois NS
        valid_options.sort(key=lambda x: (x['prazo'], -x['ns'], x['cmu']))
        pareto = []
        best_ns = -1
        for opt in valid_options:
            if opt['ns'] > best_ns:
                pareto.append(opt)
                best_ns = opt['ns']
        
        resultados_pareto.append({
            'chave': loc,
            'uf': uf_ref,
            'vol': group['vol_total'].sum(),
            'transp_ant': transp_atual_row['Transportador'],
            'prazo_ant': transp_atual_row['Prazo Prometido'],
            'cmu_ant': transp_atual_row['CMU'],
            'ns_ant': transp_atual_row['NS Atual'] if 'NS Atual' in transp_atual_row else transp_atual_row['NS (-3 dias)'],
            'pareto': pareto
        })
        
    return resultados_pareto

# ==========================================
# EXECUÇÃO DA LÓGICA
# ==========================================
uploaded_file = st.file_uploader("Suba seu arquivo CSV", type=["csv"])

if uploaded_file:
    data_pareto = processar_otimizacao(uploaded_file, nivel_analise)
    
    final_rows = []
    bloqueios = []
    
    for item in data_pareto:
        pareto = item['pareto']
        # 1. Encontrar o cenário que atende a Meta de NS
        sel_idx = len(pareto) - 1
        for i, s in enumerate(pareto):
            if s['ns'] >= (meta_ns * 100):
                sel_idx = i
                break
        
        selecionado = pareto[sel_idx]
        
        # --- LÓGICA DE BLOQUEIO ---
        # Se a transportadora selecionada NÃO atinge a meta mesmo com +3 dias
        if selecionado['ns_max_possivel'] < (meta_ns * 100):
            # Tenta buscar QUALQUER outra no pareto que performe melhor, mesmo que mais cara
            melhor_alternativa = max(pareto, key=lambda x: x['ns'])
            
            bloqueios.append({
                'Localidade': item['chave'],
                'Transportadora Crítica': selecionado['transp'],
                'NS Máximo (+3d)': f"{selecionado['ns_max_possivel']:.1f}%",
                'Sugestão': "BLOQUEIO",
                'Substituta': melhor_alternativa['transp'] if melhor_alternativa['transp'] != selecionado['transp'] else "Nenhuma disponível"
            })
        
        final_rows.append({
            'Localidade': item['chave'],
            'UF': item['uf'],
            'Vol': item['vol'],
            'Transp. Anterior': item['transp_ant'],
            'Prazo Ant.': item['prazo_ant'],
            'CMU Ant.': item['cmu_ant'],
            'Transp. Selecionada': selecionado['transp'],
            'Prazo Novo': selecionado['prazo'],
            'CMU Novo': selecionado['cmu'],
            'NS Projetado': selecionado['ns'],
            'Status': "⚠️ Risco NS" if selecionado['ns_max_possivel'] < (meta_ns * 100) else "✅ OK"
        })

    df_final = pd.DataFrame(final_rows)

    # Exibição de Métricas
    st.subheader(f"📊 Resultados por {nivel_analise}")
    
    m1, m2, m3 = st.columns(3)
    avg_ns = (df_final['NS Projetado'] * df_final['Vol']).sum() / df_final['Vol'].sum()
    avg_prazo = (df_final['Prazo Novo'] * df_final['Vol']).sum() / df_final['Vol'].sum()
    
    m1.metric("NS Médio Projetado", f"{avg_ns:.1f}%")
    m2.metric("Prazo Médio", f"{avg_prazo:.2f} d")
    m3.metric("Localidades com Risco", len(bloqueios))

    # Tabela de Bloqueios Sugeridos
    if bloqueios:
        st.error("🚨 Recomendações de Bloqueio por Baixa Performance")
        st.table(pd.DataFrame(bloqueios))

    # Tabela Geral
    st.write("### Detalhamento da Otimização")
    st.dataframe(df_final.style.format({
        'Prazo Ant.': "{:.1f}", 'Prazo Novo': "{:.1f}",
        'CMU Ant.': "R$ {:.2f}", 'CMU Novo': "R$ {:.2f}",
        'NS Projetado': "{:.1f}%"
    }), use_container_width=True)

    # Download
    buffer = io.BytesIO()
    df_final.to_excel(buffer, index=False)
    st.download_button("📥 Baixar Relatório Completo", buffer.getvalue(), file_name="otimizacao_logistica.xlsx")
