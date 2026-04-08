import streamlit as st
import pandas as pd
import numpy as np
import io
import heapq
import itertools

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")
st.write("Faça o upload da sua planilha CSV para gerar os cenários otimizados com foco em Média Global.")

# Menu lateral para estipular as metas dinamicamente
st.sidebar.header("🎯 Parâmetros de Otimização")
meta_ns = st.sidebar.slider("Meta de Nível de Serviço (%)", min_value=0, max_value=100, value=95) / 100.0

limite_prazo = st.sidebar.number_input(
    "Prazo Máximo Aceitável (Média Global Ponderada)", 
    min_value=1.0, 
    value=7.0, 
    step=0.1, 
    format="%.2f",
    help="O motor sacrificará o Nível de Serviço apenas onde o impacto for menor."
)

st.sidebar.info(f"O motor tentará atingir **{meta_ns*100:.0f}% de NS**. Se a média passar de **{limite_prazo:.2f} dias**, ele reduzirá inteligentemente os prazos.")

# ==========================================
# 1. CONFIGURAÇÕES E LIMITES DE NEGÓCIO
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
    {'coluna': 'NS (+1 dia)',  'ajuste': 1},  {'coluna': 'NS (+ 2 dias)','ajuste': 2},
    {'coluna': 'NS (+3 dias)', 'ajuste': 3}
]

# ==========================================
# FUNÇÕES DE CACHE E OTIMIZAÇÃO (STREAMLIT)
# ==========================================

@st.cache_data(show_spinner=False)
def carregar_e_limpar_dados(uploaded_file):
    """Lê o arquivo e vetoriza a limpeza de dados para máxima velocidade."""
    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    
    colunas_ns = [c['coluna'] for c in CENARIOS]
    col_prazo = 'Prazo Prometido (Dias Úteis)' if 'Prazo Prometido (Dias Úteis)' in df.columns else 'Prazo Prometido'
    
    # Colunas que precisam ser convertidas para número
    cols_to_clean = colunas_ns + ['CMU', col_prazo, 'NS Atual']
    
    for col in cols_to_clean:
        if col in df.columns and df[col].dtype == 'object':
            # Limpeza vetorizada (muito mais rápida que apply)
            df[col] = df[col].astype(str).str.replace(r'[R\$\%\s]', '', regex=True)
            df[col] = df[col].str.replace('.', '', regex=False)
            df[col] = df[col].str.replace(',', '.', regex=False)
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if col_prazo in df.columns and col_prazo != 'Prazo Prometido':
        df['Prazo Prometido'] = df[col_prazo]
        
    return df

@st.cache_data(show_spinner=False)
def mapear_pareto_base(df):
    """Pré-calcula a Curva de Pareto para cada CEP UMA ÚNICA VEZ. Não depende das metas."""
    # Transforma em dicionário e ordena para o groupby do itertools (muito mais rápido que pandas groupby iterrows)
    records = df.to_dict('records')
    records.sort(key=lambda x: x['CEP'])
    
    cep_pareto_list = []
    
    for cep, group_iter in itertools.groupby(records, key=lambda x: x['CEP']):
        group = list(group_iter)
        vol_total_cep = sum(r['Qtd Pedidos'] for r in group)
        
        # Pega a transportadora atual (a com mais pedidos)
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
            'vol_total': vol_total_cep, 'transp_ant': transp_ant_row['Transportador'], 
            'prazo_ant': transp_ant_row['Prazo Prometido'], 'qtd_ant': transp_ant_row['Qtd Pedidos'], 
            'cmu_ant': transp_ant_row['CMU'], 'ns_atual_ant': transp_ant_row['NS Atual'], 'pareto': pareto
        })
        
    return cep_pareto_list

def calcular_agregacao_executiva(group):
    # Otimização na agregação usando numpy direto e divisões seguras
    peso_ant = group['qtd pedidos transp anterior'].values
    peso_sel = group['Qtd Pedidos Transportadora Selecionada'].values
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    
    lider_ant = group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax() if vol_ant_total > 0 else group['Transportadora Anterior'].iloc[0]
    lider_sel = group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax() if vol_sel_total > 0 else group['Transportadora Selecionada'].iloc[0]

    prazo_ant = np.dot(group['Prazo Transportadora Anterior'].values, peso_ant) / vol_ant_total if vol_ant_total > 0 else 0
    cmu_ant = np.dot(group['CMU transportadora atual'].values, peso_ant) / vol_ant_total if vol_ant_total > 0 else 0
    prazo_sel = np.dot(group['Prazo Transportadora Selecionada'].values, peso_sel) / vol_sel_total if vol_sel_total > 0 else 0
    cmu_sel = np.dot(group['CMU transportadora selecionada'].values, peso_sel) / vol_sel_total if vol_sel_total > 0 else 0
    ns_proj = np.dot(group['NS Projetado'].values, peso_sel) / vol_sel_total if vol_sel_total > 0 else 0
    ns_atual = np.dot(group['NS Atual'].values, peso_sel) / vol_sel_total if vol_sel_total > 0 else 0

    diff_prazo = prazo_ant - prazo_sel
    acao = f"Redução de {diff_prazo:.1f} dias" if diff_prazo > 0 else (f"Aumento de {-diff_prazo:.1f} dias" if diff_prazo < 0 else "Manter cenário")

    return pd.Series({
        'Transportadora Anterior': lider_ant, 'Prazo Transportadora Anterior': prazo_ant, 'qtd pedidos transp anterior': vol_ant_total,
        'CMU transportadora atual': cmu_ant, 'Transportadora Selecionada': lider_sel, 'Prazo Transportadora Selecionada': prazo_sel,
        'Qtd Pedidos Transportadora Selecionada': vol_sel_total, 'CMU transportadora selecionada': cmu_sel,
        'NS Atual': ns_atual, 'NS Projetado': ns_proj, 'Ação Sugerida': acao
    })

# ==========================================
# 2. INTERFACE E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Mapeando matriz de transporte...'):
        # 1. Carrega e limpa os dados (COM CACHE)
        df = carregar_e_limpar_dados(uploaded_file)
        if 'Prazo Prometido' not in df.columns:
            st.error("Aviso: Coluna de Prazo não encontrada.")
            st.stop()

        # 2. Gera os cenários ideais por CEP (COM CACHE)
        cep_pareto_base = mapear_pareto_base(df)

    # 3. Aplica os filtros dinâmicos de forma quase instantânea
    with st.spinner('Aplicando algoritmos e ponderando limites...'):
        cep_data_list = []
        vol_total_geral = 0
        
        # Aplica o Meta de NS para achar o cenário de partida
        for base_c in cep_pareto_base:
            c = base_c.copy() # Shallow copy para não sujar o cache
            ideal_idx = len(c['pareto']) - 1
            for i, s in enumerate(c['pareto']):
                if s['ns'] >= (meta_ns * 100):
                    ideal_idx = i
                    break
            c['curr_idx'] = ideal_idx
            cep_data_list.append(c)
            vol_total_geral += c['vol_total']

        # 4. Otimização Ponderada (Knapsack via Heap)
        if vol_total_geral > 0:
            current_prazo_sum = sum(c['pareto'][c['curr_idx']]['prazo'] * c['vol_total'] for c in cep_data_list)
            target_prazo_sum = limite_prazo * vol_total_geral
            
            if current_prazo_sum > target_prazo_sum:
                heap = []
                for list_i, c in enumerate(cep_data_list):
                    idx = c['curr_idx']
                    if idx > 0:
                        curr_state, prev_state = c['pareto'][idx], c['pareto'][idx - 1]
                        ns_loss = curr_state['ns'] - prev_state['ns']
                        days_saved = curr_state['prazo'] - prev_state['prazo']
                        if days_saved > 0:
                            metric = ns_loss / days_saved
                            heapq.heappush(heap, (metric, -c['vol_total'], list_i, idx))
                
                while heap and current_prazo_sum > target_prazo_sum:
                    metric, neg_vol, list_i, idx = heapq.heappop(heap)
                    c = cep_data_list[list_i]
                    prev_idx = idx - 1
                    
                    c['curr_idx'] = prev_idx 
                    curr_state, prev_state = c['pareto'][idx], c['pareto'][prev_idx]
                    current_prazo_sum -= (curr_state['prazo'] - prev_state['prazo']) * c['vol_total']
                    
                    if prev_idx > 0:
                        next_curr, next_prev = c['pareto'][prev_idx], c['pareto'][prev_idx - 1]
                        next_days_saved = next_curr['prazo'] - next_prev['prazo']
                        if next_days_saved > 0:
                            next_metric = (next_curr['ns'] - next_prev['ns']) / next_days_saved
                            heapq.heappush(heap, (next_metric, -c['vol_total'], list_i, prev_idx))

        # 5. Reconstrução dos Dados
        rows_out = []
        for c in cep_data_list:
            sel = c['pareto'][c['curr_idx']]
            diff_prazo = c['prazo_ant'] - sel['prazo']
            acao = f"Redução de {diff_prazo:.1f} dias" if diff_prazo > 0 else (f"Aumento de {-diff_prazo:.1f} dias" if diff_prazo < 0 else "Manter cenário")
            
            rows_out.append({
                'CEP': c['cep'], 'Região': c['regiao'], 'UF': c['uf'], 'Cidade': c['cidade'],
                'Transportadora Anterior': c['transp_ant'], 'Prazo Transportadora Anterior': c['prazo_ant'],
                'qtd pedidos transp anterior': c['qtd_ant'], 'CMU transportadora atual': c['cmu_ant'],
                'Transportadora Selecionada': sel['transp'], 'Prazo Transportadora Selecionada': sel['prazo'],
                'Qtd Pedidos Transportadora Selecionada': c['vol_total'], 'CMU transportadora selecionada': sel['cmu'],
                'NS Atual': sel['ns_atual'], 'NS Projetado': sel['ns'], 'Ação Sugerida': acao
            })
            
        df_cep = pd.DataFrame(rows_out)
        df_regiao = df_cep.groupby('Região').apply(calcular_agregacao_executiva).reset_index()
        df_uf = df_cep.groupby('UF').apply(calcular_agregacao_executiva).reset_index()
        df_cidade = df_cep.groupby(['UF', 'Cidade']).apply(calcular_agregacao_executiva).reset_index()

        # Cálculos Globais
        vol_ant_total = df_cep['qtd pedidos transp anterior'].sum()
        prazo_ant = np.dot(df_cep['Prazo Transportadora Anterior'], df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        prazo_sel = np.dot(df_cep['Prazo Transportadora Selecionada'], df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        ns_atual = np.dot(df_cep['NS Atual'], df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        ns_proj = np.dot(df_cep['NS Projetado'], df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        cmu_ant = np.dot(df_cep['CMU transportadora atual'], df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        cmu_sel = np.dot(df_cep['CMU transportadora selecionada'], df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0

    st.success('Otimização da Malha concluída com sucesso!')
    st.subheader("Resultados Globais da Otimização")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prazo Prometido Global (Dias)", f"{prazo_sel:.2f}", f"{prazo_sel - prazo_ant:.2f} dias", delta_color="inverse")
    col2.metric("Nível de Serviço Ponderado (NS)", f"{ns_proj:.1f}%", f"{ns_proj - ns_atual:.1f}%", delta_color="normal")
    col3.metric("Custo Médio Unitário (CMU)", f"R$ {cmu_sel:.2f}", f"R$ {cmu_sel - cmu_ant:.2f}", delta_color="inverse")

    st.markdown("---")

    # Geração do Excel
    cols_padrao = [
        'Transportadora Anterior', 'Prazo Transportadora Anterior', 'qtd pedidos transp anterior',
        'CMU transportadora atual', 'Transportadora Selecionada', 'Prazo Transportadora Selecionada',
        'Qtd Pedidos Transportadora Selecionada', 'CMU transportadora selecionada',
        'NS Atual', 'NS Projetado', 'Ação Sugerida'
    ]
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_cep[['CEP', 'Região', 'UF', 'Cidade'] + cols_padrao].to_excel(writer, sheet_name='Detalhado_CEP', index=False)
        df_regiao[['Região'] + cols_padrao].to_excel(writer, sheet_name='Analise_Regiao', index=False)
        df_uf[['UF'] + cols_padrao].to_excel(writer, sheet_name='Analise_UF', index=False)
        df_cidade[['UF', 'Cidade'] + cols_padrao].to_excel(writer, sheet_name='Analise_Cidade', index=False)
    
    st.download_button(
        label="📥 Baixar Planilha de Resultados Otimizados",
        data=buffer.getvalue(),
        file_name="Resultado_Otimizacao_Global_Ponderada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
