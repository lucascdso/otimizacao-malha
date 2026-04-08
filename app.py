import streamlit as st
import pandas as pd
import io
import heapq
import itertools

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")
st.write("Faça o upload da sua planilha CSV para gerar os cenários otimizados com foco em Média Global.")

# Menu lateral para estipular as metas dinamicamente
st.sidebar.header("🎯 Parâmetros de Otimização")

# O input de NS agora aceita digitação e números decimais (ex: 95.5)
meta_ns = st.sidebar.number_input(
    "Meta de Nível de Serviço (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=95.0, 
    step=0.1,
    format="%.2f",
    help="Digite o valor exato da meta de Nível de Serviço. Aceita casas decimais (ex: 95.5)."
) / 100.0

limite_prazo = st.sidebar.number_input(
    "Prazo Máximo Aceitável (Média Ponderada)", 
    min_value=1.0, 
    value=7.0, 
    step=0.1, 
    format="%.2f",
    help="O motor sacrificará o Nível de Serviço apenas onde o impacto for menor."
)

st.sidebar.info(f"O motor tentará atingir **{meta_ns*100:.2f}% de NS**. Se a média passar de **{limite_prazo:.2f} dias**, ele reduzirá inteligentemente os prazos.")

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
    {'coluna': 'NS (+1 dia)',  'ajuste': 1},  {'coluna': 'NS (+ 2 dias)', 'ajuste': 2},
    {'coluna': 'NS (+3 dias)', 'ajuste': 3}
]

# ==========================================
# FUNÇÕES DE CACHE E OTIMIZAÇÃO BLINDADAS
# ==========================================

def limpar_coluna_segura(serie):
    """Limpeza vetorizada que imita a segurança da lógica original"""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.fillna(0.0)
    
    s = serie.astype(str)
    # Remove R$, % e espaços
    s = s.str.replace(r'[R\$\%\s]', '', regex=True)
    
    # Apenas remove ponto e troca vírgula por ponto ONDE houver vírgula
    mask = s.str.contains(',')
    s.loc[mask] = s.loc[mask].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

@st.cache_data(show_spinner=False)
def carregar_e_limpar_dados(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    
    colunas_ns = [c['coluna'] for c in CENARIOS]
    col_prazo = 'Prazo Prometido (Dias Úteis)' if 'Prazo Prometido (Dias Úteis)' in df.columns else 'Prazo Prometido'
    
    # Limpeza NS (Garante que decimais 0.95 virem 95.0%)
    for col in colunas_ns + ['NS Atual']:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            # Trava de segurança: Se os valores máximos não passarem de 1.0, multiplica por 100
            if 0 < df[col].max() <= 1.0:
                df[col] = df[col] * 100

    # Limpeza Prazos e CMU
    for col in ['CMU', col_prazo]:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            
    # Garante que os pedidos não quebrem somatórias
    if 'Qtd Pedidos' in df.columns:
        df['Qtd Pedidos'] = pd.to_numeric(df['Qtd Pedidos'], errors='coerce').fillna(0)

    if col_prazo in df.columns and col_prazo != 'Prazo Prometido':
        df['Prazo Prometido'] = df[col_prazo]
        
    return df

@st.cache_data(show_spinner=False)
def mapear_pareto_base(df):
    records = df.to_dict('records')
    # Evita quebra caso a coluna CEP tenha valores vazios (NaN)
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
            'vol_total': vol_total_cep, 'transp_ant': transp_ant_row['Transportador'], 
            'prazo_ant': transp_ant_row['Prazo Prometido'], 'qtd_ant': transp_ant_row['Qtd Pedidos'], 
            'cmu_ant': transp_ant_row['CMU'], 'ns_atual_ant': transp_ant_row['NS Atual'], 'pareto': pareto
        })
        
    return cep_pareto_list

def calcular_agregacao_executiva(group):
    peso_ant = group['qtd pedidos transp anterior']
    peso_sel = group['Qtd Pedidos Transportadora Selecionada']
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    
    lider_ant = group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax() if vol_ant_total > 0 else group['Transportadora Anterior'].iloc[0]
    lider_sel = group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax() if vol_sel_total > 0 else group['Transportadora Selecionada'].iloc[0]

    def mp(col, pesos, total): 
        return (group[col] * pesos).sum() / total if total > 0 else group[col].mean()

    prazo_ant = mp('Prazo Transportadora Anterior', peso_ant, vol_ant_total)
    cmu_ant = mp('CMU transportadora atual', peso_ant, vol_ant_total)
    prazo_sel = mp('Prazo Transportadora Selecionada', peso_sel, vol_sel_total)
    cmu_sel = mp('CMU transportadora selecionada', peso_sel, vol_sel_total)
    ns_proj = mp('NS Projetado', peso_sel, vol_sel_total)
    ns_atual = mp('NS Atual', peso_sel, vol_sel_total)

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
    with st.spinner('Mapeando matriz de transporte e limpando dados...'):
        df = carregar_e_limpar_dados(uploaded_file)
        if 'Prazo Prometido' not in df.columns:
            st.error("Aviso: Coluna de Prazo não encontrada.")
            st.stop()

        cep_pareto_base = mapear_pareto_base(df)

    with st.spinner('Aplicando limites e ponderando otimização...'):
        cep_data_list = []
        vol_total_geral = 0
        
        for base_c in cep_pareto_base:
            c = base_c.copy()
            ideal_idx = len(c['pareto']) - 1
            for i, s in enumerate(c['pareto']):
                if s['ns'] >= (meta_ns * 100):
                    ideal_idx = i
                    break
            c['curr_idx'] = ideal_idx
            cep_data_list.append(c)
            vol_total_geral += c['vol_total']

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

        # Cálculos Globais usando Pandas Puro (Robusto contra valores vazios)
        vol_ant_total = df_cep['qtd pedidos transp anterior'].sum()
        prazo_ant = (df_cep['Prazo Transportadora Anterior'] * df_cep['qtd pedidos transp anterior']).sum() / vol_ant_total if vol_ant_total > 0 else 0
        prazo_sel = (df_cep['Prazo Transportadora Selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral if vol_total_geral > 0 else 0
        ns_atual = (df_cep['NS Atual'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral if vol_total_geral > 0 else 0
        ns_proj = (df_cep['NS Projetado'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral if vol_total_geral > 0 else 0
        cmu_ant = (df_cep['CMU transportadora atual'] * df_cep['qtd pedidos transp anterior']).sum() / vol_ant_total if vol_ant_total > 0 else 0
        cmu_sel = (df_cep['CMU transportadora selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral if vol_total_geral > 0 else 0

    st.success('Otimização da Malha concluída com sucesso!')
    st.subheader("Resultados Globais da Otimização")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prazo Prometido Global (Dias)", f"{prazo_sel:.2f}", f"{prazo_sel - prazo_ant:.2f} dias", delta_color="inverse")
    col2.metric("Nível de Serviço Ponderado (NS)", f"{ns_proj:.1f}%", f"{ns_proj - ns_atual:.1f}%", delta_color="normal")
    col3.metric("Custo Médio Unitário (CMU)", f"R$ {cmu_sel:.2f}", f"R$ {cmu_sel - cmu_ant:.2f}", delta_color="inverse")

    st.markdown("---")

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
