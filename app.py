import streamlit as st
import pandas as pd
import io
import heapq
import itertools

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")

# ==========================================
# GERENCIAMENTO DE ESTADO (PRIORIZAÇÃO)
# ==========================================
if 'prioridade' not in st.session_state:
    st.session_state.prioridade = "NS"

st.sidebar.header("🎯 Parâmetros de Otimização")
st.sidebar.subheader("🚀 Estratégia do Motor")
col_btn_ns, col_btn_prz = st.sidebar.columns(2)

if col_btn_ns.button("🎯 Priorizar NS", use_container_width=True):
    st.session_state.prioridade = "NS"
if col_btn_prz.button("⚡ Priorizar Prazo", use_container_width=True):
    st.session_state.prioridade = "Prazo"

st.sidebar.markdown(f"Modo de Otimização Ativo: **{st.session_state.prioridade}**")
st.sidebar.markdown("---")

st.write(f"Faça o upload da sua planilha CSV para gerar os cenários otimizados com foco em **{st.session_state.prioridade}**.")

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
    help="No modo 'Prazo', o motor sacrificará o NS apenas onde o impacto for menor para atingir esta média."
)

if st.session_state.prioridade == "NS":
    st.sidebar.info(f"O motor garantirá **{meta_ns*100:.2f}% de NS**. O prazo médio será o melhor possível dentro desta meta, sem reduções forçadas.")
else:
    st.sidebar.info(f"O motor tentará o NS de **{meta_ns*100:.2f}%**, mas se a média global passar de **{limite_prazo:.2f} dias**, ele priorizará a redução do prazo.")

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

CAPITAIS = [
    'SÃO PAULO', 'RIO DE JANEIRO', 'BELO HORIZONTE', 'VITÓRIA', 'CURITIBA', 'FLORIANÓPOLIS', 'PORTO ALEGRE',
    'GOIÂNIA', 'BRASÍLIA', 'CUIABÁ', 'CAMPO GRANDE', 'SALVADOR', 'ARACAJU', 'MACEIÓ', 'RECIFE', 'JOÃO PESSOA',
    'NATAL', 'FORTALEZA', 'TERESINA', 'SÃO LUÍS', 'PALMAS', 'BELÉM', 'MACAPÁ', 'MANAUS', 'BOA VISTA', 
    'PORTO VELHO', 'RIO BRANCO'
]

# ==========================================
# FUNÇÕES DE CACHE E OTIMIZAÇÃO BLINDADAS
# ==========================================

def limpar_coluna_segura(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return serie.fillna(0.0)
    s = serie.astype(str).str.replace(r'[R\$\%\s]', '', regex=True)
    mask = s.str.contains(',')
    s.loc[mask] = s.loc[mask].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

@st.cache_data(show_spinner=False)
def carregar_e_limpar_dados(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)
    colunas_ns = [c['coluna'] for c in CENARIOS]
    col_prazo = 'Prazo Prometido (Dias Úteis)' if 'Prazo Prometido (Dias Úteis)' in df.columns else 'Prazo Prometido'
    
    for col in colunas_ns + ['NS Atual']:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            if 0 < df[col].max() <= 1.0:
                df[col] = df[col] * 100

    for col in ['CMU', col_prazo]:
        if col in df.columns:
            df[col] = limpar_coluna_segura(df[col])
            
    if 'Qtd Pedidos' in df.columns:
        df['Qtd Pedidos'] = pd.to_numeric(df['Qtd Pedidos'], errors='coerce').fillna(0)

    if col_prazo in df.columns and col_prazo != 'Prazo Prometido':
        df['Prazo Prometido'] = df[col_prazo]
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
            'vol_total': vol_total_cep, 'transp_ant': transp_ant_row['Transportador'], 
            'prazo_ant': transp_ant_row['Prazo Prometido'], 'qtd_ant': transp_ant_row['Qtd Pedidos'], 
            'cmu_ant': transp_ant_row['CMU'], 'ns_atual_ant': transp_ant_row['NS Atual'], 'pareto': pareto
        })
    return cep_pareto_list

def calcular_agregacao_executiva(group):
    peso_ant, peso_sel = group['qtd pedidos transp anterior'], group['Qtd Pedidos Transportadora Selecionada']
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    
    lider_ant = group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax() if vol_ant_total > 0 else group['Transportadora Anterior'].iloc[0]
    lider_sel = group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax() if vol_sel_total > 0 else group['Transportadora Selecionada'].iloc[0]

    def mp(col, pesos, total): 
        return (group[col] * pesos).sum() / total if total > 0 else group[col].mean()

    return pd.Series({
        'Transportadora Anterior': lider_ant, 'Prazo Transportadora Anterior': mp('Prazo Transportadora Anterior', peso_ant, vol_ant_total), 
        'qtd pedidos transp anterior': vol_ant_total, 'CMU transportadora atual': mp('CMU transportadora atual', peso_ant, vol_ant_total), 
        'Transportadora Selecionada': lider_sel, 'Prazo Transportadora Selecionada': mp('Prazo Transportadora Selecionada', peso_sel, vol_sel_total),
        'Qtd Pedidos Transportadora Selecionada': vol_sel_total, 'CMU transportadora selecionada': mp('CMU transportadora selecionada', peso_sel, vol_sel_total),
        'NS Atual': mp('NS Atual', peso_sel, vol_sel_total), 'NS Projetado': mp('NS Projetado', peso_sel, vol_sel_total), 
        'Ação Sugerida': "Otimizado"
    })

# ==========================================
# 2. INTERFACE E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Mapeando matriz de transporte...'):
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
            
            if st.session_state.prioridade == "NS":
                target_prazo_sum = float('inf') 
            else:
                target_prazo_sum = limite_prazo * vol_total_geral 
            
            if current_prazo_sum > target_prazo_sum:
                heap = []
                for list_i, c in enumerate(cep_data_list):
                    idx = c['curr_idx']
                    if idx > 0:
                        curr_s, prev_s = c['pareto'][idx], c['pareto'][idx - 1]
                        days_saved = curr_s['prazo'] - prev_s['prazo']
                        if days_saved > 0:
                            metric = (curr_s['ns'] - prev_s['ns']) / days_saved
                            heapq.heappush(heap, (metric, -c['vol_total'], list_i, idx))
                
                while heap and current_prazo_sum > target_prazo_sum:
                    metric, neg_vol, list_i, idx = heapq.heappop(heap)
                    c = cep_data_list[list_i]
                    prev_idx = idx - 1
                    current_prazo_sum -= (c['pareto'][c['curr_idx']]['prazo'] - c['pareto'][prev_idx]['prazo']) * c['vol_total']
                    c['curr_idx'] = prev_idx
                    if prev_idx > 0:
                        next_cur, next_pre = c['pareto'][prev_idx], c['pareto'][prev_idx - 1]
                        ds = next_cur['prazo'] - next_pre['prazo']
                        if ds > 0:
                            heapq.heappush(heap, ((next_cur['ns'] - next_pre['ns'])/ds, -c['vol_total'], list_i, prev_idx))

        # Geração de resultados
        rows_out = []
        for c in cep_data_list:
            sel = c['pareto'][c['curr_idx']]
            diff_prazo = c['prazo_ant'] - sel['prazo']
            rows_out.append({
                'CEP': c['cep'], 'Região': c['regiao'], 'UF': c['uf'], 'Cidade': c['cidade'],
                'Transportadora Anterior': c['transp_ant'], 'Prazo Transportadora Anterior': c['prazo_ant'],
                'qtd pedidos transp anterior': c['qtd_ant'], 'CMU transportadora atual': c['cmu_ant'],
                'Transportadora Selecionada': sel['transp'], 'Prazo Transportadora Selecionada': sel['prazo'],
                'Qtd Pedidos Transportadora Selecionada': c['vol_total'], 'CMU transportadora selecionada': sel['cmu'],
                'NS Atual': sel['ns_atual'], 'NS Projetado': sel['ns'], 
                'Ação Sugerida': f"Redução de {diff_prazo:.1f} dias" if diff_prazo > 0 else "Manter/Aumentar"
            })
            
        df_cep = pd.DataFrame(rows_out)
        
        # Agregações
        df_regiao = df_cep.groupby('Região', group_keys=False).apply(calcular_agregacao_executiva).reset_index()
        df_cidade = df_cep.groupby(['UF', 'Cidade'], group_keys=False).apply(calcular_agregacao_executiva).reset_index()
        
        # Filtro de Capitais para a Tela
        df_cidade['Cidade_Upper'] = df_cidade['Cidade'].astype(str).str.upper()
        df_capitais = df_cidade[df_cidade['Cidade_Upper'].isin(CAPITAIS)].copy()
        
        # Visão global
        prazo_ant = (df_cep['Prazo Transportadora Anterior'] * df_cep['qtd pedidos transp anterior']).sum() / vol_total_geral
        prazo_sel = (df_cep['Prazo Transportadora Selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral
        ns_proj = (df_cep['NS Projetado'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral
        cmu_sel = (df_cep['CMU transportadora selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']).sum() / vol_total_geral

    st.success(f'Otimização concluída com foco em {st.session_state.prioridade}!')
    
    st.subheader("🌎 Resultados Globais")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prazo Médio Global", f"{prazo_sel:.2f} d", f"{prazo_sel - prazo_ant:.2f} d", delta_color="inverse")
    col2.metric("NS Ponderado", f"{ns_proj:.1f}%")
    col3.metric("CMU Médio", f"R$ {cmu_sel:.2f}")

    st.markdown("---")
    
    # Exibição das Capitais na Tela
    if not df_capitais.empty:
        st.subheader("🏙️ Detalhamento Antes e Depois - Capitais")
        
        # Formatando a tabela para exibição na tela
        cols_tela = [
            'UF', 'Cidade', 'Prazo Transportadora Anterior', 'Prazo Transportadora Selecionada',
            'NS Atual', 'NS Projetado', 'CMU transportadora atual', 'CMU transportadora selecionada'
        ]
        
        df_capitais_display = df_capitais[cols_tela].rename(columns={
            'Prazo Transportadora Anterior': 'Prazo Antigo',
            'Prazo Transportadora Selecionada': 'Novo Prazo',
            'CMU transportadora atual': 'CMU Antigo',
            'CMU transportadora selecionada': 'Novo CMU'
        })
        
        # Arredondando os valores para ficar visualmente limpo no app
        df_capitais_display = df_capitais_display.round({
            'Prazo Antigo': 1, 'Novo Prazo': 1, 'NS Atual': 1, 'NS Projetado': 1, 'CMU Antigo': 2, 'Novo CMU': 2
        })
        
        st.dataframe(df_capitais_display, use_container_width=True, hide_index=True)

    # Preparação para Download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_cep.to_excel(writer, sheet_name='Detalhado_CEP', index=False)
        df_cidade.drop(columns=['Cidade_Upper'], errors='ignore').to_excel(writer, sheet_name='Por_Cidade', index=False)
        df_regiao.to_excel(writer, sheet_name='Por_Regiao', index=False)
    
    st.download_button("📥 Baixar Planilha Completa (Com Nível Cidade)", data=buffer.getvalue(), file_name="otimizacao.xlsx")
