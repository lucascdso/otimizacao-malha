import streamlit as st
import pandas as pd
import numpy as np
import io
import heapq

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")
st.write("Faça o upload da sua planilha CSV para gerar os cenários otimizados com foco em Média Global.")

# Menu lateral para estipular as metas dinamicamente
st.sidebar.header("🎯 Parâmetros de Otimização")
meta_ns = st.sidebar.slider("Meta de Nível de Serviço (%)", min_value=0, max_value=100, value=95) / 100.0

# O limite de prazo agora aceita números decimais para a ponderação geral (ex: 7.5 dias)
limite_prazo = st.sidebar.number_input(
    "Prazo Máximo Aceitável (Média Global Ponderada em Dias Úteis)", 
    min_value=1.0, 
    value=7.0, 
    step=0.1, 
    format="%.2f",
    help="O motor sacrificará o Nível de Serviço apenas onde o impacto for menor, até que a média ponderada de toda a malha atinja este limite numérico exato."
)

st.sidebar.info(f"O motor tentará atingir **{meta_ns*100:.0f}% de NS** para a malha. Se a média final passar de **{limite_prazo:.2f} dias úteis**, ele reduzirá inteligentemente os prazos onde houver a menor perda de eficiência.")

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
    {'coluna': 'NS (-1 dia)',  'ajuste': -1}, {'coluna': 'NS Atual',     'ajuste': 0},
    {'coluna': 'NS (+1 dia)',  'ajuste': 1},  {'coluna': 'NS (+ 2 dias)','ajuste': 2},
    {'coluna': 'NS (+3 dias)', 'ajuste': 3}
]

# ==========================================
# FUNÇÃO BLINDADA DE LIMPEZA DE NÚMEROS
# ==========================================
def limpar_numero(valor):
    if pd.isna(valor): return 0.0
    if isinstance(valor, (int, float)): return float(valor)
    v = str(valor).replace('R$', '').replace('%', '').strip()
    if ',' in v:
        v = v.replace('.', '')
        v = v.replace(',', '.')
    try:
        return float(v)
    except:
        return 0.0

# ==========================================
# MOTOR DE OTIMIZAÇÃO GLOBAL (ALGORITMO GULOSO)
# ==========================================
def preparar_cenarios_cep(group, limite_cmu_dict, meta_ns_val):
    """Mapeia e cria uma Curva de Pareto para cada CEP: Otimiza as melhores combinações de Prazo vs NS"""
    vol_total_cep = group['Qtd Pedidos'].sum()
    transp_ant_row = group.sort_values(by='Qtd Pedidos', ascending=False).iloc[0]
    
    transp_ant, prazo_ant, qtd_ant = transp_ant_row['Transportador'], transp_ant_row['Prazo Prometido'], transp_ant_row['Qtd Pedidos']
    cmu_ant, ns_atual_ant = transp_ant_row['CMU'], transp_ant_row['NS Atual']
    uf, regiao, cidade, cep = transp_ant_row['UF'], transp_ant_row['Região'], transp_ant_row['Cidade'], transp_ant_row['CEP']
    
    limite_cmu = limite_cmu_dict.get(uf, float('inf'))
    
    valid_scenarios = []
    # Explora todas as transportadoras e todos os cenários disponíveis para o CEP
    for idx, row in group.iterrows():
        cmu, prazo_orig, transp, ns_atual = row['CMU'], row['Prazo Prometido'], row['Transportador'], row['NS Atual']
        if cmu > limite_cmu: continue
        
        for c in CENARIOS:
            ns, ajuste = row[c['coluna']], c['ajuste']
            prazo_final = prazo_orig + ajuste
            if prazo_final < 1 or pd.isna(ns): continue # Garante o mínimo de 1 dia de prazo
            
            valid_scenarios.append({
                'prazo': prazo_final, 'ns': ns, 'cmu': cmu, 'transp': transp, 'ns_atual': ns_atual, 'ajuste': ajuste
            })
    
    if not valid_scenarios:
        # Fallback de segurança caso nenhuma opção seja viável no Estado
        pareto = [{'prazo': max(prazo_ant, 1), 'ns': ns_atual_ant, 'cmu': cmu_ant, 'transp': transp_ant, 'ns_atual': ns_atual_ant, 'ajuste': 0}]
    else:
        # Cria a Curva de Pareto (Elimina opções que demoram mais para entregar o mesmo ou pior NS)
        valid_scenarios.sort(key=lambda x: (x['prazo'], -x['ns'], x['cmu']))
        pareto = []
        best_ns_so_far = -1
        for s in valid_scenarios:
            if s['ns'] > best_ns_so_far:
                pareto.append(s)
                best_ns_so_far = s['ns']
                
    # O cenário ideal é o menor prazo possível que ainda atinja a Meta de NS estipulada
    ideal_idx = len(pareto) - 1
    for i, s in enumerate(pareto):
        if s['ns'] >= (meta_ns_val * 100):
            ideal_idx = i
            break
            
    return {
        'cep': cep, 'regiao': regiao, 'uf': uf, 'cidade': cidade, 'vol_total': vol_total_cep,
        'transp_ant': transp_ant, 'prazo_ant': prazo_ant, 'qtd_ant': qtd_ant, 'cmu_ant': cmu_ant,
        'ns_atual_ant': ns_atual_ant, 'pareto': pareto, 'curr_idx': ideal_idx
    }

def calcular_agregacao_executiva(group):
    peso_ant, peso_sel = group['qtd pedidos transp anterior'], group['Qtd Pedidos Transportadora Selecionada']
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    lider_ant = group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax() if vol_ant_total > 0 else group['Transportadora Anterior'].iloc[0]
    lider_sel = group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax() if vol_sel_total > 0 else group['Transportadora Selecionada'].iloc[0]

    def mp(col, pesos, total): return np.sum(group[col] * pesos) / total if total > 0 else group[col].mean()

    prazo_ant, cmu_ant = mp('Prazo Transportadora Anterior', peso_ant, vol_ant_total), mp('CMU transportadora atual', peso_ant, vol_ant_total)
    prazo_sel, cmu_sel = mp('Prazo Transportadora Selecionada', peso_sel, vol_sel_total), mp('CMU transportadora selecionada', peso_sel, vol_sel_total)
    ns_proj, ns_atual = mp('NS Projetado', peso_sel, vol_sel_total), mp('NS Atual', peso_sel, vol_sel_total)

    diff_prazo = prazo_ant - prazo_sel
    acao = f"Redução média de {diff_prazo:.1f} dias úteis" if diff_prazo > 0 else (f"Aumento médio de {-diff_prazo:.1f} dias úteis" if diff_prazo < 0 else "Manter cenário original")

    res = {
        'Transportadora Anterior': lider_ant, 'Prazo Transportadora Anterior': prazo_ant,
        'qtd pedidos transp anterior': vol_ant_total, 'CMU transportadora atual': cmu_ant,
        'Transportadora Selecionada': lider_sel, 'Prazo Transportadora Selecionada': prazo_sel,
        'Qtd Pedidos Transportadora Selecionada': vol_sel_total, 'CMU transportadora selecionada': cmu_sel,
        'NS Atual': ns_atual, 'NS Projetado': ns_proj, 'Ação Sugerida': acao
    }
    return pd.Series(res)

# ==========================================
# 2. INTERFACE E PROCESSAMENTO
# ==========================================
uploaded_file = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Mapeando matriz de transporte e otimizando algoritmos...'):
        
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)

        colunas_ns = [c['coluna'] for c in CENARIOS]
        for col in colunas_ns:
            if col in df.columns: df[col] = df[col].apply(limpar_numero)

        if 'CMU' in df.columns: df['CMU'] = df['CMU'].apply(limpar_numero)
        
        # Validando e focando no Prazo em Dias Úteis
        col_prazo = 'Prazo Prometido (Dias Úteis)' if 'Prazo Prometido (Dias Úteis)' in df.columns else 'Prazo Prometido'
        if col_prazo in df.columns:
            df['Prazo Prometido'] = df[col_prazo].apply(limpar_numero)
        else:
            st.error("Aviso: Coluna de Prazo não encontrada.")
            st.stop()

        # ----------------------------------------------------------------------------------
        # PASSO 1: Levantar Cenários Ideais por CEP
        # ----------------------------------------------------------------------------------
        df_grouped = df.groupby('CEP')
        cep_data_list = []
        for cep, group in df_grouped:
            cep_data_list.append(preparar_cenarios_cep(group, CMU_MAX_UF, meta_ns))
            
        vol_total_geral = sum(c['vol_total'] for c in cep_data_list)
        
        # ----------------------------------------------------------------------------------
        # PASSO 2: Otimização Ponderada (Knapsack Problem via Heap)
        # ----------------------------------------------------------------------------------
        if vol_total_geral > 0:
            current_prazo_sum = sum(c['pareto'][c['curr_idx']]['prazo'] * c['vol_total'] for c in cep_data_list)
            target_prazo_sum = limite_prazo * vol_total_geral
            
            if current_prazo_sum > target_prazo_sum:
                heap = []
                # Popula a Fila de Prioridade analisando o "Custo-Benefício" de rebaixar cada CEP
                for list_i, c in enumerate(cep_data_list):
                    idx = c['curr_idx']
                    if idx > 0:
                        curr_state, prev_state = c['pareto'][idx], c['pareto'][idx - 1]
                        ns_loss = curr_state['ns'] - prev_state['ns']
                        days_saved = curr_state['prazo'] - prev_state['prazo']
                        
                        if days_saved > 0:
                            # Métrica chave: Quanto de NS nós perdemos por CADA DIA salvo? (Menor é melhor)
                            metric = ns_loss / days_saved
                            heapq.heappush(heap, (metric, -c['vol_total'], list_i, idx))
                
                # Desce os prazos estrategicamente até atingir a média global estipulada
                while heap and current_prazo_sum > target_prazo_sum:
                    metric, neg_vol, list_i, idx = heapq.heappop(heap)
                    c = cep_data_list[list_i]
                    prev_idx = idx - 1
                    
                    c['curr_idx'] = prev_idx # Aplica o rebaixamento de prazo neste CEP
                    
                    curr_state, prev_state = c['pareto'][idx], c['pareto'][prev_idx]
                    days_saved_total = (curr_state['prazo'] - prev_state['prazo']) * c['vol_total']
                    current_prazo_sum -= days_saved_total
                    
                    # Se este CEP ainda puder ser reduzido novamente, recalcula o custo e volta pra fila
                    if prev_idx > 0:
                        next_curr, next_prev = c['pareto'][prev_idx], c['pareto'][prev_idx - 1]
                        next_ns_loss = next_curr['ns'] - next_prev['ns']
                        next_days_saved = next_curr['prazo'] - next_prev['prazo']
                        if next_days_saved > 0:
                            next_metric = next_ns_loss / next_days_saved
                            heapq.heappush(heap, (next_metric, -c['vol_total'], list_i, prev_idx))

        # ----------------------------------------------------------------------------------
        # PASSO 3: Reconstrução dos Dados com Base nos Resultados Otimizados
        # ----------------------------------------------------------------------------------
        rows_out = []
        for c in cep_data_list:
            sel = c['pareto'][c['curr_idx']]
            diff_prazo = c['prazo_ant'] - sel['prazo']
            acao = f"Redução média de {diff_prazo:.1f} dias" if diff_prazo > 0 else (f"Aumento médio de {-diff_prazo:.1f} dias" if diff_prazo < 0 else "Manter cenário original")
            
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

        # Cálculos Globais para a Tela
        vol_ant_total = df_cep['qtd pedidos transp anterior'].sum()

        prazo_ant = np.sum(df_cep['Prazo Transportadora Anterior'] * df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        prazo_sel = np.sum(df_cep['Prazo Transportadora Selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        ns_atual = np.sum(df_cep['NS Atual'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        ns_proj = np.sum(df_cep['NS Projetado'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0
        cmu_ant = np.sum(df_cep['CMU transportadora atual'] * df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        cmu_sel = np.sum(df_cep['CMU transportadora selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total_geral if vol_total_geral > 0 else 0

    st.success('Otimização da Malha concluída com sucesso!')
    st.subheader("Resultados Globais da Otimização")

    col1, col2, col3 = st.columns(3)
    col1.metric("Prazo Prometido Global (Dias Úteis)", f"{prazo_sel:.2f}", f"{prazo_sel - prazo_ant:.2f} dias", delta_color="inverse")
    col2.metric("Nível de Serviço Ponderado (NS)", f"{ns_proj:.1f}%", f"{ns_proj - ns_atual:.1f}%", delta_color="normal")
    col3.metric("Custo Médio Unitário (CMU)", f"R$ {cmu_sel:.2f}", f"R$ {cmu_sel - cmu_ant:.2f}", delta_color="inverse")

    st.markdown("---")

    # Geração do Excel em Memória para Download
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
        label="📥 Baixar Planilha de Resultados Otimizados (Excel)",
        data=buffer.getvalue(),
        file_name="Resultado_Otimizacao_Global_Ponderada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
