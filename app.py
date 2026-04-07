import streamlit as st
import pandas as pd
import numpy as np
import io

# Configuração da página
st.set_page_config(page_title="Otimização de Malha", layout="wide")
st.title("📦 Análise de Otimização de Transportes")
st.write("Faça o upload da sua planilha CSV para gerar os cenários otimizados.")

# Menu lateral para estipular as metas dinamicamente
st.sidebar.header("🎯 Parâmetros de Otimização")
meta_ns = st.sidebar.slider("Meta de Nível de Serviço (%)", min_value=0, max_value=100, value=95) / 100.0
limite_prazo = st.sidebar.number_input("Prazo Máximo Aceitável (Dias Úteis)", min_value=1, value=7)

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
    """Função inteligente para limpar R$, %, pontos de milhar e converter para número"""
    if pd.isna(valor):
        return 0.0
    if isinstance(valor, (int, float)):
        return float(valor)
    
    # Transforma em string e limpa lixos visuais
    v = str(valor).replace('R$', '').replace('%', '').strip()
    
    # Se o número tem vírgula (padrão Brasil: 1.596,81)
    if ',' in v:
        v = v.replace('.', '')  # Remove o ponto de milhar -> 1596,81
        v = v.replace(',', '.') # Troca a vírgula decimal por ponto -> 1596.81
        
    try:
        return float(v)
    except:
        return 0.0

# Funções do seu motor mantidas intactas, mas conectadas ao menu lateral
def avaliar_cenarios_cep(group):
    vol_total_cep = group['Qtd Pedidos'].sum()
    transp_ant_row = group.sort_values(by='Qtd Pedidos', ascending=False).iloc[0]
    transp_ant, prazo_ant, qtd_ant, cmu_ant, ns_atual_ant = transp_ant_row['Transportador'], transp_ant_row['Prazo Prometido'], transp_ant_row['Qtd Pedidos'], transp_ant_row['CMU'], transp_ant_row['NS Atual']
    uf = transp_ant_row['UF']
    limite_cmu = CMU_MAX_UF.get(uf, float('inf'))
    cenarios_validos, cenarios_fallback = [], []

    for idx, row in group.iterrows():
        cmu, prazo_orig, transp = row['CMU'], row['Prazo Prometido'], row['Transportador']
        for c in CENARIOS:
            ns, ajuste = row[c['coluna']], c['ajuste']
            prazo_final = prazo_orig + ajuste
            if prazo_final < 0: continue
            
            if cmu <= limite_cmu:
                # Conecta com as variáveis do menu lateral
                if ns >= (meta_ns * 100) and prazo_final <= limite_prazo: 
                    cenarios_validos.append((prazo_final, ns, cmu, transp, row['NS Atual']))
                
                # O fallback também só deve aceitar cenários dentro do prazo limite estipulado
                if prazo_final <= limite_prazo:
                    cenarios_fallback.append((prazo_final, ns, cmu, transp, row['NS Atual']))

    if cenarios_validos:
        cenarios_validos.sort(key=lambda x: (x[0], -x[1]))
        best = cenarios_validos[0]
    elif cenarios_fallback:
        cenarios_fallback.sort(key=lambda x: (-x[1], x[0]))
        best = cenarios_fallback[0]
    else:
        best = (prazo_ant, ns_atual_ant, cmu_ant, transp_ant, ns_atual_ant)

    transp_sel, prazo_sel, cmu_sel, ns_proj, ns_atual_sel = best[3], best[0], best[2], best[1], best[4]
    diff_prazo = prazo_ant - prazo_sel
    acao = f"Redução média de {diff_prazo:.1f} dias" if diff_prazo > 0 else (f"Aumento médio de {-diff_prazo:.1f} dias" if diff_prazo < 0 else "Manter cenário original")

    return pd.Series({
        'Região': transp_ant_row['Região'], 'UF': uf, 'Cidade': transp_ant_row['Cidade'],
        'Transportadora Anterior': transp_ant, 'Prazo Transportadora Anterior': prazo_ant,
        'qtd pedidos transp anterior': qtd_ant, 'CMU transportadora atual': cmu_ant,
        'Transportadora Selecionada': transp_sel, 'Prazo Transportadora Selecionada': prazo_sel,
        'Qtd Pedidos Transportadora Selecionada': vol_total_cep, 'CMU transportadora selecionada': cmu_sel,
        'NS Atual': ns_atual_sel, 'NS Projetado': ns_proj, 'Ação Sugerida': acao
    })

def calcular_agregacao_executiva(group):
    # A correção acontece aqui: removemos a injeção manual das "colunas chave"
    peso_ant, peso_sel = group['qtd pedidos transp anterior'], group['Qtd Pedidos Transportadora Selecionada']
    vol_ant_total, vol_sel_total = peso_ant.sum(), peso_sel.sum()
    lider_ant = group.groupby('Transportadora Anterior')['qtd pedidos transp anterior'].sum().idxmax() if vol_ant_total > 0 else group['Transportadora Anterior'].iloc[0]
    lider_sel = group.groupby('Transportadora Selecionada')['Qtd Pedidos Transportadora Selecionada'].sum().idxmax() if vol_sel_total > 0 else group['Transportadora Selecionada'].iloc[0]

    def mp(col, pesos, total): return np.sum(group[col] * pesos) / total if total > 0 else group[col].mean()

    prazo_ant, cmu_ant = mp('Prazo Transportadora Anterior', peso_ant, vol_ant_total), mp('CMU transportadora atual', peso_ant, vol_ant_total)
    prazo_sel, cmu_sel = mp('Prazo Transportadora Selecionada', peso_sel, vol_sel_total), mp('CMU transportadora selecionada', peso_sel, vol_sel_total)
    ns_proj, ns_atual = mp('NS Projetado', peso_sel, vol_sel_total), mp('NS Atual', peso_sel, vol_sel_total)

    diff_prazo = prazo_ant - prazo_sel
    acao = f"Redução média de {diff_prazo:.1f} dias" if diff_prazo > 0 else (f"Aumento médio de {-diff_prazo:.1f} dias" if diff_prazo < 0 else "Manter cenário original")

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
    with st.spinner('Lendo e limpando os dados... Isso pode levar alguns segundos.'):
        
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', encoding='utf-8', low_memory=False)

        colunas_ns = [c['coluna'] for c in CENARIOS]
        for col in colunas_ns:
            if col in df.columns:
                df[col] = df[col].apply(limpar_numero)

        if 'CMU' in df.columns:
            df['CMU'] = df['CMU'].apply(limpar_numero)
            
        if 'Prazo Prometido' in df.columns:
            df['Prazo Prometido'] = df['Prazo Prometido'].apply(limpar_numero)

        # Usamos o reset_index() ao invés do as_index=False para garantir compatibilidade com as versões mais novas do Pandas
        df_cep = df.groupby('CEP').apply(avaliar_cenarios_cep).reset_index()
        df_regiao = df_cep.groupby('Região').apply(calcular_agregacao_executiva).reset_index()
        df_uf = df_cep.groupby('UF').apply(calcular_agregacao_executiva).reset_index()
        df_cidade = df_cep.groupby(['UF', 'Cidade']).apply(calcular_agregacao_executiva).reset_index()

        # Cálculos Globais para a Tela
        vol_total = df_cep['Qtd Pedidos Transportadora Selecionada'].sum()
        vol_ant_total = df_cep['qtd pedidos transp anterior'].sum()

        prazo_ant = np.sum(df_cep['Prazo Transportadora Anterior'] * df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        prazo_sel = np.sum(df_cep['Prazo Transportadora Selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total if vol_total > 0 else 0
        ns_atual = np.sum(df_cep['NS Atual'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total if vol_total > 0 else 0
        ns_proj = np.sum(df_cep['NS Projetado'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total if vol_total > 0 else 0
        cmu_ant = np.sum(df_cep['CMU transportadora atual'] * df_cep['qtd pedidos transp anterior']) / vol_ant_total if vol_ant_total > 0 else 0
        cmu_sel = np.sum(df_cep['CMU transportadora selecionada'] * df_cep['Qtd Pedidos Transportadora Selecionada']) / vol_total if vol_total > 0 else 0

    st.success('Análise concluída com sucesso!')
    st.subheader("Resultados Globais da Otimização")

    # Exibe os Cards de forma simples
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Prazo Prometido (Dias)", f"{prazo_sel:.1f}", f"{prazo_sel - prazo_ant:.1f} dias", delta_color="inverse")
    col2.metric("Nível de Serviço (NS)", f"{ns_proj:.1f}%", f"{ns_proj - ns_atual:.1f}%", delta_color="normal")
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
        label="📥 Baixar Planilha de Resultados (Excel)",
        data=buffer.getvalue(),
        file_name="Resultado_Otimizacao_Padronizado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
