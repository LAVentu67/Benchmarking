import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
import sqlalchemy
import joblib
import time
import unicodedata  # <--- MODIFICACIÓN 2: Importado para normalize_text

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Benchmarking Ventura", layout="wide", initial_sidebar_state="expanded")

# --- <--- MODIFICACIÓN 2: INICIO (Helper functions) ---
# Inicializar st.session_state
if "cliente_autenticado" not in st.session_state:
    st.session_state.cliente_autenticado = None

def normalize_text(s):
    """Normaliza el texto para comparación (quita acentos, minúsculas, espacios)."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    return s
# --- <--- MODIFICACIÓN 2: FIN (Helper functions) ---


# --- CSS PERSONALIZADO ---
st.markdown("""
<style>
@import url('https://fonts.com/css2?family=Roboto:wght@300;400;700&display=swap');
:root {
--primary-color: #5C1212;
--sidebar-bg-color: #212529;
--app-bg-color: #f0f2ff;
--white: #ffffff;
}
body { font-family: 'Roboto', sans-serif; background-color: var(--app-bg-color); }
.title-banner { background-color: var(--primary-color); color: var(--white); font-size: 2.2em; font-weight: bold; padding: 25px; text-align: center; margin-bottom: 30px; }
[data-testid="stSidebar"] { background-color: var(--sidebar-bg-color); border-right: 3px solid var(--primary-color); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: var(--white); }
[data-testid="stSidebar"] [data-testid="stImage"] { background-color: var(--white); padding: 10px; border-radius: 10px; margin-bottom: 20px; }
[data-testid="stSidebar"] .stSelectbox > div, [data-testid="stSidebar"] .stMultiSelect > div { background-color: #343a3a; color: var(--white); }
.stButton > button { background-color: var(--primary-color); color: white; border-radius: 6px; border: none; width: 100%; font-weight: 700; padding: 8px 0; }
.stButton > button:hover { background-color: #4a0f0f; }

/* Estilo para las tablas comparativas */
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- BANNER DEL TÍTULO ---
st.markdown('<div class="title-banner">Benchmarking Ventura</div>', unsafe_allow_html=True)

# --- CARGA DE RECURSOS Y CONEXIÓN A LA BASE DE DATOS (CACHEADA) ---
@st.cache_resource
def cargar_recursos():
    """Carga los filtros y establece la conexión a la base de datos."""
    try:
        opciones_filtros = joblib.load('opciones_filtros.joblib')
    except FileNotFoundError:
        st.error("Error: Archivo 'opciones_filtros.joblib' no encontrado. Asegúrate de tener los recursos necesarios.")
        opciones_filtros = {'clientes': [], 'clas_venta': [], 'condiciones': [], 'clas_modelo': [], 'origen_marca': [], 'combustibles': []}

    if "postgres" not in st.secrets or "db_url" not in st.secrets["postgres"]:
        st.error("Error: La configuración de la base de datos 'postgres' en st.secrets no es válida.")
        DATABASE_URL = "sqlite:///:memory:" 
    else:
        DATABASE_URL = st.secrets["postgres"]["db_url"].replace("postgresql://", "postgresql+psycopg2://")
    
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "sslmode": "require", "keepalives": 1,
            "keepalives_idle": 20, "keepalives_interval": 5,
        }
    )
    return engine, opciones_filtros

engine, opciones_filtros = cargar_recursos()

# --- FUNCIÓN OPTIMIZADA PARA OBTENER DATOS (CON REINTENTOS) ---
@st.cache_data(ttl="1h")
def obtener_datos_filtrados(_engine, query, params=None, retries=3, delay=5):
    """Ejecuta una consulta SQL y devuelve un DataFrame, con reintentos."""
    last_exception = None
    for attempt in range(retries):
        try:
            with _engine.connect() as conn:
                df = pd.read_sql_query(text(query), conn, params=params)
            
                if 'CLIENTE' in df.columns:
                    df.rename(columns={'CLIENTE': 'GRUPO'}, inplace=True)
                if 'FECHA_DE_PAGO' in df.columns:
                    df['FECHA_DE_PAGO'] = pd.to_datetime(df['FECHA_DE_PAGO'], errors='coerce')
                return df
        
        except sqlalchemy.exc.OperationalError as e:
            last_exception = e
            time.sleep(delay)
        
        except Exception as e:
            st.error(f"Error de SQL no recuperable: {e}")
            raise e
    
    st.error(f"Fallaron todos los reintentos para conectar a la base de datos. Último error: {last_exception}")
    raise last_exception


# --- INICIO DEFINICIÓN DE SIDEBAR ---

st.sidebar.image("https://tse1.mm.bing.net/th/id/OIP.dZs9yNpJVa2kZjoE9rx54gAAAA?cb=12&rs=1&pid=ImgDetMain&o=7&rm=3", use_container_width=True)

# <--- MODIFICACIÓN 1: Hipervínculo ---
st.sidebar.markdown('''
<div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
    <a href="https://pronostico-ventura-lagara7yjtmtcgbmjcoqyntbilee.streamlit.app/" target="_blank" style="text-decoration: none; font-weight: bold; color: black;">
        VENTAS
    </a>
</div>
''', unsafe_allow_html=True)
st.sidebar.divider()

if st.sidebar.button("Borrar Filtros", key="clear_filters"):
    # Reiniciar filtros a 'TODOS' sin borrar todo el estado
    st.session_state['grupo_sel'] = 'TODOS'
    st.session_state['clas_venta_sel'] = 'TODOS'
    st.session_state['condicion_sel'] = 'TODOS'
    st.session_state['clas_modelo_sel'] = 'TODOS'
    st.session_state['origen_marca_sel'] = 'TODOS'
    st.session_state['combustible_sel'] = 'TODOS'
    st.rerun()

# --- Filtro de Mes (Solo para Comparativo) ---
st.sidebar.header("Segmentadores")
# <--- MODIFICACIÓN: Meses en Español ---
meses_dict = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}
año_actual = pd.Timestamp.today().year
mes_actual_num = pd.Timestamp.today().month
mes_nombre = st.sidebar.selectbox(
    "Selecciona Mes (Solo funciona en Comparativo)", 
    list(meses_dict.values()), 
    index=mes_actual_num - 1 
)
mes_num = next((k for k, v in meses_dict.items() if v == mes_nombre), None)
st.sidebar.divider()

# --- Filtros Generales (para AMBAS pestañas) ---


# <--- MODIFICACIÓN 2: INICIO Lógica de Autenticación ---
grupo_sel = st.sidebar.selectbox("Cliente", ["TODOS"] + opciones_filtros['clientes'],   key="grupo_sel")

acceso_permitido = False

if grupo_sel == "TODOS":
    st.session_state.cliente_autenticado = None
    acceso_permitido = True
else:
    if st.session_state.get("cliente_autenticado") == grupo_sel:
        st.sidebar.success(f"Acceso verificado para: {grupo_sel}")
        acceso_permitido = True
    else:
        st.sidebar.warning(f"Se requiere un código para acceder a los datos de '{grupo_sel}'.")
        client_code = st.sidebar.text_input("Código de acceso:", type="password", key=f"password_{grupo_sel}")

        if st.sidebar.button("Verificar Código", key=f"verify_{grupo_sel}"):
            try:
                # Cargar los códigos de acceso desde secrets.toml
                client_codes_raw = st.secrets.get("client_codes", {})
                client_codes_normalized = {normalize_text(k): v for k, v in client_codes_raw.items()}
                grupo_sel_normalized = normalize_text(grupo_sel)

                if client_codes_normalized.get(grupo_sel_normalized) == client_code:
                    st.session_state.cliente_autenticado = grupo_sel
                    st.rerun()
                else:
                    st.sidebar.error("Código incorrecto. Acceso denegado.")
                    st.session_state.cliente_autenticado = None
            
            except Exception as e:
                st.sidebar.error(f"Error al leer los códigos de acceso: {e}")
                st.session_state.cliente_autenticado = None
# --- <--- MODIFICACIÓN 2: FIN Lógica de Autenticación ---


clas_venta_sel = st.sidebar.selectbox("Clasificación de Venta", ["TODOS"] + opciones_filtros['clas_venta'],key="clas_venta_sel")
condicion_sel = st.sidebar.selectbox("Condición de Venta", ["TODOS"] + opciones_filtros['condiciones'],    key="condicion_sel")
clas_modelo_sel = st.sidebar.selectbox("Clasificación de Modelo", ["TODOS"] + opciones_filtros['clas_modelo'],  key="clas_modelo_sel")
origen_marca_sel = st.sidebar.selectbox("Marca China", ["TODOS"] + opciones_filtros['origen_marca'],  key="origen_marca_sel")
combustible_sel = st.sidebar.selectbox("Combustible", ["TODOS"] + opciones_filtros['combustibles'], key="combustible_sel")
st.sidebar.divider()


# <--- MODIFICACIÓN 2: Botón deshabilitado si no hay acceso ---
run_evolucion = st.sidebar.button("Analizar Evolución", disabled=(not acceso_permitido))

# --- FIN DEFINICIÓN DE SIDEBAR ---


# --- DEFINICIÓN DE PESTAÑAS ---
tabs = st.tabs(["Evolución", "Comparativo"])

# --- PESTAÑA DE EVOLUCIÓN ---
with tabs[0]:
    
    if run_evolucion: 
        if not acceso_permitido:
             st.error("Acceso denegado. Verifica el código en la barra lateral.")
        else:
            query_base = 'SELECT "CLIENTE", "SEGMENTO", "FECHA_DE_PAGO", "CANTIDAD_OFERTADA", "PRECIO_RESERVA", "COSTO_CLIENTE", "PRECIO_DE_MERCADO", "DIAS_HABILES_VENTA", "NUMERO_DE_OFERTAS", "RECUPERACION_PRECIO", "RECUPERACION_VALOR", "MARCA", "MODELO", "CLASIFICACION_VENTA", "CONDICION_DE_VENTA", "CLASIFICACION_MODELO", "ORIGEN_MARCA", "COMBUSTIBLE" FROM ventas_historicas WHERE 1=1'
            params = {}
            
            filtros = {
                'CLIENTE': grupo_sel, 
                'CLASIFICACION_VENTA': clas_venta_sel, 
                'CONDICION_DE_VENTA': condicion_sel,
                'CLASIFICACION_MODELO': clas_modelo_sel, 
                'ORIGEN_MARCA': origen_marca_sel, 
                'COMBUSTIBLE': combustible_sel
            }

            for col, val in filtros.items():
                if val != "TODOS":
                    param_name = f"p_{col.lower()}"
                    query_base += f' AND "{col}" = :{param_name}'
                    params[param_name] = val
            
            df_grupo = obtener_datos_filtrados(engine, query_base, params)
            df_grupo['MES_AÑO_PAGO'] = df_grupo['FECHA_DE_PAGO'].dt.to_period('M').dt.to_timestamp()

            if df_grupo.empty:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")
            else:
                segmento_actual = df_grupo['SEGMENTO'].mode().get(0, None)
                
                query_segmento = 'SELECT "CLIENTE", "SEGMENTO", "FECHA_DE_PAGO", "CANTIDAD_OFERTADA", "PRECIO_RESERVA", "COSTO_CLIENTE", "PRECIO_DE_MERCADO", "DIAS_HABILES_VENTA", "NUMERO_DE_OFERTAS", "RECUPERACION_PRECIO", "RECUPERACION_VALOR", "MARCA", "MODELO" FROM ventas_historicas WHERE "SEGMENTO" = :segmento'
                df_segmento_completo = obtener_datos_filtrados(engine, query_segmento, params={'segmento': segmento_actual})
                df_segmento_completo['MES_AÑO_PAGO'] = df_segmento_completo['FECHA_DE_PAGO'].dt.to_period('M').dt.to_timestamp()
                df_segmento_sin_grupo = df_segmento_completo[df_segmento_completo['GRUPO'] != grupo_sel]

                cols_to_analyze = {
                    'UNIDADES': 'Unidades Vendidas', 
                    'CANTIDAD_OFERTADA': 'Precio de Venta Promedio', 
                    'PRECIO_RESERVA': 'Precio de Reserva Promedio', 
                    'COSTO_CLIENTE': 'Costo Cliente Promedio', 
                    'PRECIO_DE_MERCADO': 'Valor de Mercado Promedio', 
                    'DIAS_HABILES_VENTA': 'Días de Venta Promedio', 
                    'NUMERO_DE_OFERTAS': 'Número de Ofertas Promedio', 
                    'RECUPERACION_PRECIO': 'Recuperación de Venta Promedio', 
                    'RECUPERACION_VALOR': 'Recuperación de Mercado Promedio'
                }
                
                actual_cols_to_analyze = list(cols_to_analyze.keys())
                if 'UNIDADES' in actual_cols_to_analyze:
                    actual_cols_to_analyze.remove('UNIDADES')
                    actual_cols_to_analyze.insert(0, 'UNIDADES')
                
                for col in actual_cols_to_analyze:
                    title = cols_to_analyze[col]
                    
                    if col == 'UNIDADES':
                        df_g = df_grupo.groupby('MES_AÑO_PAGO').size().reset_index(name='UNIDADES_Grupo')
                        df_s = df_segmento_sin_grupo.groupby('MES_AÑO_PAGO').size().reset_index(name='UNIDADES_Segmento')
                        df_merge = df_g.merge(df_s, on='MES_AÑO_PAGO', suffixes=('_Grupo', '_Segmento'), how='outer').sort_values('MES_AÑO_PAGO').fillna(0)
                        col_name_g = 'UNIDADES_Grupo'
                        col_name_s = 'UNIDADES_Segmento'
                        hover_format = ":,.0f"
                        text_format = ".0f"
                        y_title_text = "Conteo"
                        
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge[col_name_s], 
                            mode='lines', name=f'Segmento: {segmento_actual}', 
                            line=dict(color='#C87E7E', width=0),
                            fill='tozeroy', fillcolor='rgba(200, 126, 126, 0.4)',
                            stackgroup='one',
                            hovertemplate=f"Mes: %{{x|%Y-%m}}<br><b>Segmento:</b> %{{y{hover_format}}}<extra></extra>",
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge[col_name_g], 
                            mode='lines', name=f'Cliente: {grupo_sel}', 
                            line=dict(color='#5C1212', width=0),
                            fill='tonexty', fillcolor='rgba(92, 18, 18, 0.6)',
                            stackgroup='one',
                            hovertemplate=f"Mes: %{{x|%Y-%m}}<br><b>Cliente:</b> %{{y{hover_format}}}<extra></extra>",
                        ))
                        
                        df_merge['Total'] = df_merge[col_name_g] + df_merge[col_name_s]
                        
                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge[col_name_s], 
                            mode='text',
                            text=[f'{val:{text_format}}' for val in df_merge[col_name_s]], 
                            textfont=dict(size=11, color='#333333', 
                                          family='Roboto, sans-serif'),
                            textposition="bottom center", 
                            showlegend=False,
                            hovertemplate=None
                        ))

                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge['Total'],
                            mode='text',
                            text=[f'{val:{text_format}}' for val in df_merge['Total']], 
                            textfont=dict(size=12, color='black',
                                          family='Roboto, sans-serif'), 
                            textposition="top center",
                            showlegend=False
                        ))
                        
                    else:
                        df_g = df_grupo.groupby('MES_AÑO_PAGO')[col].mean().reset_index()
                        df_s = df_segmento_sin_grupo.groupby('MES_AÑO_PAGO')[col].mean().reset_index()
                        df_merge = df_g.merge(df_s, on='MES_AÑO_PAGO', suffixes=('_Grupo', '_Segmento'), how='outer').sort_values('MES_AÑO_PAGO').fillna(0)
                        col_name_g = f'{col}_Grupo'
                        col_name_s = f'{col}_Segmento'
                        
                        if col in ['DIAS_HABILES_VENTA', 'NUMERO_DE_OFERTAS']:
                            hover_format = ":,.0f"
                            text_spec = ".0f" 
                            text_prefix = ""
                            text_suffix = ""
                        elif col in ['RECUPERACION_PRECIO', 'RECUPERACION_VALOR']:
                            hover_format = ":,.1f" 
                            text_spec = ".1f" 
                            text_prefix = ""
                            text_suffix = "%"   
                        else: 
                            hover_format = ":$,.0f"
                            text_spec = ",.0f"     
    
                            text_prefix = "$ "    
                            text_suffix = ""
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge[col_name_g], 
                            mode='lines+markers+text', name=f'Cliente: {grupo_sel}', 
                            line=dict(color='#5C1212', width=3),
                            text=[f'{text_prefix}{val:{text_spec}}{text_suffix}' for val in df_merge[col_name_g]], 
                            textfont=dict(size=12, color='black',
                                          family='Roboto, sans-serif'),
                            textposition="top center",
                            hovertemplate=f"Mes: %{{x|%Y-%m}}<br><b>Cliente:</b> %{{y{hover_format}}}{text_suffix}<extra></extra>",
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df_merge['MES_AÑO_PAGO'], y=df_merge[col_name_s], 
                            mode='lines+markers+text', name=f'Segmento: {segmento_actual}', 
                            line=dict(color='#C87E7E', dash='dash', width=3),
                            text=[f'{text_prefix}{val:{text_spec}}{text_suffix}' for val in df_merge[col_name_s]], 
                            textfont=dict(size=12, color='black', 
                                          family='Roboto, sans-serif'), 
                            textposition="bottom center",
                            hovertemplate=f"Mes: %{{x|%Y-%m}}<br><b>Segmento:</b> %{{y{hover_format}}}{text_suffix}<extra></extra>",
                        ))

                    # <--- MODIFICACIÓN 1: Quitar (Conteo/Promedio) del título del eje Y ---
                    fig.update_layout(
                        title=f"Evolución de {title} por Fecha de Pago", 
                        xaxis_title="Mes de Pago", 
                        xaxis=dict(tickformat="%Y-%m"),
                        yaxis=dict(
                            title=dict(
                                text=f"{title}",  # <--- MODIFICADO
                                font=dict(color="black")
                            ),
                            tickfont=dict(color="black")
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona un Grupo y presiona 'Analizar Evolución'.")


# --- PESTAÑA COMPARATIVO ---
with tabs[1]:
    st.markdown('<div style="text-align:center; padding-bottom:10px;"><h2 style="color:#5C1212;">Comparativo Cliente vs Segmento</h2></div>', unsafe_allow_html=True)

    # <--- MODIFICACIÓN 2: Lógica de acceso para la pestaña ---
    if not acceso_permitido:
        st.warning("Selecciona un Grupo y verifica tu código de acceso en la barra lateral para ver los datos.")
    
    elif mes_num and grupo_sel == "TODOS":
        st.info("Selecciona un Grupo en la barra lateral para iniciar el comparativo.")

    elif mes_num and grupo_sel != "TODOS":
        if mes_num == 1:
            mes_anterior_num, año_para_mes_anterior = 12, año_actual - 1
        else:
            mes_anterior_num, año_para_mes_anterior = mes_num - 1, año_actual

        query_comp = """
        SELECT "CLIENTE", "SEGMENTO", "CANTIDAD_OFERTADA", "PRECIO_RESERVA", "COSTO_CLIENTE", "PRECIO_DE_MERCADO", "FECHA_DE_PAGO", "MARCA", "MODELO", "DIAS_HABILES_VENTA", "NUMERO_DE_OFERTAS"
        FROM ventas_historicas
        WHERE (
            (EXTRACT(MONTH FROM "FECHA_DE_PAGO") = :mes_actual AND EXTRACT(YEAR FROM "FECHA_DE_PAGO") = :ano_actual)
            OR (EXTRACT(MONTH FROM "FECHA_DE_PAGO") = :mes_actual AND EXTRACT(YEAR FROM "FECHA_DE_PAGO") = :ano_anterior)
            OR (EXTRACT(MONTH FROM "FECHA_DE_PAGO") = :mes_anterior AND EXTRACT(YEAR FROM "FECHA_DE_PAGO") = :ano_para_mes_anterior)
        ) AND 1=1
        """
        params_comp = {
            'mes_actual': mes_num, 'ano_actual': año_actual, 'ano_anterior': año_actual - 1,
            'mes_anterior': mes_anterior_num, 'ano_para_mes_anterior': año_para_mes_anterior
        }
        
        filtros_comp = {
    'CLASIFICACION_VENTA': clas_venta_sel,
    'CONDICION_DE_VENTA': condicion_sel,
    'CLASIFICACION_MODELO': clas_modelo_sel,
    'ORIGEN_MARCA': origen_marca_sel,
    'COMBUSTIBLE': combustible_sel
}# <--- MODIFICACIÓN 2: El filtro de grupo también se aplica aquí
        

        for col, val in filtros_comp.items():
            if val != "TODOS":
                param_name = f"p_comp_{col.lower()}"
                query_comp += f' AND "{col}" = :{param_name}'
                params_comp[param_name] = val
        
        df_comp = obtener_datos_filtrados(engine, query_comp, params_comp)
        
        df_comp['AÑO'] = df_comp['FECHA_DE_PAGO'].dt.year
        df_comp['MES'] = df_comp['FECHA_DE_PAGO'].dt.month

        df_grupo_actual = df_comp[(df_comp['GRUPO'] == grupo_sel) & (df_comp['AÑO'] == año_actual) & (df_comp['MES'] == mes_num)]
        
        if df_grupo_actual.empty:
            st.warning(f"El grupo '{grupo_sel}' no tiene datos para {mes_nombre} de {año_actual} con los filtros seleccionados.")
        else:
            df_grupo_anterior_anual = df_comp[(df_comp['GRUPO'] == grupo_sel) & (df_comp['AÑO'] == año_actual - 1) & (df_comp['MES'] == mes_num)]
            df_grupo_anterior_mensual = df_comp[(df_comp['GRUPO'] == grupo_sel) & (df_comp['AÑO'] == año_para_mes_anterior) & (df_comp['MES'] == mes_anterior_num)]
            
            segmento = df_grupo_actual['SEGMENTO'].mode().iloc[0]
            df_segmento_actual = df_comp[(df_comp['SEGMENTO'] == segmento) & (df_comp['GRUPO'] != grupo_sel) & (df_comp['AÑO'] == año_actual) & (df_comp['MES'] == mes_num)]
            df_segmento_anterior_anual = df_comp[(df_comp['SEGMENTO'] == segmento) & (df_comp['GRUPO'] != grupo_sel) & (df_comp['AÑO'] == año_actual - 1) & (df_comp['MES'] == mes_num)]
            df_segmento_anterior_mensual = df_comp[(df_comp['SEGMENTO'] == segmento) & (df_comp['GRUPO'] != grupo_sel) & (df_comp['AÑO'] == año_para_mes_anterior) & (df_comp['MES'] == mes_anterior_num)]

            # --- FUNCIONES DE CÁLCULO Y GRÁFICOS (REUTILIZABLES) ---
            
            def calcular_metricas(df_actual, df_anterior, col_actual, col_anterior):
                def prom(df, col): return df[col].mean() if not df.empty and col in df.columns else 0
                def suma(df, col): return df[col].sum() if not df.empty and col in df.columns else 0
                def delta(a, b): return ((a - b) / b * 100) if b != 0 else (100.0 if a != 0 else 0.0)
                def div(a, b): return (a / b * 100) if b != 0 else 0

                PMV_act, PMV_ant = prom(df_actual, 'CANTIDAD_OFERTADA'), prom(df_anterior, 'CANTIDAD_OFERTADA')
                CMI_act, CMI_ant = prom(df_actual, 'COSTO_CLIENTE'), prom(df_anterior, 'COSTO_CLIENTE')
                PMM_act, PMM_ant = prom(df_actual, 'PRECIO_DE_MERCADO'), prom(df_anterior, 'PRECIO_DE_MERCADO')
                
                PRP_act, PRP_ant = prom(df_actual, 'PRECIO_RESERVA'), prom(df_anterior, 'PRECIO_RESERVA')
                DVP_act, DVP_ant = prom(df_actual, 'DIAS_HABILES_VENTA'), prom(df_anterior, 'DIAS_HABILES_VENTA')
                NOP_act, NOP_ant = prom(df_actual, 'NUMERO_DE_OFERTAS'), prom(df_anterior, 'NUMERO_DE_OFERTAS')

                return pd.DataFrame([
                    ["Unidades", len(df_anterior), len(df_actual), delta(len(df_actual), len(df_anterior))],
                    ["PMV", PMV_ant, PMV_act, delta(PMV_act, PMV_ant)],
                    ["CMI", CMI_ant, CMI_act, delta(CMI_act, CMI_ant)],
                    ["PMM", PMM_ant, PMM_act, delta(PMM_act, PMM_ant)],
                    ["Precio Reserva Promedio", PRP_ant, PRP_act, delta(PRP_act, PRP_ant)],
                    ["Días Venta Promedio", DVP_ant, DVP_act, delta(DVP_act, DVP_ant)], 
                    ["Ofertas Promedio", NOP_ant, NOP_act, delta(NOP_act, NOP_ant)], 
                    ["%Rec CMI", div(PMV_ant, CMI_ant), div(PMV_act, CMI_act), delta(div(PMV_act, CMI_act), div(PMV_ant, CMI_ant))],
                    ["%Rec EBC", div(PMV_ant, PMM_ant), div(PMV_act, PMM_act), delta(div(PMV_act, PMM_act), div(PMV_ant, PMM_ant))],
                    ["%Rec CC/EBC", div(CMI_ant, PMM_ant), div(CMI_act, PMM_act), delta(div(CMI_act, PMM_act), div(CMI_ant, PMM_ant))]
                ], columns=["Indicador", col_anterior, col_actual, "Δ%"])

            def generar_tabla_html(df, titulo, color_header, col_anterior, col_actual):
                html = f'<h4 style="text-align:center; color:{color_header};">{titulo}</h4>'
                html += '<table style="width:100%; border-collapse: collapse; text-align: center;">'
                html += f'<tr><th style="background-color:{color_header};color:white;">Indicador</th><th style="background-color:{color_header};color:white;">{col_anterior}</th><th style="background-color:{color_header};color:white;">{col_actual}</th><th style="background-color:{color_header};color:white;">Δ%</th></tr>'
                for _, row in df.iterrows():
                    indicador, val_ant, val_act, delta_val = row['Indicador'], row[col_anterior], row[col_actual], row['Δ%']
                    flecha = ''  # Flechas eliminadas para Cliente vs Segmento (2025)
                    
                    if indicador == "Unidades": f_ant, f_act = f"{val_ant:,.0f}", f"{val_act:,.0f}"
                    elif "%Rec" in indicador: f_ant, f_act = f"{val_ant:.1f}%", f"{val_act:.1f}%"
                    # <--- MODIFICACIÓN 2 (TABLAS): Quitar decimales ---
                    elif "Días Venta" in indicador or "Ofertas Promedio" in indicador: f_ant, f_act = f"{val_ant:,.0f}", f"{val_act:,.0f}" 
                    else: f_ant, f_act = f"$ {val_ant:,.0f}", f"$ {val_act:,.0f}"
                    
                    html += f"<tr><td>{indicador}</td><td>{f_ant}</td><td>{f_act}</td><td><b>{flecha} {delta_val:.1f}%</b></td></tr>"
                html += '</table>'
                return html

            def generar_tabla_delta(df, titulo, color_header="#6c757d"):
                """Genera una tabla HTML para comparar Cliente vs Segmento."""
                html = f'<h4 style="text-align:center; color:{color_header};">{titulo}</h4>'
                html += '<table style="width:100%; border-collapse: collapse; text-align: center;">'
                html += f'<tr><th style="background-color:{color_header};color:white;">Indicador</th>'
                html += f'<th style="background-color:{color_header};color:white;">Valor Grupo</th>'
                html += f'<th style="background-color:{color_header};color:white;">Valor Segmento</th>'
                html += f'<th style="background-color:{color_header};color:white;">∝ %</th></tr>'
                
                for _, row in df.iterrows():
                    indicador, val_gpo, val_seg, delta_val = row['Indicador'], row['Valor_Grupo'], row['Valor_Segmento'], row['Delta %']
                    flecha = ''  # Flechas eliminadas para Cliente vs Segmento (2025)
                    
                    if indicador == "Unidades":
                        f_gpo, f_seg = f"{val_gpo:,.0f}", f"{val_seg:,.0f}"
                    elif "%Rec" in indicador:
                        f_gpo, f_seg = f"{val_gpo:.1f}%", f"{val_seg:.1f}%"
                    # <--- MODIFICACIÓN 2 (TABLAS): Quitar decimales ---
                    elif "Días Venta" in indicador or "Ofertas Promedio" in indicador:
                        f_gpo, f_seg = f"{val_gpo:,.0f}", f"{val_seg:,.0f}" 
                    else: # PMV, CMI, PMM, etc.
                        f_gpo, f_seg = f"$ {val_gpo:,.0f}", f"$ {val_seg:,.0f}"
                        
                    html += f"<tr><td>{indicador}</td><td>{f_gpo}</td><td>{f_seg}</td><td><b>{flecha} {delta_val:.1f}%</b></td></tr>"
                
                html += '</table>'
                return html
            
            def mostrar_seccion_graficos(df_actual_grupo, df_actual_segmento, grupo_sel, segmento):
                
                col_actual, col_anterior = 'Actual', 'Anterior' 
                df_grupo_metrics = calcular_metricas(df_actual_grupo, df_actual_grupo, col_actual, col_anterior) 
                df_segmento_metrics = calcular_metricas(df_actual_segmento, df_actual_segmento, col_actual, col_anterior)
                
                # --- MODIFICACIÓN 1: Gráficos de pastel eliminados de aquí ---

                # --- Generar Barras (3 Columnas) ---
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<h6>Métricas Promedio</h6>", unsafe_allow_html=True)
                bar_cols = st.columns(3) # 3 Columnas para barras
                bar_i = 0

                metricas_promedio = [
                    'PMV', 'CMI', 'PMM', 'Precio Reserva Promedio', 
                    'Días Venta Promedio', 'Ofertas Promedio', 
                    '%Rec CMI', '%Rec EBC', '%Rec CC/EBC'
                ]
                df_grupo_prom = df_grupo_metrics[df_grupo_metrics['Indicador'].isin(metricas_promedio)].set_index('Indicador')[col_actual]
                df_segmento_prom = df_segmento_metrics[df_segmento_metrics['Indicador'].isin(metricas_promedio)].set_index('Indicador')[col_actual]
                
                df_bar_data = pd.DataFrame({
                    'Grupo': df_grupo_prom,
                    'Segmento': df_segmento_prom
                }).reset_index()

                bar_metrics = [m for m in df_grupo_metrics['Indicador'].unique() if m in metricas_promedio]

                for metric in bar_metrics:
                    data_bar = df_bar_data[df_bar_data['Indicador'] == metric]
                    
                    if data_bar.empty:
                        continue 

                    # <--- MODIFICACIÓN 3: INICIO (Formatos de Barras) ---
                    if metric == "Unidades": 
                        format_spec = ",.0f"
                        format_prefix = ""
                        format_suffix = "" # Sufijo para la etiqueta
                        title_suffix = ""
                    elif "%Rec" in metric:
                        format_spec = ".1f"  # 1 decimal
                        format_prefix = ""
                        format_suffix = "%"  # Sufijo de %
                        title_suffix = ""    # Título limpio
                    elif "Días Venta" in metric or "Ofertas Promedio" in metric:
                        format_spec = ".0f"  # Sin decimales
                        format_prefix = ""
                        format_suffix = ""
                        title_suffix = ""
                    else: # PMV, CMI, PMM, PRP
                        format_spec = ",.0f"  # Sin decimales
                        format_prefix = "$ "     
                        format_suffix = ""
                        title_suffix = ""
                    # <--- MODIFICACIÓN 3: FIN (Formatos de Barras) ---

                    fig = go.Figure(data=[
                        go.Bar(name=grupo_sel, 
                            x=['Grupo'], y=[data_bar['Grupo'].iloc[0]], 
                            marker_color='#5C1212', 
                            # <--- MODIFICACIÓN 3: Aplicar formato completo ---
                            text=f'{format_prefix}{data_bar["Cliente"].iloc[0]:{format_spec}}{format_suffix}',
                            textposition='outside'),
                        go.Bar(name=f'Segmento ({segmento})', 
                            x=['Segmento'], y=[data_bar['Segmento'].iloc[0]], 
                            marker_color='#C87E7E', 
                            # <--- MODIFICACIÓN 3: Aplicar formato completo ---
                            text=f'{format_prefix}{data_bar["Segmento"].iloc[0]:{format_spec}}{format_suffix}',
                            textposition='outside')
                    ])

                    fig.update_traces(textfont=dict(size=14, color='black', weight='bold'), showlegend=True) 
                    fig.update_layout(
                        barmode='group', 
                        # <--- MODIFICACIÓN 3: Título limpio ---
                        title=f'{metric}{title_suffix}',
                        yaxis_title='Valor',
                        showlegend=False,
                        margin=dict(t=50, l=10, r=10, b=10) 
                    )
                    bar_cols[bar_i % 3].plotly_chart(fig, use_container_width=True)
                    bar_i += 1

            # ---- SECCIÓN 1: COMPARATIVO ANUAL ----
            st.subheader("Comparativo Anual")
            
            with st.container(border=True):
                col_actual_anual, col_anterior_anual = str(año_actual), str(año_actual - 1)
                tabla_grupo_anual = calcular_metricas(df_grupo_actual, df_grupo_anterior_anual, col_actual_anual, col_anterior_anual)
                tabla_segmento_anual = calcular_metricas(df_segmento_actual, df_segmento_anterior_anual, col_actual_anual, col_anterior_anual)
                
                t1, t2 = st.columns(2)
                t1.markdown(generar_tabla_html(tabla_grupo_anual, f"Cliente: {grupo_sel}", "#5C1212", col_anterior_anual, col_actual_anual), unsafe_allow_html=True)
                t2.markdown(generar_tabla_html(tabla_segmento_anual, f"Segmento (Sin Grupo): {segmento}", "#C87E7E", col_anterior_anual, col_actual_anual), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # <--- MODIFICACIÓN 1: INICIO (Colocar Delta y Pastel lado a lado) ---
                t_delta, t_pie = st.columns(2)
                
                with t_delta:
                    df_delta_anual = pd.DataFrame()
                    df_delta_anual['Indicador'] = tabla_grupo_anual['Indicador']
                    df_delta_anual['Valor_Grupo'] = tabla_grupo_anual[col_actual_anual]
                    df_delta_anual['Valor_Segmento'] = tabla_segmento_anual[col_actual_anual]
                    df_delta_anual['Delta %'] = df_delta_anual.apply(lambda row: (row['Valor_Grupo'] / row['Valor_Segmento'] * 100) if row['Valor_Segmento'] != 0 else 0, axis=1)
                    st.markdown(generar_tabla_delta(df_delta_anual, f"Cliente vs Segmento ({año_actual})"), unsafe_allow_html=True)

                with t_pie:
                    # --- Lógica del pastel "Unidades" movida aquí ---
                    metric = "Unidades"
                    col_actual = col_actual_anual 
                    
                    val_grupo = tabla_grupo_anual[tabla_grupo_anual['Indicador'] == metric][col_actual].iloc[0]
                    val_segmento = tabla_segmento_anual[tabla_segmento_anual['Indicador'] == metric][col_actual].iloc[0]
                    
                    df_pie = pd.DataFrame({
                        'Entidad': [grupo_sel, f'Segmento ({segmento})'],
                        'Valor': [val_grupo, val_segmento]
                    })

                    format_label_hover = ":,.0f"

                    fig = px.pie(
                        df_pie, 
                        names='Entidad', 
                        values='Valor', 
                        # <--- MODIFICACIÓN 4: Título de Pastel sin () ---
                        title=f'{metric} Absoluto - {año_actual}',
                        color_discrete_sequence=['#5C1212', '#C87E7E']
                    )
                    
                    # <--- MODIFICACIÓN 4: Porcentaje con 1 decimal ---
                    fig.update_traces(
                        texttemplate=f"%{{label}}<br>%{{percent:.1%}}<br>%{{value:{format_label_hover}}}",
                        textposition='inside',
                        hovertemplate=f"<b>%{{label}}</b><br>{metric}: %{{value{format_label_hover}}}<br>Porcentaje: %{{percent:.1%}}<extra></extra>",
                        textfont=dict(size=14, color='white', weight='bold')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                # --- <--- MODIFICACIÓN 1: FIN ---

            st.markdown("#### Comparación de Métricas (Mes Actual)")
            mostrar_seccion_graficos(df_grupo_actual, df_segmento_actual, grupo_sel, segmento)

            # ---- SECCIÓN 2: COMPARATIVO MENSUAL ----
            st.subheader("Comparativo Mensual")
            with st.container(border=True):
                col_actual_mensual = f"{mes_nombre} {año_actual}"
                col_anterior_mensual = f"{meses_dict[mes_anterior_num]} {año_para_mes_anterior}"
                tabla_grupo_mensual = calcular_metricas(df_grupo_actual, df_grupo_anterior_mensual, col_actual_mensual, col_anterior_mensual)
                tabla_segmento_mensual = calcular_metricas(df_segmento_actual, df_segmento_anterior_mensual, col_actual_mensual, col_anterior_mensual)

                t3, t4 = st.columns(2)
                t3.markdown(generar_tabla_html(tabla_grupo_mensual, f"Cliente: {grupo_sel}", "#5C1212", col_anterior_mensual, col_actual_mensual), unsafe_allow_html=True)
                t4.markdown(generar_tabla_html(tabla_segmento_mensual, f"Segmento (Sin Grupo): {segmento}", "#C87E7E", col_anterior_mensual, col_actual_mensual), unsafe_allow_html=True)
                
                # <--- MODIFICACIÓN 1: Tabla Delta Mensual eliminada ---

            # ---- SECCIÓN 3: UNIVERSO ESTADÍSTICO ----
            st.subheader(f"Universo de Unidades")
            with st.container(border=True):
                
                # --- Gráficos de Dispersión ---
                st.markdown("#### Gráficos de Dispersión")
                
                def crear_grafico_dispersion(df_grupo, df_segmento, x_col, y_col, x_label, y_label, key):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_segmento[x_col], y=df_segmento[y_col],
                        mode='markers', name=f'Segmento: {segmento}',
                        marker=dict(color='#C87E7E', opacity=0.6),
                        hovertemplate=f"<b>Segmento</b><br>{x_label}: %{{x:$,.0f}}<br>{y_label}: %{{y:$,.0f}}<extra></extra>"
                    ))
                    fig.add_trace(go.Scatter(
                        x=df_grupo[x_col], y=df_grupo[y_col],
                        mode='markers', name=f'Cliente: {grupo_sel}', 
                        marker=dict(color='#5C1212', opacity=0.8, line=dict(width=1, color='Black')),
                        hovertemplate=f"<b>Grupo</b><br>{x_label}: %{{x:$,.0f}}<br>{y_label}: %{{y:$,.0f}}<extra></extra>"
                    ))
                    fig.update_layout(
                        title=f"<b>{y_label} vs. {x_label}</b>",
                        xaxis_title=x_label, yaxis_title=y_label,
                        legend_title="Comparativo",
                        font=dict(family="Roboto, sans-serif")
                    )
                    st.plotly_chart(fig, use_container_width=True, key=key)

                disp_col1, disp_col2 = st.columns(2)
                with disp_col1:
                    crear_grafico_dispersion(df_grupo_actual, df_segmento_actual, 
                                            'COSTO_CLIENTE', 'CANTIDAD_OFERTADA', 
                                            'Costo Cliente', 'Precio de Venta', 'scatter_costo')
                with disp_col2:
                    crear_grafico_dispersion(df_grupo_actual, df_segmento_actual, 
                                            'PRECIO_DE_MERCADO', 'CANTIDAD_OFERTADA', 
                                            'Valor de Mercado', 'Precio de Venta', 'scatter_mercado')

                st.markdown("---")
                st.markdown("####  Marcas y Modelos")

                # 1. Gráficos de Jerarquía (Treemap)
                def crear_grafico_jerarquia(df, titulo, key, color_scale='Reds', base_color='#5C1212'):
                    if df.empty or 'MARCA' not in df.columns or 'MODELO' not in df.columns:
                        st.warning(f"No hay datos de Marca/Modelo para el gráfico: {titulo}")
                        return

                    df_cleaned = df.copy() 
                    df_cleaned['MARCA'] = df_cleaned['MARCA'].fillna('SIN MARCA').astype(str).replace("", "SIN MARCA")
                    df_cleaned['MODELO'] = df_cleaned['MODELO'].fillna('SIN MODELO').astype(str).replace("", "SIN MODELO")

                    df_agg = df_cleaned.groupby(['MARCA', 'MODELO']).size().reset_index(name='Unidades')
                    
                    if df_agg.empty:
                        st.warning(f"No hay datos agregados para el gráfico: {titulo}")
                        return
                    
                    marca_totals = df_agg.groupby('MARCA')['Unidades'].transform('sum')
                    df_agg['Pct_Marca'] = (df_agg['Unidades'] / marca_totals)
                        
                    df_agg['Total'] = 'Total'

                    if key == 'tree_grupo':
                        colors = ['#5C1212', '#7A1818', '#991E1E', '#B82424', '#D72A2A'] 
                    else: # Segmento
                        colors = ['#C87E7E', '#D19292', '#DAB6B6', '#E3CACB', '#ECECEC'] 


                    fig = px.treemap(
                        df_agg, 
                        path=['Total', 'MARCA', 'MODELO'], 
                        values='Unidades',
                        color='Pct_Marca',
                        color_continuous_scale=colors, 
                        color_continuous_midpoint=np.average(df_agg['Pct_Marca'], weights=df_agg['Unidades']),
                        hover_data={'Unidades': ':,', 'MARCA': True, 'MODELO': True}
                    )
                    
                    fig.update(layout_coloraxis_showscale=False)
                    
                    fig.update_traces(
                        textinfo="label+value+percent parent", 
                        hovertemplate="<b>Marca:</b> %{parent}<br><b>Modelo:</b> %{label}<br><b>Unidades:</b> %{value:,}<br><b>% de la Marca:</b> %{percentParent:.1%}<br><b>% del Total:</b> %{percentRoot:.1%}<extra></extra>"
                    )

                    fig.update_layout(
                        title_text=f"<b>{titulo}</b>", 
                        font=dict(family="Roboto, sans-serif"),
                        margin=dict(t=50, l=0, r=0, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True, key=key)

                tree_col1, tree_col2 = st.columns(2)
                with tree_col1:
                    crear_grafico_jerarquia(df_grupo_actual, f"Cliente: {grupo_sel}", 'tree_grupo')
                with tree_col2:
                    crear_grafico_jerarquia(df_segmento_actual, f"Segmento: {segmento}", 'tree_segmento')

    else:
        st.info("Selecciona un Mes y un Grupo en la barra lateral para iniciar.")