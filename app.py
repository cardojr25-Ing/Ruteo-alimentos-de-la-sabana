import math
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Optimizador de Rutas de Transporte - Sabana de Bogotá",
    page_icon="🚚",
    layout="wide"
)

# --- FUNCIONES DE CÁLCULO ---

def calcular_distancia_euclidiana(coord1, coord2):
    """
    Calcula la distancia en línea recta (Euclidiana) entre dos coordenadas GPS.
    Utiliza el factor de conversión aproximado de 111,000 metros por cada grado.
    Retorna la distancia en metros enteros.
    """
    lat1, lon1 = coord1[0], coord1[1]
    lat2, lon2 = coord2[0], coord2[1]

    # Diferencias en latitud y longitud
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Teorema de Pitágoras para hallar la distancia en grados decimales
    distancia_grados = math.sqrt(delta_lat**2 + delta_lon**2)

    # Conversión directa a metros (1 grado aprox = 111 km = 111,000 m)
    distancia_metros = distancia_grados * 111000

    return int(distancia_metros)

# --- MODELO DE DATOS DINÁMICO ---

def resolver_ruteo(data):
    """ Inicializa y resuelve el problema de ruteo óptimo usando Google OR-Tools """
    
    # Gestor de índices de tránsito de OR-Tools
    manager = pywrapcp.RoutingIndexManager(
        len(data['matriz_distancias']),
        data['num_vehiculos'],
        data['deposito']
    )
    routing = pywrapcp.RoutingModel(manager)

    # Callback de distancia (costo de viaje entre nodos)
    def callback_distancia(desde_index, hacia_index):
        desde_nodo = manager.IndexToNode(desde_index)
        hacia_nodo = manager.IndexToNode(hacia_index)
        return data['matriz_distancias'][desde_nodo][hacia_nodo]

    transit_callback_index = routing.RegisterTransitCallback(callback_distancia)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Callback de demanda (peso cargado en cada nodo)
    def callback_demanda(desde_index):
        desde_nodo = manager.IndexToNode(desde_index)
        return data['demandas'][desde_nodo]

    demand_callback_index = routing.RegisterUnaryTransitCallback(callback_demanda)

    # Añadir dimensión de capacidad (CVRP)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Holgura de capacidad permitida
        data['capacidades_vehiculos'],  # Capacidades de los vehículos
        True,  # Forzar inicio de carga en 0
        'Capacidad'
    )

    # Configuración de parámetros de búsqueda
    parametros_busqueda = pywrapcp.DefaultRoutingSearchParameters()
    parametros_busqueda.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    parametros_busqueda.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parametros_busqueda.time_limit.seconds = 2

    # Resolver problema
    solucion = routing.SolveWithParameters(parametros_busqueda)

    # Procesar resultados
    if solucion:
        return extraer_rutas(data, manager, routing, solucion)
    else:
        return None

def extraer_rutas(data, manager, routing, solution):
    """ Convierte la solución de OR-Tools en una estructura de datos de Python más amigable """
    rutas = []
    for id_vehiculo in range(data['num_vehiculos']):
        ruta_vehiculo = []
        index = routing.Start(id_vehiculo)
        distancia_acumulada = 0

        while not routing.IsEnd(index):
            nodo_actual = manager.IndexToNode(index)
            indice_anterior = index
            index = solution.Value(routing.NextVar(index))
            distancia_tramo = routing.GetArcCostForVehicle(indice_anterior, index, id_vehiculo)
            distancia_acumulada += distancia_tramo

            ruta_vehiculo.append({
                'nodo': nodo_actual,
                'nombre': data['nombres_nodos'][nodo_actual],
                'coordenadas': data['coordenadas'][nodo_actual],
                'demanda': data['demandas'][nodo_actual]
            })

        # Añadir el nodo de retorno (depósito)
        nodo_final = manager.IndexToNode(index)
        ruta_vehiculo.append({
            'nodo': nodo_final,
            'nombre': data['nombres_nodos'][nodo_final],
            'coordenadas': data['coordenadas'][nodo_final],
            'demanda': data['demandas'][nodo_final]
        })

        rutas.append({
            'id_vehiculo': id_vehiculo + 1,
            'trayecto': ruta_vehiculo,
            'distancia_total_m': distancia_acumulada,
            'capacidad_maxima': data['capacidades_vehiculos'][id_vehiculo]
        })
    return rutas

def graficar_rutas(data, rutas):
    """ Genera un mapa bidimensional con las rutas, clientes y depósito """
    fig, ax = plt.subplots(figsize=(10, 8))

    colores_rutas = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

    coords = data['coordenadas']
    nombres = data['nombres_nodos']

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    # Dibujar clientes
    ax.scatter(lons[1:], lats[1:], color='#E67E22', s=180, zorder=5, label='Clientes')
    # Dibujar CEDI
    ax.scatter(lons[0], lats[0], color='#2C3E50', s=350, marker='H', zorder=6, label=f"CEDI: {nombres[0]}")

    # Dibujar trayectos de los camiones
    for idx, r in enumerate(rutas):
        trayecto = r['trayecto']
        if len(trayecto) <= 2 and sum([p['demanda'] for p in trayecto]) == 0:
            continue

        color = colores_rutas[idx % len(colores_rutas)]
        ruta_x = [p['coordenadas'][1] for p in trayecto]
        ruta_y = [p['coordenadas'][0] for p in trayecto]

        ax.plot(ruta_x, ruta_y, color=color, linestyle='-', linewidth=2.5,
                 label=f'Camión {r["id_vehiculo"]} ({r["distancia_total_m"]/1000:.2f} km)', zorder=3)

        # Flechas direccionales
        for i in range(len(trayecto) - 1):
            x_origen = trayecto[i]['coordenadas'][1]
            y_origen = trayecto[i]['coordenadas'][0]
            x_destino = trayecto[i+1]['coordenadas'][1]
            y_destino = trayecto[i+1]['coordenadas'][0]

            dx = x_destino - x_origen
            dy = y_destino - y_origen

            ax.annotate('', xy=(x_origen + dx*0.55, y_origen + dy*0.55),
                         xytext=(x_origen + dx*0.45, y_origen + dy*0.45),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2.5, mutation_scale=15),
                         zorder=4)

    # Etiquetas de nombres de los nodos
    for i, nombre in enumerate(nombres):
        offset_y = 0.003 if i != 0 else -0.005
        offset_x = 0.001

        etiqueta = f"{nombre}\n(+{data['demandas'][i]} kg)" if i != 0 else f"🏬 {nombres[0]}"
        ax.text(coords[i][1] + offset_x, coords[i][0] + offset_y, etiqueta,
                 fontsize=9, weight='bold' if i==0 else 'normal',
                 bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.2'),
                 zorder=7)

    ax.set_title("MAPA DE DISTRIBUCIÓN Y RUTAS OPTIMIZADAS", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Longitud", fontsize=11, labelpad=10)
    ax.set_ylabel("Latitud", fontsize=11, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, shadow=True, facecolor='white')

    return fig

# --- INTERFAZ STREAMLIT ---

st.title("🚚 Optimizador de Rutas de Entrega Capasitado (CVRP)")
st.markdown("""
Esta herramienta calcula el ruteo óptimo para una flota de vehículos con capacidad limitada de carga, 
partiendo desde un Centro de Distribución (CEDI) común hacia múltiples clientes georreferenciados.
""")

# Valores por defecto para inicializar la aplicación
if 'cedi' not in st.session_state:
    st.session_state['cedi'] = {
        'nombre': "CEDI Tocancipá",
        'lat': 4.964,
        'lon': -73.912
    }

if 'clientes' not in st.session_state:
    # Datos iniciales tomados de tu notebook
    st.session_state['clientes'] = [
        {"nombre": "Centro de Chia (Parque principal)", "lat": 4.863, "lon": -74.053, "demanda": 1100},
        {"nombre": "Sector Comercial Cajicá", "lat": 4.918, "lon": -74.029, "demanda": 750},
        {"nombre": "Zona de súpermercados Zipaquirá", "lat": 4.996, "lon": -74.003, "demanda": 1400},
        {"nombre": "Almacén de Distribución Sopó", "lat": 4.908, "lon": -73.938, "demanda": 900},
        {"nombre": "Punto de venta Briceño", "lat": 4.945, "lon": -73.921, "demanda": 500}
    ]

# Usar columnas para organizar la barra lateral y el flujo de entrada
col_config, col_map = st.columns([1, 2])

with col_config:
    st.header("⚙️ Configuración del Sistema")
    
    # 1. Configuración del CEDI
    st.subheader("🏬 Centro de Distribución")
    cedi_nombre = st.text_input("Nombre del CEDI", st.session_state['cedi']['nombre'])
    col_cedi_lat, col_cedi_lon = st.columns(2)
    with col_cedi_lat:
        cedi_lat = st.number_input("Latitud CEDI", value=st.session_state['cedi']['lat'], format="%.4f")
    with col_cedi_lon:
        cedi_lon = st.number_input("Longitud CEDI", value=st.session_state['cedi']['lon'], format="%.4f")
        
    st.session_state['cedi'] = {'nombre': cedi_nombre, 'lat': cedi_lat, 'lon': cedi_lon}

    # 2. Configuración de Flota
    st.subheader("🚚 Flota de Vehículos")
    num_camiones = st.number_input("Número de camiones disponibles", min_value=1, max_value=10, value=3)
    capacidad_camion = st.number_input("Capacidad de carga por camión (kg)", min_value=100, max_value=10000, value=2200)
    
    # 3. Administrar Clientes
    st.subheader("📍 Clientes")
    
    # Formulario para agregar nuevo cliente
    with st.expander("➕ Agregar nuevo cliente"):
        nuevo_nombre = st.text_input("Nombre del Punto", value="Nuevo Punto")
        col_new_lat, col_new_lon = st.columns(2)
        with col_new_lat:
            nuevo_lat = st.number_input("Latitud Cliente", value=4.950, format="%.4f")
        with col_new_lon:
            nuevo_lon = st.number_input("Longitud Cliente", value=-74.000, format="%.4f")
        nueva_demanda = st.number_input("Demanda del Cliente (kg)", min_value=1, value=500)
        
        if st.button("Guardar Cliente"):
            st.session_state['clientes'].append({
                "nombre": nuevo_nombre,
                "lat": nuevo_lat,
                "lon": nuevo_lon,
                "demanda": nueva_demanda
            })
            st.rerun()

    # Mostrar la tabla actual de clientes y opción de eliminarlos
    st.markdown("**Lista de Clientes Actuales:**")
    df_clientes = pd.DataFrame(st.session_state['clientes'])
    
    # Crear un control para borrar clientes uno a uno
    for index, row in df_clientes.iterrows():
        col_name, col_delete = st.columns([4, 1])
        with col_name:
            st.text(f"{row['nombre']} ({row['demanda']} kg)")
        with col_delete:
            if st.button("🗑️", key=f"del_{index}"):
                st.session_state['clientes'].pop(index)
                st.rerun()

# Preparación de datos para enviar al optimizador
nodos_coordenadas = [[st.session_state['cedi']['lat'], st.session_state['cedi']['lon']]]
nodos_nombres = [st.session_state['cedi']['nombre']]
nodos_demandas = [0] # El depósito no tiene demanda

for c in st.session_state['clientes']:
    nodos_coordenadas.append([c['lat'], c['lon']])
    nodos_nombres.append(c['nombre'])
    nodos_demandas.append(c['demanda'])

# Construcción dinámica de la matriz de distancias euclidianas
num_puntos = len(nodos_coordenadas)
matriz_distancias = []
for i in range(num_puntos):
    fila = []
    for j in range(num_puntos):
        fila.append(calcular_distancia_euclidiana(nodos_coordenadas[i], nodos_coordenadas[j]))
    matriz_distancias.append(fila)

modelo_datos = {
    'coordenadas': nodos_coordenadas,
    'nombres_nodos': nodos_nombres,
    'matriz_distancias': matriz_distancias,
    'demandas': nodos_demandas,
    'capacidades_vehiculos': [capacidad_camion] * int(num_camiones),
    'num_vehiculos': int(num_camiones),
    'deposito': 0
}

# --- EJECUCIÓN DEL MODELO ---
with col_map:
    st.header("🗺️ Resultados y Mapa de Distribución")
    
    demanda_total = sum(nodos_demandas)
    capacidad_total = capacidad_camion * num_camiones
    
    # Validaciones iniciales
    if len(st.session_state['clientes']) == 0:
        st.warning("⚠️ Agrega al menos un cliente en la barra de configuración para iniciar el cálculo.")
    elif demanda_total > capacidad_total:
        st.error(f"❌ Capacidad insuficiente. La demanda total es de **{demanda_total:,} kg**, pero la capacidad máxima de tu flota es de **{capacidad_total:,} kg** ({num_camiones} camiones de {capacidad_camion} kg).")
    else:
        # Botón para ejecutar el optimizador de rutas
        if st.button("🚀 Calcular Ruteo Óptimo", use_container_width=True):
            with st.spinner("Calculando las mejores trayectorias..."):
                rutas = resolver_ruteo(modelo_datos)
                
                if rutas:
                    # Mostrar consolidado global
                    distancia_total_m = sum([r['distancia_total_m'] for r in rutas])
                    mercancia_despachada = sum([sum([p['demanda'] for p in r['trayecto']]) for r in rutas])
                    
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    with col_metric1:
                        st.metric("Distancia Total Recorrida", f"{distancia_total_m / 1000:.2f} km")
                    with col_metric2:
                        st.metric("Mercancía Entregada", f"{mercancia_despachada:,} kg")
                    with col_metric3:
                        camiones_usados = sum(1 for r in rutas if sum([p['demanda'] for p in r['trayecto']]) > 0)
                        st.metric("Flota Utilizada", f"{camiones_usados} de {num_camiones} Furgones")
                    
                    # Dibujar gráfico de matplotlib en streamlit
                    figura_mapa = graficar_rutas(modelo_datos, rutas)
                    st.pyplot(figura_mapa)
                    
                    # Desglose de rutas por cada vehículo
                    st.subheader("📋 Detalle de Ruta por Vehículo")
                    for r in rutas:
                        id_v = r['id_vehiculo']
                        trayecto = r['trayecto']
                        distancia_km = r['distancia_total_m'] / 1000
                        cap_max = r['capacidad_maxima']
                        carga_total_vehiculo = sum([p['demanda'] for p in trayecto])
                        eficiencia = (carga_total_vehiculo / cap_max) * 100
                        
                        # Saltar camiones vacíos
                        if len(trayecto) <= 2 and carga_total_vehiculo == 0:
                            continue
                        
                        with st.expander(f"🚚 Furgón #{id_v} - Carga: {carga_total_vehiculo:,} kg / {cap_max:,} kg ({eficiencia:.1f}%)"):
                            st.write(f"**Distancia de Viaje:** {distancia_km:.2f} Kilómetros")
                            st.write("**Secuencia de Entregas:**")
                            
                            secuencia = []
                            carga_actual = 0
                            for p in trayecto:
                                if p['demanda'] > 0:
                                    carga_actual += p['demanda']
                                    secuencia.append(f"👉 **{p['nombre']}** (Carga: +{p['demanda']} kg / Acumulado: {carga_actual} kg)")
                                else:
                                    secuencia.append(f"🏠 **{p['nombre']}** (CEDI)")
                            
                            st.markdown("\n".join(secuencia))
                else:
                    st.error("❌ No se pudo encontrar una solución viable dentro del límite de tiempo. Prueba aumentando el número de vehículos o sus capacidades.")
