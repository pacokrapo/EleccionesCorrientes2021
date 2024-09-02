import streamlit as st
import googlemaps
import folium
from streamlit_folium import st_folium
import pandas as pd
from geopy.distance import geodesic
import pickle
import plotly.graph_objects as go

from streamlit_folium import folium_static
from folium.plugins import HeatMap
import geopandas as gpd

import os

# Configurar la API de Google
API_KEY = os.environ.get("API_KEY")
gmaps = googlemaps.Client(key=API_KEY)

# Cargar el dataset de escuelas (esto debe contener las coordenadas de cada escuela)
with open('Datos.pkl', 'rb') as f:
    Datos = pickle.load(f)

# Configurar el sidebar con las opciones
opcion = st.sidebar.selectbox(
    'Selecciona una visualización',
    ('Porcentajes', 'Mapa de Calor')
)

if opcion == 'Porcentajes':

    # Solicitar al usuario que ingrese un término de búsqueda
    search_query = st.text_input("Ingresa el nombre de un lugar para buscar y presiona enter:")

    # Definir el radio de búsqueda (en kilómetros)
    radio_busqueda = st.slider("Radio de búsqueda (km):", 1, 50, 5)

    if search_query:
        # Realizar la búsqueda utilizando la API de Google Places
        places_result = gmaps.geocode(search_query)
        
        if places_result:
            # Obtener las coordenadas del primer resultado
            location = places_result[0]['geometry']['location']
            lugar_latlng = (location['lat'], location['lng'])

            # Filtrar las escuelas dentro del radio especificado
            escuelas_cercanas = []
            for idx, row in Datos.iterrows():
                escuela_latlng = (row['Latitud'], row['Longitud'])
                distancia = geodesic(lugar_latlng, escuela_latlng).km
                if distancia <= radio_busqueda:
                    escuelas_cercanas.append({
                        'Escuela': row['Escuela'],
                        'Latitud': row['Latitud'],
                        'Longitud': row['Longitud'],
                        'Distancia': distancia
                    })
            
            # Convertir la lista a un DataFrame
            df_escuelas = pd.DataFrame(escuelas_cercanas)
            
            if not df_escuelas.empty:
                # Mostrar las escuelas encontradas
                st.write(f"Escuelas dentro de {radio_busqueda} km de {search_query}:")
                st.dataframe(df_escuelas[['Escuela', 'Distancia']])
                
                # Crear el mapa centrado en el lugar buscado
                m = folium.Map(location=lugar_latlng, zoom_start=13)

                # Añadir marcador del lugar buscado
                folium.Marker(
                    location=lugar_latlng,
                    popup=f"<strong>{search_query}</strong>",
                    icon=folium.Icon(color='blue')
                ).add_to(m)

                folium.Circle(
                    location=lugar_latlng,
                    radius=radio_busqueda * 1000,  # 'radio_busqueda' está en km, convertimos a metros
                    color='blue',
                    fill=False,
                    fill_opacity=0.2
                ).add_to(m)

                # Añadir marcadores para las escuelas encontradas
                for idx, row in df_escuelas.iterrows():
                    folium.Marker(
                        location=[row['Latitud'], row['Longitud']],
                        popup=row['Escuela'],
                        tooltip=row['Escuela'],
                        icon=folium.Icon(color='red')
                    ).add_to(m)

                # Mostrar el mapa en Streamlit
                output = st_folium(m, width=700, height=500)


                st.write("Seleccione una escuela para ver los resultados en particular:")
                # Intentar capturar el objeto clickeado
                if output and 'last_object_clicked_popup' in output:
                    last_clicked = output['last_object_clicked_popup']

                else:
                    st.write("No se detectó ningún clic en los marcadores.")

                # Filtrar la fila correspondiente en el DataFrame 'Datos' donde la columna 'Escuela' coincida con el nombre almacenado en 'clicked_school'
                selected_row = Datos[Datos['Escuela'] == last_clicked]

                # Verifica si se encontró la fila
                if not selected_row.empty:
                    # Haz lo que necesites con 'selected_row'
                    df_seleccionado = selected_row['Datos'].values[0]  # Supongamos que 'Datos' tiene un DataFrame

                    st.dataframe(df_seleccionado)
                else:
                    st.warning("No hay selección.")

                # Filtrar las filas de 'Datos' correspondientes a las escuelas cercanas
                nombres_escuelas = df_escuelas['Escuela']
                datos_cercanos = Datos[Datos['Escuela'].isin(nombres_escuelas)]['Datos']
                # Inicializar un DataFrame vacío para acumular los resultados
                resultados_acumulados = None
                partidos = []
                ordencolumnas = ["gobernador", "senadores", "diputados", "intendente", "concejales"]

                # Suponiendo que datos_cercanos contiene varios DataFrames y eliges uno de ellos
                df = datos_cercanos.iloc[0]  # Tomamos el primer DataFrame de la lista

                # Reemplazar '-' por 0 y convertir a numérico solo las columnas numéricas
                df.replace('-', 0, inplace=True)
                df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

                # Excluir solo la primera columna y limitar las filas a 59
                numeric_cols = df.iloc[:59, 1:]  # Tomar solo las primeras 59 filas excluyendo la primera columna

                # Guardar los nombres de los partidos de esas 59 filas
                partidos = df.iloc[:59, 0].tolist()

                # Si tienes múltiples DataFrames en datos_cercanos, sumar solo los primeros 59 de cada uno
                for df in datos_cercanos:
                    # Reemplazar '-' por 0 y convertir a numérico solo las columnas numéricas
                    df.replace('-', 0, inplace=True)
                    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

                    # Excluir solo la primera columna y limitar las filas a 59
                    numeric_cols = df.iloc[:59, 1:]

                    # Asegurarse de que las categorías faltantes se incluyan en el DataFrame
                    if resultados_acumulados is None:
                        resultados_acumulados = numeric_cols.copy()
                    else:
                        # Reindexar para asegurarse de que ambos DataFrames tengan las mismas categorías (columnas y filas)
                        resultados_acumulados = resultados_acumulados.reindex(numeric_cols.index.union(resultados_acumulados.index),
                                                                            columns=numeric_cols.columns.union(resultados_acumulados.columns)).fillna(0)
                        numeric_cols = numeric_cols.reindex(resultados_acumulados.index, columns=resultados_acumulados.columns).fillna(0)

                        # Sumar el DataFrame al acumulador
                        resultados_acumulados = resultados_acumulados.add(numeric_cols, fill_value=0)

                # Crear un DataFrame final con la columna de 'partido' y los resultados acumulados
                if resultados_acumulados is not None:
                    if len(partidos) == len(resultados_acumulados):
                        resultados_acumulados.insert(0, 'partido', partidos)
                    else:
                        st.error(f"No hay resultados disponibles para esta localidad.")

                # Reordenar las columnas al final, después de la suma
                ordencolumnas_presentes = [col for col in ordencolumnas if col in resultados_acumulados.columns]

                if resultados_acumulados is not None and 'partido' in resultados_acumulados.columns:
                    resultados_acumulados = resultados_acumulados[['partido'] + ordencolumnas_presentes]

                # Mostrar los resultados acumulados
                st.write("Resultados acumulados para las escuelas dentro del área:")
                st.dataframe(resultados_acumulados)

                # Selector de partido
                partidos = resultados_acumulados['partido'].unique()
                partido_seleccionado = st.selectbox("Selecciona un partido", partidos)

                # Filtrar los resultados para el partido seleccionado
                resultados_partido = resultados_acumulados[resultados_acumulados['partido'] == partido_seleccionado].iloc[0, 1:]

                # Sumar los resultados de los otros partidos
                resultados_otros = resultados_acumulados[resultados_acumulados['partido'] != partido_seleccionado].iloc[:, 1:].mean()

                # Calcular el total de votos para cada categoría (la suma de todos los partidos)
                total_por_categoria = resultados_acumulados.iloc[:, 1:].sum()

                # Calcular el porcentaje del partido seleccionado en cada categoría
                porcentajes = (resultados_partido / total_por_categoria) * 100

                # Título principal
                st.write(f"Porcentajes para el partido {partido_seleccionado}:")

                # Crear una fila con 3 columnas
                col1, col2, col3 = st.columns(3)

                # Asumiendo que porcentajes es un array de NumPy
                categorias = list(porcentajes.keys())  # O asegurarte de que tienes las claves correctas si es un diccionario
                porcentajes_valores = porcentajes  # Si porcentajes ya es un array, no necesitas convertirlo

                # Asignar cada categoría y porcentaje a una columna
                for i, (categoria, porcentaje) in enumerate(zip(categorias, porcentajes_valores)):
                    if i % 3 == 0:
                        with col1:
                            st.markdown(f"**{categoria.capitalize()}**")
                            st.markdown(f"<h2>{porcentaje:.2f}%</h2>", unsafe_allow_html=True)
                    elif i % 3 == 1:
                        with col2:
                            st.markdown(f"**{categoria.capitalize()}**")
                            st.markdown(f"<h2>{porcentaje:.2f}%</h2>", unsafe_allow_html=True)
                    else:
                        with col3:
                            st.markdown(f"**{categoria.capitalize()}**")
                            st.markdown(f"<h2>{porcentaje:.2f}%</h2>", unsafe_allow_html=True)

                # Gráfico de barras del promedio de los partidos ordenado
                # Calcular el promedio de todos los resultados por partido
                resultados_promedio = resultados_acumulados.iloc[:, 1:].mean(axis=1)

                # Crear un nuevo DataFrame con los promedios
                df_promedios = pd.DataFrame({
                    'partido': resultados_acumulados['partido'],
                    'promedio': resultados_promedio
                })

                # Ordenar el DataFrame por los promedios
                df_promedios = df_promedios.sort_values(by='promedio', ascending=False)

                # Resaltar el partido seleccionado cambiando el color de su barra
                colors = ['#1f77b4' if partido != partido_seleccionado else '#ff7f0e' for partido in df_promedios['partido']]

                # Crear el gráfico de barras con Plotly
                fig_barras = go.Figure(data=[go.Bar(
                    x=df_promedios['partido'],
                    y=df_promedios['promedio'],
                    marker_color=colors  # Aplica los colores definidos
                )])

                # Personalizar el gráfico
                fig_barras.update_layout(
                    title="Promedio de votos por partido (ordenado)",
                    xaxis_title='Partido',
                    yaxis_title='Promedio de votos',
                    showlegend=False,
                    height=800,
                    xaxis=dict(
                        tickmode='array',  # Modo de ticks personalizado
                        tickvals=list(range(len(df_promedios['partido']))),  # Valores de ticks (uno para cada partido)
                        ticktext=df_promedios['partido'],  # Textos para los ticks (nombres de partidos)
                    )
                )

                # Mostrar el gráfico de barras en Streamlit
                st.plotly_chart(fig_barras, use_container_width=True)
            else:
                st.warning(f"No se encontraron escuelas dentro de {radio_busqueda} km de {search_query}.")
        else:
            st.error("No se encontraron resultados para la búsqueda.")

elif opcion == 'Mapa de Calor':
    st.write("Mapa de calor")

    ListaPartidos = Datos["Datos"][0]["partido"].unique()
    
    partido_seleccionado = st.selectbox(
    'Selecciona un partido para ver el mapa de calor de este:',
    ListaPartidos
    )

    indice_partido = None
    for i in range(len(Datos)):
        if partido_seleccionado in Datos["Datos"][0]["partido"].unique():
            indice_partido = i
            break

    
    if indice_partido is not None:

        # Crear una lista vacía para almacenar los datos del partido seleccionado
        lista_datos_partido = []

        # Iterar sobre cada DataFrame en Datos["Datos"]
        for i in range(len(Datos["Datos"])):
            # Filtrar las filas que corresponden al partido seleccionado
            df_partido_temp = Datos["Datos"][i][Datos["Datos"][i]["partido"] == partido_seleccionado]
            
            # Añadir las columnas de Latitud y Longitud desde el DataFrame principal
            df_partido_temp['Latitud'] = Datos['Latitud'][i]
            df_partido_temp['Longitud'] = Datos['Longitud'][i]
            
            # Añadir las filas filtradas a la lista
            lista_datos_partido.append(df_partido_temp)

        lista_totales = []
        for i in range(len(Datos["Datos"])):
            # Acceder al DataFrame anidado
            df = Datos["Datos"][i]

            # Reemplazar los valores "-" por 0
            df_replaced = df.replace("-", 0, regex=True)

            # Asegurarse de que todos los valores sean numéricos
            df_replaced = df_replaced.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Seleccionar todas las filas y columnas, excluyendo la primera columna y la primera fila
            df_sliced = df_replaced.iloc[1:, 1:]

            # Calcular la sumatoria de todas las columnas y filas
            sumatoria_total = df_sliced.to_numpy().sum()

            # Obtener la cantidad de columnas (sin contar la primera)
            num_columnas = df_sliced.shape[1]

            # Calcular el promedio (o total de votos, según tu interpretación)
            promedio_votos = sumatoria_total / num_columnas

            lista_totales.append(promedio_votos)

        # Concatenar todos los DataFrames de la lista en un solo DataFrame
        df_partido = pd.concat(lista_datos_partido, ignore_index=True)

        # Opción para seleccionar entre valores netos y valores porcentuales
        opcion = st.selectbox("Seleccione el tipo de valores:", ["Valores Netos", "Valores Porcentuales"])

        # Convertir las columnas de interés a numéricas (por si acaso contienen cadenas)
        columnas_interes = ["gobernador", "senadores", "diputados", "intendente", "concejales"]

        columnas_presentes = [col for col in columnas_interes if col in df_partido.columns]

        df_partido[columnas_presentes] = df_partido[columnas_presentes].apply(pd.to_numeric, errors='coerce')

        # Reemplazar los valores negativos (representando ceros) por ceros
        df_partido[columnas_presentes] = df_partido[columnas_presentes].replace(-1, 0)

        if opcion == "Valores Netos":
            # Sumar las columnas de interés y calcular el promedio (sin tener en cuenta Latitud y Longitud)
            df_partido["total_votos"] = df_partido[columnas_presentes].sum(axis=1)
            df_partido["promedio_votos"] = df_partido["total_votos"] / len(columnas_presentes)


            # Añadir las columnas de Latitud y Longitud desde el DataFrame principal
            df_partido['Latitud'] = Datos['Latitud']
            df_partido['Longitud'] = Datos['Longitud']

            # Definir los límites de latitud y longitud
            lat_min = -30.0
            lat_max = -26.5
            lon_min = -59.5
            lon_max = -56.5

            # Filtrar el DataFrame para eliminar las filas que cumplen con las características especificadas
            df_partido = df_partido[
                ~((df_partido['Latitud'] == -28.5841599) & (df_partido['Longitud'] == -58.00719220000001)) &
                (df_partido['Latitud'].between(lat_min, lat_max)) &
                (df_partido['Longitud'].between(lon_min, lon_max))
            ]
            
            # Convertir DataFrame a GeoDataFrame, utilizando las coordenadas para la geometría
            gdf = gpd.GeoDataFrame(df_partido, geometry=gpd.points_from_xy(df_partido['Longitud'], df_partido['Latitud']))

            # Calcular la intensidad de votos para el partido seleccionado (evitar divisiones por cero)
            gdf['intensidad_votos'] = gdf.apply(
                lambda row: row['promedio_votos'] / row['total_votos'] if row['total_votos'] > 0 else 0,
                axis=1
            )

            # Crear un mapa base centrado en la ubicación media de las escuelas
            map_center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
            m = folium.Map(location=map_center, zoom_start=10)

            # Crear un mapa de calor con los datos de intensidad de votos
            heat_data = [[point.xy[1][0], point.xy[0][0], intensidad] for point, intensidad in zip(gdf.geometry, gdf['intensidad_votos'])]
            HeatMap(heat_data, radius=15).add_to(m)

            # Mostrar el mapa en Streamlit
            st.title(f"Mapa de calor de votos para {partido_seleccionado}")
            folium_static(m)
        elif opcion == "Valores Porcentuales":       
            # Calcular la suma de los votos totales del partido por fila
            df_partido["total_votos"] = df_partido[columnas_presentes].sum(axis=1)
            
            df_partido["totales"] = lista_totales
            # Dividir el total de votos del partido por los valores correspondientes en lista_totales
            # Aquí asumimos que lista_totales tiene un valor correspondiente para cada fila del DataFrame.
            df_partido["promedio_votos"] = df_partido["total_votos"] / df_partido["totales"]

             # Añadir las columnas de Latitud y Longitud desde el DataFrame principal
            df_partido['Latitud'] = Datos['Latitud']
            df_partido['Longitud'] = Datos['Longitud']

            # Definir los límites de latitud y longitud
            lat_min = -30.0
            lat_max = -26.5
            lon_min = -59.5
            lon_max = -56.5

            # Filtrar el DataFrame para eliminar las filas que cumplen con las características especificadas
            df_partido = df_partido[
                ~((df_partido['Latitud'] == -28.5841599) & (df_partido['Longitud'] == -58.00719220000001)) &
                (df_partido['Latitud'].between(lat_min, lat_max)) &
                (df_partido['Longitud'].between(lon_min, lon_max))
            ]

            # Convertir DataFrame a GeoDataFrame, utilizando las coordenadas para la geometría
            gdf = gpd.GeoDataFrame(df_partido, geometry=gpd.points_from_xy(df_partido['Longitud'], df_partido['Latitud']))

            # Usar promedio_votos directamente como intensidad de votos
            gdf['intensidad_votos'] = gdf['promedio_votos']

            # Crear un mapa base centrado en la ubicación media de las escuelas
            map_center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
            m = folium.Map(location=map_center, zoom_start=10)

            # Crear un mapa de calor con los datos de intensidad de votos
            heat_data = [[point.xy[1][0], point.xy[0][0], intensidad] for point, intensidad in zip(gdf.geometry, gdf['intensidad_votos'])]
            HeatMap(heat_data, radius=15).add_to(m)

            # Mostrar el mapa en Streamlit
            st.title(f"Mapa de calor de votos para {partido_seleccionado}")
            folium_static(m)
    else:
        st.write("No se encontraron datos para el partido seleccionado.")