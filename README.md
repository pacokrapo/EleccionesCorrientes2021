Este repositorio muestra como trabajé para hacer un visualizador de los resultados de las elecciones 2021 en la provincia de Corrientes.

Partiendo de la página web [Elecciones 2021](https://elecciones2021.corrientes.gob.ar/), descargué la lista de nombres de las escuelas y a partir de estos nombres los adapté para que sean como los distintos links de cada escuela e hice un proceso de Web Scrapping.

Una vez hecho el Web Scrapping modifiqué los nombres de los archivos, obtuve coordenadas de las escuelas que me permitió la API de Google (no todas obtenian resultados precisos) e hice el anidado de datos.

Una vez anidados y listos los datos, hice el visualizador de datos en Streamlit, por un lado se pueden ver los resultados de un determinado área geográfica de la provincia y por el otro un mapa de calor de los resultados.

Si bien lo hice para el partido MID, es apto para ser utilizado por todos los demás partidos también.
