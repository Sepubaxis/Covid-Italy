---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.0
  nbformat: 4
  nbformat_minor: 5
---

::: {#028b8f2b-9e6b-48d9-b8b7-64661867912b .cell .markdown}
# **Covid en italia**
:::

::: {#86c69d26-40f8-4299-9a23-28e64409523a .cell .markdown}
![image.png](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/d138b027-d722-4051-acb6-7c451f031695.png)
:::

::: {#5bc7a7a0-4f11-4cec-a48f-e6627b5bb9e2 .cell .markdown}
## Trabajo de Alexis Alvarez \| Documento: 38691776 {#trabajo-de-alexis-alvarez--documento-38691776}

`<a id="index">`{=html}`</a>`{=html}

### Tabla de contenido:

-   [Intro](#section_intro)
-   [EDA](#section_descriptive)
-   [Limpieza de Datos](#section_limpieza)
-   [Medidas de dispersión](#covarianza)
-   [Graficos](#section_grafico)
-   [Mapa de Correlacion (estandarizacion/normalizacion)](#corr)
-   [Regresion lineal](#lineal)
-   [Dataframe 2 (merge)](#merge)
-   [Concluciones](#concluciones)
:::

::: {#f755c9fe-aa4b-47ee-ba71-dc2c3739e052 .cell .markdown}
# Intro`<a id="section_intro">`{=html}`</a>`{=html}:

Trabajo de Alexis Alvarez, grupo 11. Solamente yo de intengrante. Voy a
realizar un analisis sobre el Covid-19 en Italia e intentar utilizar
algunas cosas vistas en clase y ademas, responder a algunas preguntas y
concluciones.

Tambien voy a utilizar algunos metodos encontrados en los colab
trabajados en clase
:::

::: {#a68ea29a-d6e2-4f6f-ac4c-7eab72f4b03b .cell .markdown}
## **Analisis exploratorio de los datos**

`<a id="section_descriptive">`{=html}`</a>`{=html}

### Primero importamos librerias
:::

::: {#1af72f53-a86a-4bea-b34f-d041c3180964 .cell .code execution_count="1" tags="[]"}
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
```
:::

::: {#d3b25f85-8889-4fd9-ac60-4d31c5cc458d .cell .markdown}
esta funcion de abajo nos permite utilizar una url para agarrar datos,
en este caso github
:::

::: {#93c50b01-e322-4789-ab8b-a957ec3c2c03 .cell .code execution_count="2" tags="[]"}
``` python
from urllib.request import urlretrieve
```
:::

::: {#9e39fddd-8e5e-4c6f-a4e4-a58f022beafc .cell .code execution_count="3"}
``` python
italy_covid_url = 'https://gist.githubusercontent.com/aakashns/f6a004fa20c84fec53262f9a8bfee775/raw/f309558b1cf5103424cef58e2ecb8704dcd4d74c/italy-covid-daywise.csv'

urlretrieve(italy_covid_url, 'italy-covid-daywise.csv')
```

::: {.output .execute_result execution_count="3"}
    ('italy-covid-daywise.csv', <http.client.HTTPMessage at 0x1a4a7848d50>)
:::
:::

::: {#96a357df-fbc0-49d9-9bbb-a2ae859b7180 .cell .markdown}
### Ahora el archivo \"italy-covid-daywise.csv\" queda guardado en nuestra carpeta {#ahora-el-archivo-italy-covid-daywisecsv-queda-guardado-en-nuestra-carpeta}
:::

::: {#4c83211f-3172-4d29-a70e-53ee7cdfebfe .cell .code execution_count="4" tags="[]"}
``` python
covid_df = pd.read_csv('italy-covid-daywise.csv')
```
:::

::: {#2a618430-46b8-4961-8fde-0f7041d6a6b9 .cell .code execution_count="5" tags="[]"}
``` python
covid_df.isnull().sum()
```

::: {.output .execute_result execution_count="5"}
    date            0
    new_cases       0
    new_deaths      0
    new_tests     113
    dtype: int64
:::
:::

::: {#db5a7ce2-843f-4b2c-b455-38a7fb88269f .cell .code execution_count="6" tags="[]"}
``` python
covid_df
```

::: {.output .execute_result execution_count="6"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-08-30</td>
      <td>1444.0</td>
      <td>1.0</td>
      <td>53541.0</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-08-31</td>
      <td>1365.0</td>
      <td>4.0</td>
      <td>42583.0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2020-09-01</td>
      <td>996.0</td>
      <td>6.0</td>
      <td>54395.0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2020-09-02</td>
      <td>975.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>247</th>
      <td>2020-09-03</td>
      <td>1326.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 4 columns</p>
</div>
```
:::
:::

::: {#0094a4c4-a1ec-4d5f-8d32-6bc1ad3f5540 .cell .markdown}
Dataframe:

-   date: La fecha del dia registrado
-   new_cases: Casos nuevos de covid-19 registrados
-   new_deaths: Casos nuevos de muertes registradas por covid
-   new_tests: Pruebas de covid realizadas en la fecha

Por ahora mi data tiene solo variables cuantitativas, mas adelante
fusionare mi tabla con otra
:::

::: {#53b38c37-cf2e-4bf8-9e4c-131482d48854 .cell .code execution_count="7" tags="[]"}
``` python
covid_df.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 248 entries, 0 to 247
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   date        248 non-null    object 
     1   new_cases   248 non-null    float64
     2   new_deaths  248 non-null    float64
     3   new_tests   135 non-null    float64
    dtypes: float64(3), object(1)
    memory usage: 7.9+ KB
:::
:::

::: {#6f3defab-4fec-476c-a519-8c54c963b454 .cell .markdown}
Tenemos para visualizar informacion importante, columnas, filas, tipos
de datos
:::

::: {#71f38c12-a1d4-421c-9285-4710443054ca .cell .markdown}
`<a id="section_limpieza">`{=html}`</a>`{=html}

# Limpieza de datos
:::

::: {#24e285b4-ff82-4188-b73b-2b92d3b93811 .cell .code execution_count="8" tags="[]"}
``` python
type(covid_df)
```

::: {.output .execute_result execution_count="8"}
    pandas.core.frame.DataFrame
:::
:::

::: {#e4220a85-fbc5-447b-bc07-58638c3ef75a .cell .markdown}
type para ver si efectivamente es formato dataframe
:::

::: {#8ba07f49-2e42-4dcf-9fdb-e305cc74c31b .cell .code execution_count="9" tags="[]"}
``` python
covid_df.dtypes
```

::: {.output .execute_result execution_count="9"}
    date           object
    new_cases     float64
    new_deaths    float64
    new_tests     float64
    dtype: object
:::
:::

::: {#32acfcc2-4271-48bb-9b5d-d3bfd6b4d10d .cell .markdown}
Dtype comprueba los tipos de dato dentro del dataframe
:::

::: {#5a3c1ca1-97b8-41a1-a185-c5e1b4b206f3 .cell .code execution_count="10" tags="[]"}
``` python
covid_df[['new_cases', 'new_deaths', 'new_tests']].mean()
```

::: {.output .execute_result execution_count="10"}
    new_cases      1094.818548
    new_deaths      143.133065
    new_tests     31699.674074
    dtype: float64
:::
:::

::: {#580b85c8-a07a-4c21-87ed-d5f67543f304 .cell .markdown}
la media fue de:

-   1096 para **nuevos casos**
-   143 para **fallecidos**
-   31699 para **nuevas pruebas**
:::

::: {#fe3a8645-4fda-4186-82c8-bd814d42fea5 .cell .markdown}
# Manejo de datos ausentes o erroneos
:::

::: {#f4de9a54-8953-4ce6-be2a-a2c63ab16053 .cell .markdown}
## ¿Existe algún sesgo en los datos recogidos? ¿Hay errores en la codificación de los datos?
:::

::: {#c2af0099-4cad-4472-8c29-f767cefd5ede .cell .markdown}
#### Utilizando describe para ver un paneo general de los numeros, encontramos que hay en nuevos casos un numero que no corresponde (seria imposible que hayan negativos nuevos casos y muertes)
:::

::: {#c1d682a2-987d-485c-9519-11ad4fe6ab71 .cell .code execution_count="11" tags="[]"}
``` python
covid_df.describe()
```

::: {.output .execute_result execution_count="11"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>248.000000</td>
      <td>248.000000</td>
      <td>135.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1094.818548</td>
      <td>143.133065</td>
      <td>31699.674074</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1554.508002</td>
      <td>227.105538</td>
      <td>11622.209757</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-148.000000</td>
      <td>-31.000000</td>
      <td>7841.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>123.000000</td>
      <td>3.000000</td>
      <td>25259.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>342.000000</td>
      <td>17.000000</td>
      <td>29545.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1371.750000</td>
      <td>175.250000</td>
      <td>37711.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6557.000000</td>
      <td>971.000000</td>
      <td>95273.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#12e34ae0-6a34-4309-9531-3d9b2255691a .cell .markdown}
Detetados valores negativos son errores ya que es imposible que haya un
conteo negativo
:::

::: {#312f3840-e039-4832-83fd-2d5b5f3698c4 .cell .markdown}
#### Lo Arreglamos
:::

::: {#dae6ac11-a9d1-477f-a6ea-7c5deffdf5c0 .cell .code execution_count="12" tags="[]"}
``` python
covid_df.sort_values('new_cases').head(5)
```

::: {.output .execute_result execution_count="12"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>172</th>
      <td>2020-06-20</td>
      <td>-148.0</td>
      <td>47.0</td>
      <td>29875.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2020-01-29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2020-01-30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020-02-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#27387369-a689-4cfa-a282-b67bdd0a4507 .cell .markdown}
Sort_values es utilizado para ordenar los datos \'new_cases\' de menor a
mayor
:::

::: {#388651f9-a656-42a9-a15a-52bcc89ed033 .cell .code execution_count="13" tags="[]"}
``` python
np.std(covid_df['new_deaths']) #deviacion estandar
```

::: {.output .execute_result execution_count="13"}
    226.6472017652724
:::
:::

::: {#5b15efca-1687-4328-b817-cbf50cb90944 .cell .code execution_count="14" tags="[]"}
``` python
covid_df['new_deaths'].mean() #media
```

::: {.output .execute_result execution_count="14"}
    143.13306451612902
:::
:::

::: {#20b6acc5-ae8f-42ef-b73f-e9f4c93268ac .cell .code execution_count="15" tags="[]"}
``` python
covid_df.sort_values('new_deaths').head(5) #new_deaths ordenado de menor a mayor
```

::: {.output .execute_result execution_count="15"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>2020-06-25</td>
      <td>577.0</td>
      <td>-31.0</td>
      <td>29421.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2020-01-30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2020-01-31</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020-02-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2020-02-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#462c8192-5d18-41e1-818a-f0340e8cf411 .cell .markdown}
Vamos a asumir que fue un data entry error. Podemos utilizar uno de los
siguientes metodos para lidiar con valores faltantes: \*\*\*

-   Reemplazarlo con 0
-   Reemplazarlo con el promedio de las columnas adyacentes
-   Reemplazarlo con el promedio de la columna entera
-   Descartar la fila entera
:::

::: {#5336ef35-6d91-4e33-a943-b77837c30f51 .cell .code execution_count="16" tags="[]"}
``` python
covid_df.at[172, 'new_cases'] = (covid_df.at[171, 'new_cases'] + covid_df.at[173, 'new_cases'])/2
```
:::

::: {#c7200ad6-114e-4bfc-bb18-877822a850c8 .cell .markdown}
la posicion 172 (el valor incorrecto) pasa a ser la mitad de su anterior
y su proximo sumados
:::

::: {#8fd00c0b-8eac-4913-899d-e462c61b3450 .cell .code execution_count="17" tags="[]"}
``` python
covid_df[171:174] #mostramos solo esas posiciones en la tabla
```

::: {.output .execute_result execution_count="17"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>171</th>
      <td>2020-06-19</td>
      <td>331.0</td>
      <td>66.0</td>
      <td>28570.0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2020-06-20</td>
      <td>297.5</td>
      <td>47.0</td>
      <td>29875.0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2020-06-21</td>
      <td>264.0</td>
      <td>49.0</td>
      <td>24581.0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#7314b085-f39f-4e66-b264-47516c9a2b9b .cell .markdown}
Para muertes voy a asumir que el menor valor seria 0
:::

::: {#972fdf7f-a2e7-4e3b-a74f-636d8e31ddf0 .cell .code execution_count="18" tags="[]"}
``` python
covid_df.at[177, 'new_deaths'] = 0.0
```
:::

::: {#da09045f-054e-4468-85b7-2b4e3c411d63 .cell .code execution_count="19" tags="[]"}
``` python
covid_df[177:178]
```

::: {.output .execute_result execution_count="19"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177</th>
      <td>2020-06-25</td>
      <td>577.0</td>
      <td>0.0</td>
      <td>29421.0</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#5f1dddf6-6c9c-4f23-adf6-0873464a631c .cell .code execution_count="20" tags="[]"}
``` python
covid_df['new_tests'] = covid_df['new_tests'].fillna(0) #metodo fill n.a
```
:::

::: {#eff9e3fb-da23-4cf9-9b37-39d81f0ab85b .cell .code execution_count="21" tags="[]"}
``` python
covid_df.describe() #esta funcion describe todos los valores comunes calculados
```

::: {.output .execute_result execution_count="21"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>248.000000</td>
      <td>248.000000</td>
      <td>248.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1096.614919</td>
      <td>143.258065</td>
      <td>17255.870968</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1553.322957</td>
      <td>227.017821</td>
      <td>17986.924137</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>127.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>342.000000</td>
      <td>17.000000</td>
      <td>17758.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1371.750000</td>
      <td>175.250000</td>
      <td>30019.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6557.000000</td>
      <td>971.000000</td>
      <td>95273.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#198c6c0d-b486-4d87-af30-f486297e2295 .cell .markdown}
#### listo

------------------------------------------------------------------------
:::

::: {#bc10df54-35b7-4ec3-a74e-6ccb32357bd9 .cell .markdown}
Creamos nuevas columnas a base de los resultados totales, que nos puede
proporcionar informacion valiosa a la vista
:::

::: {#23dc08d7-e835-4750-bf90-bf2b66f86756 .cell .code execution_count="22" tags="[]"}
``` python
total_cases = covid_df.new_cases.sum() #sum para la suma de todos los valores
total_deaths = covid_df.new_deaths.sum()
initial_tests = 935310
total_tests = initial_tests + covid_df.new_tests.sum()
```
:::

::: {#890d5a09-148e-46cf-b1cc-cf96d1163f34 .cell .code execution_count="23" tags="[]"}
``` python
covid_df['total_cases'] = covid_df.new_cases.cumsum() #cumsum va sumando el total al proximo valor
covid_df['total_deaths'] = covid_df.new_deaths.cumsum()
covid_df['total_tests'] = covid_df.new_tests.cumsum() + initial_tests
```
:::

::: {#b16e1986-5330-41ed-a55a-106aa3bac486 .cell .code execution_count="24" tags="[]"}
``` python
covid_df
```

::: {.output .execute_result execution_count="24"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
      <th>total_cases</th>
      <th>total_deaths</th>
      <th>total_tests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-08-30</td>
      <td>1444.0</td>
      <td>1.0</td>
      <td>53541.0</td>
      <td>267298.5</td>
      <td>35504.0</td>
      <td>5117788.0</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2020-08-31</td>
      <td>1365.0</td>
      <td>4.0</td>
      <td>42583.0</td>
      <td>268663.5</td>
      <td>35508.0</td>
      <td>5160371.0</td>
    </tr>
    <tr>
      <th>245</th>
      <td>2020-09-01</td>
      <td>996.0</td>
      <td>6.0</td>
      <td>54395.0</td>
      <td>269659.5</td>
      <td>35514.0</td>
      <td>5214766.0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2020-09-02</td>
      <td>975.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>270634.5</td>
      <td>35522.0</td>
      <td>5214766.0</td>
    </tr>
    <tr>
      <th>247</th>
      <td>2020-09-03</td>
      <td>1326.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>271960.5</td>
      <td>35528.0</td>
      <td>5214766.0</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 7 columns</p>
</div>
```
:::
:::

::: {#6b7582ca-043c-468f-9dda-99c3a97b41d2 .cell .markdown}
Ahora tenemos los valores para cada dia y ademas la sumatoria total para
cada dia individual
:::

::: {#bcd09647-f252-4594-9931-325ef323ae8b .cell .markdown}

------------------------------------------------------------------------
:::

::: {#39e78f28-2679-445c-849b-d2d1aa333a17 .cell .markdown}
# Medidas de dispercion

A continuacion ejemplificare algunas medidasd de dispercion
:::

::: {#ed229b3f-0c21-4c7e-9cb3-eceb5edb90b4 .cell .markdown tags="[]"}
`<a id="covarianza">`{=html}`</a>`{=html}

## Covarianza para muertes / casos {#covarianza-para-muertes--casos}
:::

::: {#191b6309-7d78-4a19-904c-10cf4aefd77a .cell .code execution_count="25" tags="[]"}
``` python
np.cov([covid_df['new_deaths'], covid_df['total_deaths']])
```

::: {.output .execute_result execution_count="25"}
    array([[ 5.15370910e+04, -3.18361685e+05],
           [-3.18361685e+05,  2.39397439e+08]])
:::
:::

::: {#0e87b891-057c-47d9-b396-fd07ac3d5401 .cell .code execution_count="26" tags="[]"}
``` python
np.cov([covid_df['new_cases'], covid_df['total_cases']])
```

::: {.output .execute_result execution_count="26"}
    array([[ 2.41281221e+06, -9.30843832e+06],
           [-9.30843832e+06,  1.16989474e+10]])
:::
:::

::: {#096eb7ba-ba6b-43f0-a5ff-7ca501f765ca .cell .markdown tags="[]"}
`<a id="correlacion">`{=html}`</a>`{=html}

## Correlacion para muertes / casos {#correlacion-para-muertes--casos}
:::

::: {#d5867647-37d1-499e-8892-f93b18ac0691 .cell .code execution_count="27" tags="[]"}
``` python
np.corrcoef([covid_df['new_deaths'], covid_df['total_deaths']])
```

::: {.output .execute_result execution_count="27"}
    array([[ 1.        , -0.09063608],
           [-0.09063608,  1.        ]])
:::
:::

::: {#6683c4cd-7fb2-4cb0-935a-c11324607007 .cell .code execution_count="28" tags="[]"}
``` python
np.corrcoef([covid_df['new_cases'], covid_df['total_cases']])
```

::: {.output .execute_result execution_count="28"}
    array([[ 1.        , -0.05540407],
           [-0.05540407,  1.        ]])
:::
:::

::: {#5fbec104-25bb-4d8a-86c2-b920ce74167a .cell .markdown}
## Desviacion standar
:::

::: {#3d1a8eb4-7c75-4e96-a15e-b2b64811860a .cell .code execution_count="29" tags="[]"}
``` python
std = covid_df['total_deaths'].std()
print('La Standard deviation es: ', std)
```

::: {.output .stream .stdout}
    La Standard deviation es:  15472.473605111934
:::
:::

::: {#842d9062-65b8-498d-915f-34da29a0919e .cell .markdown}
## Coeficiente de Variabilidad
:::

::: {#cd6a4c96-b9a9-4f5d-a03b-897c0e6dc2a7 .cell .code execution_count="30" tags="[]"}
``` python
CoefVariab = (std/covid_df['total_deaths'].mean())*100
print('El coeficiente de variabilidad es de: ', round(CoefVariab,2),'%')
```

::: {.output .stream .stdout}
    El coeficiente de variabilidad es de:  76.49 %
:::
:::

::: {#32edbf77-78cf-4a27-8980-4169518de754 .cell .markdown}
## Coeficiciente de variacion
:::

::: {#812c8eeb-ec08-4d09-9887-fe320f82393c .cell .code execution_count="31" tags="[]"}
``` python
import scipy
scipy.stats.variation(covid_df['total_deaths'])
```

::: {.output .execute_result execution_count="31"}
    0.7633558140651867
:::
:::

::: {#f01aaf88-69a1-4bd4-bb6a-65f63560ade7 .cell .markdown}
## Rango interquartil
:::

::: {#6b322c86-d20d-4a6d-8d33-8854b49d014f .cell .code execution_count="32" tags="[]"}
``` python
scipy.stats.iqr(covid_df['total_deaths'])
```

::: {.output .execute_result execution_count="32"}
    34819.25
:::
:::

::: {#53e3fdfa-75e9-43fe-acdd-3a5afdcb7e0e .cell .markdown}
## Error Estandar
:::

::: {#b7ed2b82-32c1-443e-8850-7dbabafe6a61 .cell .code execution_count="33" tags="[]"}
``` python
scipy.stats.sem(covid_df['total_deaths'])
```

::: {.output .execute_result execution_count="33"}
    982.5030564281554
:::
:::

::: {#e9a692ca-1f89-493c-87c0-a99c896c9147 .cell .markdown}

------------------------------------------------------------------------
:::

::: {#6b904c2e-1aa2-44ac-86a7-0a838819d1f4 .cell .markdown}
### Podemos convertir cada año/mes/dia en su propia columna
:::

::: {#e38827e0-ed3f-4a60-b209-818747bdb8ce .cell .code execution_count="34" tags="[]"}
``` python
covid_df.dtypes
```

::: {.output .execute_result execution_count="34"}
    date             object
    new_cases       float64
    new_deaths      float64
    new_tests       float64
    total_cases     float64
    total_deaths    float64
    total_tests     float64
    dtype: object
:::
:::

::: {#3b90f6eb-b662-4cb5-9e0a-9827e35fae58 .cell .markdown}
Transformamos la columna \'date\' de **objeto** a **datetime**
:::

::: {#268155b4-1343-4f70-8206-b3a607bb05a8 .cell .code execution_count="35" tags="[]"}
``` python
covid_df['date'] = pd.to_datetime(covid_df.date)
```
:::

::: {#b812693d-03d0-4965-b852-8a31cd5c5993 .cell .code execution_count="36" tags="[]"}
``` python
covid_df.date
```

::: {.output .execute_result execution_count="36"}
    0     2019-12-31
    1     2020-01-01
    2     2020-01-02
    3     2020-01-03
    4     2020-01-04
             ...    
    243   2020-08-30
    244   2020-08-31
    245   2020-09-01
    246   2020-09-02
    247   2020-09-03
    Name: date, Length: 248, dtype: datetime64[ns]
:::
:::

::: {#dd9b7798-d73f-4657-9265-7629c382e893 .cell .code execution_count="37" tags="[]"}
``` python
covid_df['year'] = pd.DatetimeIndex(covid_df.date).year
covid_df['month'] = pd.DatetimeIndex(covid_df.date).month
covid_df['day'] = pd.DatetimeIndex(covid_df.date).day
covid_df['weekday'] = pd.DatetimeIndex(covid_df.date).weekday
```
:::

::: {#67f81a04-75e9-4518-8ee1-e8d8fa210415 .cell .code execution_count="38" tags="[]"}
``` python
covid_df.head(5)
```

::: {.output .execute_result execution_count="38"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
      <th>total_cases</th>
      <th>total_deaths</th>
      <th>total_tests</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-12-31</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2019</td>
      <td>12</td>
      <td>31</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-02</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-03</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-04</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#6e7249b1-b2bb-4b7f-89a3-972507ddf7ba .cell .markdown}
### Podemos filtrar por mes de esta manera
:::

::: {#f88cbab8-380f-40bc-8d45-6d65e875581f .cell .code execution_count="39" tags="[]"}
``` python
covid_df.set_index('date', inplace=True)
```
:::

::: {#b7d32a74-babe-4e05-8031-85a5f0d70924 .cell .code execution_count="40" tags="[]"}
``` python
covid_abril = covid_df[covid_df.month == 4]
```
:::

::: {#d2b17485-89a0-4dc9-b61c-9d5e6c26297b .cell .code execution_count="41" tags="[]"}
``` python
covid_abril.head(5)
```

::: {.output .execute_result execution_count="41"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
      <th>total_cases</th>
      <th>total_deaths</th>
      <th>total_tests</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-04-01</th>
      <td>4053.0</td>
      <td>839.0</td>
      <td>0.0</td>
      <td>105792.0</td>
      <td>12430.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2020-04-02</th>
      <td>4782.0</td>
      <td>727.0</td>
      <td>0.0</td>
      <td>110574.0</td>
      <td>13157.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-04-03</th>
      <td>4668.0</td>
      <td>760.0</td>
      <td>0.0</td>
      <td>115242.0</td>
      <td>13917.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2020-04-04</th>
      <td>4585.0</td>
      <td>764.0</td>
      <td>0.0</td>
      <td>119827.0</td>
      <td>14681.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2020-04-05</th>
      <td>4805.0</td>
      <td>681.0</td>
      <td>0.0</td>
      <td>124632.0</td>
      <td>15362.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#5b49e2d6-1017-49bd-88b1-ffa932d7105a .cell .markdown}
## ¿Cuales son los valores totales para el mes de abril? ¿Cual es la media de muertes en el mes de abril?

`<a id="section_grafico">`{=html}`</a>`{=html}
:::

::: {#d3a724f7-2139-4dd1-882f-5c6600939de3 .cell .code execution_count="42" tags="[]"}
``` python
metricas_abril = covid_abril[['new_cases', 'new_deaths', 'new_tests']]
```
:::

::: {#91b2cdc2-09d8-4f3d-baf3-2b16efe872c2 .cell .code execution_count="43" tags="[]"}
``` python
metricas_abril.sum()
```

::: {.output .execute_result execution_count="43"}
    new_cases     101852.0
    new_deaths     16091.0
    new_tests     419591.0
    dtype: float64
:::
:::

::: {#490daf92-8df9-430b-9077-bd7aed00e87a .cell .code execution_count="44" tags="[]"}
``` python
metricas_abril['new_deaths'].mean()
```

::: {.output .execute_result execution_count="44"}
    536.3666666666667
:::
:::

::: {#39cc3388-dc31-4de1-964d-80b3d9d32cb9 .cell .markdown}
La media de fallecidos fue de 536 en el mes de abril
:::

::: {#c69c24d5-789e-4274-b45c-de0a9906db4f .cell .markdown}
### casos en abril
:::

::: {#c848fc05-51af-4667-a1bb-18cc999ef088 .cell .code execution_count="45" tags="[]"}
``` python
plt.figure(figsize=(13,5))
plt.plot(metricas_abril['new_cases'])
plt.title('Casos a lo largo de Abril')
plt.ylabel('Total de nuevos casos')
plt.xlabel('fecha');
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/0bf74dfd32733a2fe44a8efbac4e611c81235979.png)
:::
:::

::: {#68ac0f8d-c315-4f22-b641-dd1df168fef3 .cell .markdown}
Bastante variabilidad con subidas y bajas a medida del tiempo pero
descendiendo
:::

::: {#8f706d68-1b99-4921-958e-1439999a4597 .cell .markdown}
### Muertes en abril
:::

::: {#47ded310-202a-4a0d-9c89-f75700317256 .cell .code execution_count="46" tags="[]"}
``` python
plt.figure(figsize=(13,5))
plt.plot(metricas_abril['new_deaths'])
plt.title("Muertes a lo largo de abril")
plt.ylabel('Total de nuevas muertes')
plt.xlabel("fecha");
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/2bf35f330cad4eea7b49f1ab14b1587763548c7b.png)
:::
:::

::: {#36837c6e-6551-4c8e-8868-a0df798653ac .cell .markdown}
De igual manera decendiendo al paso del tiempo
:::

::: {#f838001a-6d60-4282-9d75-764b57c6f31b .cell .markdown}
### Pruebas en abril
:::

::: {#0a5d914f-258c-4ba2-b631-5ced1a3aa09a .cell .code execution_count="47" tags="[]"}
``` python
plt.figure(figsize=(13,5))
plt.plot(metricas_abril['new_tests'])
plt.title('Pruebas a lo largo de abril')
plt.ylabel('Total de pruebas')
plt.xlabel("fecha");
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/4da85def4b656600a5db4a06e184e67b9e403005.png)
:::
:::

::: {#1b8d5d17-9029-40ca-b100-0e0395ada77f .cell .markdown}
Se inicia sin pruebas algunas (0) a partir del dia 18 en adelante del
mes de abril empiezan las pruebas de covid
:::

::: {#e45ace26-c565-4088-b89b-0aaaaa979eac .cell .markdown}
### De esta manera podemos tener todas las filas que fueron del mes \"abril\" por ejemplo
:::

::: {#df074905-dd3f-47dd-a8df-7171789fbfca .cell .markdown}

------------------------------------------------------------------------
:::

::: {#34f244f1-8aa0-4c3c-a295-a2b7bdaa8431 .cell .markdown}
## ¿Como es la distribucion total para muertes y casos?
:::

::: {#e122c231-1dd2-41f8-bdd8-44b1c591389e .cell .code execution_count="48" tags="[]"}
``` python
plt.figure(figsize=(15,5))
plt.plot(covid_df.new_deaths)
plt.plot(covid_df.new_cases)
plt.legend(['Muertes','Casos']);
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/710db6b1f8554cf29511b2aad25bf859c6806ef9.png)
:::
:::

::: {#3134a835-5bb0-438f-ac5c-9b74d2c063c8 .cell .markdown}
## ¿Cuantos casos totales hubo registrados en nuestro dataset?
:::

::: {#1a6a4879-d6fe-4302-b9aa-ccc1c963fde7 .cell .code execution_count="49" tags="[]"}
``` python
total_cases = covid_df.new_cases.sum()
print("hubo", total_cases, "casos totales")
```

::: {.output .stream .stdout}
    hubo 271960.5 casos totales
:::
:::

::: {#934aab37-81ca-43ba-84b1-3a5fe748d060 .cell .markdown}
# ¿Como son los valores totales con respecto a cada mes?
:::

::: {#c103e7c7-2cba-47c6-b081-ba0e1e18832c .cell .code execution_count="50" tags="[]"}
``` python
covid_month_df = covid_df.groupby('month')[['new_cases', 'new_deaths', 'new_tests']].sum()
```
:::

::: {#3db7297b-c803-4ab8-9609-925d8af26175 .cell .markdown tags="[]"}
### Relacion de valores con respecto al mes
:::

::: {#b3afb3f2-0ad6-405f-abf5-a8e61bb8d8a7 .cell .code execution_count="51" tags="[]"}
``` python
plt.bar(covid_month_df.index, covid_month_df.new_cases, color='y');
plt.bar(covid_month_df.index, covid_month_df.new_deaths, color='r', alpha=0.7);
plt.title("Comparacion Casos/Muertes durante el año")
plt.ylabel('Cantidad de casos nuevos');
plt.xlabel('Numero de mes');
plt.legend(['Nuevos casos', 'Muertes']);
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/24e91976fa64dd608a4fb2742d04f2f8af9383d0.png)
:::
:::

::: {#89cae0ba-01e3-48e5-8fe8-e3dd7d517db5 .cell .markdown}
## ¿Cuantas pruebas fueros realizadas durante cada mes?
:::

::: {#aef66214-c4e9-4242-a9f2-07a3be14381d .cell .code execution_count="52" tags="[]"}
``` python
Pruebas_totales = covid_month_df.new_tests
```
:::

::: {#1fc81ad7-ae1d-469a-a25a-5564f9f855ea .cell .code execution_count="53" tags="[]"}
``` python
Pruebas_totales.sum()
```

::: {.output .execute_result execution_count="53"}
    4279456.0
:::
:::

::: {#6e13e5ab-e6a1-486b-9414-828de77eb461 .cell .code execution_count="54" tags="[]"}
``` python
plt.figure(figsize=(10,5))
plt.plot(Pruebas_totales);
plt.title("Pruebas realizadas para cada mes")
plt.xlabel('Mes')
plt.ylabel('Cantidad de pruebas')
plt.legend(['Pruebas totales (en millones)']);
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/7f2ae084dadcc9bd2ae559a824269b7417fd8296.png)
:::
:::

::: {#b8170ea9-10ec-44a2-927c-92900e4ea8c0 .cell .markdown}
## Mapa de correlacion (Normalizacion)

`<a id="corr">`{=html}`</a>`{=html}
:::

::: {#22f09d33-ce45-4b16-bb66-afaf7a8d4534 .cell .code execution_count="55" tags="[]"}
``` python
from sklearn.preprocessing import StandardScaler
escala = StandardScaler()
escalado = escala.fit_transform(covid_df[['new_cases',
       'new_deaths', 'new_tests',
       'total_deaths', 'total_tests', 'total_cases']])
escalado.T
```

::: {.output .execute_result execution_count="55"}
    array([[-0.70740765, -0.70740765, -0.70740765, ..., -0.06490497,
            -0.07845172,  0.14797242],
           [-0.63231938, -0.63231938, -0.63231938, ..., -0.60583629,
            -0.59700859, -0.60583629],
           [-0.96129647, -0.96129647, -0.96129647, ...,  2.06896039,
            -0.96129647, -0.96129647],
           [-1.31000509, -1.31000509, -1.31000509, ...,  0.9899386 ,
             0.9904567 ,  0.99084527],
           [-0.87049461, -0.87049461, -0.87049461, ...,  2.21605564,
             2.21605564,  2.21605564],
           [-1.35328674, -1.35328674, -1.35328674, ...,  1.14487006,
             1.15390257,  1.16618679]])
:::
:::

::: {#d9990319-46b9-4960-9d56-4b547c2dc4f3 .cell .code execution_count="56" tags="[]"}
``` python
matriz_cov = np.cov(escalado.T)
matriz_cov
```

::: {.output .execute_result execution_count="56"}
    array([[ 1.00404858,  0.94044473, -0.18839191, -0.11437431, -0.32139589,
            -0.05562838],
           [ 0.94044473,  1.00404858, -0.18218394, -0.09100302, -0.41546645,
            -0.03842987],
           [-0.18839191, -0.18218394,  1.00404858,  0.80715672,  0.66186581,
             0.80336681],
           [-0.11437431, -0.09100302,  0.80715672,  1.00404858,  0.8154806 ,
             1.00177842],
           [-0.32139589, -0.41546645,  0.66186581,  0.8154806 ,  1.00404858,
             0.80905304],
           [-0.05562838, -0.03842987,  0.80336681,  1.00177842,  0.80905304,
             1.00404858]])
:::
:::

::: {#5b07185d-d8ff-4294-ac2b-1ddb827e1492 .cell .code execution_count="57" tags="[]"}
``` python
plt.figure(figsize=(9,9))
mapa_calor = sns.heatmap(matriz_cov, fmt='.2f', square=True, cbar=True, annot=True , xticklabels= ['new_cases',
       'new_deaths', 'new_tests',
       'total_deaths', 'total_tests', 'total_cases'], yticklabels=['new_cases',
       'new_deaths', 'new_tests',
       'total_deaths', 'total_tests', 'total_cases'] )
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/4f7add144e0bde2d57e3c1e2ba18a5360711d062.png)
:::
:::

::: {#bbfddf49-d0d6-4c3d-baeb-2d842cb97343 .cell .markdown}
Segun el matriz claramente queda visto que las muertes totales son
perfectamente correlativos a los casos totales
:::

::: {#4a47ba45-403e-4c14-b003-bdca3ead7824 .cell .markdown}
## Regresion lineal

`<a id="lineal">`{=html}`</a>`{=html}
:::

::: {#3eaad449-56d5-43aa-b572-017ee4889926 .cell .code execution_count="58"}
``` python
sns.regplot(data = covid_df, x= 'month', y= 'total_deaths', fit_reg=True, color='g',label= 'Fallecidos',marker='o')
plt.xlabel('Mes')
plt.ylabel('Muertes')
plt.title('Regresion de muertes totales')
plt.grid()
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/549de3a33b3ed99a3627b526469146786485a4ab.png)
:::
:::

::: {#5f1e0631-1183-42ab-97fa-65a3a961830d .cell .markdown}
Intento de aplicacion de regresion lineal
:::

::: {#0db02292-3a02-46a5-8f00-cd1f402f7b1f .cell .markdown}
### ¿Cual fue el mes con mas muertes? / ¿Cual fue el mes con mas casos? {#cual-fue-el-mes-con-mas-muertes--cual-fue-el-mes-con-mas-casos}
:::

::: {#2e26db0e-13fe-4ef7-b9f1-b32937d679e8 .cell .code execution_count="59" tags="[]"}
``` python
plt.scatter(covid_month_df['new_deaths'], covid_month_df.index);
plt.title('Muestra Fallecidos  por mes')
plt.xlabel('Muertes totales')
plt.ylabel('Numero de mes');
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/a3fac6c011bcdd7295df0d249bd2830e03b87fcb.png)
:::
:::

::: {#aaf11f79-f40d-4450-b9cd-5ddaae36bfab .cell .markdown tags="[]"}
En el mes de **abril** fueron registrados los mayores conteos de
fallecidos en Italia
:::

::: {#ce782aef-24c7-4f72-bd1d-a185f30c3606 .cell .code execution_count="60" tags="[]"}
``` python
plt.scatter(covid_month_df['new_cases'], covid_month_df.index);
plt.title('Muestra de valores altos por mes')
plt.xlabel('Casos totales')
plt.ylabel('Numero de mes');
```

::: {.output .display_data}
![](vertopal_d7e57302d5274c1d81d3f796ca0bacf0/4177d3ae5ed8d173462aa4be7f8d76dd7899bb36.png)
:::
:::

::: {#ac407911-ff24-4be4-9a72-6dad8276a53d .cell .markdown}
De manera similar **Abril** fue el mes de mas casos registrados en
Italia, seguido muy cerca del mes de **Marzo**
:::

::: {#6088b350-13ce-4c57-9b57-d00165d36af4 .cell .markdown}
# Combinando tablas

`<a id="merge">`{=html}`</a>`{=html}
:::

::: {#1404136e-f8c6-40fa-a376-96a704ab85d6 .cell .markdown}
### Podemos combinar los dataframes usando merge
:::

::: {#058928e1-c61b-4ac1-b1e6-7117634b7b19 .cell .code execution_count="61" tags="[]"}
``` python
urlretrieve('https://gist.githubusercontent.com/aakashns/8684589ef4f266116cdce023377fc9c8/raw/99ce3826b2a9d1e6d0bde7e9e559fc8b6e9ac88b/locations.csv', 
            'locations.csv')
```

::: {.output .execute_result execution_count="61"}
    ('locations.csv', <http.client.HTTPMessage at 0x1a4aac42b90>)
:::
:::

::: {#cbe40e7c-e1d8-4064-99f4-ebb5d569e903 .cell .code execution_count="62" tags="[]"}
``` python
poblacion_df = pd.read_csv('locations.csv')
```
:::

::: {#85ed6893-038d-4055-a629-169d5e6bad7e .cell .code execution_count="63" tags="[]"}
``` python
poblacion_df # nueva tabla
```

::: {.output .execute_result execution_count="63"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>continent</th>
      <th>population</th>
      <th>life_expectancy</th>
      <th>hospital_beds_per_thousand</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>3.892834e+07</td>
      <td>64.83</td>
      <td>0.500</td>
      <td>1803.987</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>2.877800e+06</td>
      <td>78.57</td>
      <td>2.890</td>
      <td>11803.431</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Africa</td>
      <td>4.385104e+07</td>
      <td>76.88</td>
      <td>1.900</td>
      <td>13913.839</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>7.726500e+04</td>
      <td>83.73</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>3.286627e+07</td>
      <td>61.15</td>
      <td>NaN</td>
      <td>5819.495</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>207</th>
      <td>Yemen</td>
      <td>Asia</td>
      <td>2.982597e+07</td>
      <td>66.12</td>
      <td>0.700</td>
      <td>1479.147</td>
    </tr>
    <tr>
      <th>208</th>
      <td>Zambia</td>
      <td>Africa</td>
      <td>1.838396e+07</td>
      <td>63.89</td>
      <td>2.000</td>
      <td>3689.251</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1.486293e+07</td>
      <td>61.49</td>
      <td>1.700</td>
      <td>1899.775</td>
    </tr>
    <tr>
      <th>210</th>
      <td>World</td>
      <td>NaN</td>
      <td>7.794799e+09</td>
      <td>72.58</td>
      <td>2.705</td>
      <td>15469.207</td>
    </tr>
    <tr>
      <th>211</th>
      <td>International</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>212 rows × 6 columns</p>
</div>
```
:::
:::

::: {#9d3fedf5-7d72-4755-b96b-72cbbef892c1 .cell .markdown}
-   location: El pais en efecto
-   continent: Continente
-   population: Cantidad de poblacion total
-   life expentancy: El valor calculado de la esperanza de vida
-   hospital_beds_per_thousand: Cuantas camas de hospital hay cada 1000
    habitantes
-   gdp_per_capita: basicamente el Pib (El Producto Interno Bruto)
:::

::: {#48e41b2f-b493-43c8-953d-c8e7fc4f2d3b .cell .code execution_count="64" tags="[]"}
``` python
poblacion_df[poblacion_df.location == "Italy"] ## seleciono solamente para italia
```

::: {.output .execute_result execution_count="64"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>continent</th>
      <th>population</th>
      <th>life_expectancy</th>
      <th>hospital_beds_per_thousand</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#30ea2724-34dc-43ef-9415-152729725ae8 .cell .markdown}
Selecciono solamente Italia ya que es el analizado en en anterior
dataset
:::

::: {#2aed65e3-baba-46e5-a530-05d4f5c34fbf .cell .code execution_count="65" tags="[]"}
``` python
covid_df['location'] = "Italy"
```
:::

::: {#702a42ce-3316-477a-8670-17ccf9fd97e0 .cell .code execution_count="66" tags="[]"}
``` python
covid_df
```

::: {.output .execute_result execution_count="66"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
      <th>total_cases</th>
      <th>total_deaths</th>
      <th>total_tests</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>location</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-12-31</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2019</td>
      <td>12</td>
      <td>31</td>
      <td>1</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-01-04</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-08-30</th>
      <td>1444.0</td>
      <td>1.0</td>
      <td>53541.0</td>
      <td>267298.5</td>
      <td>35504.0</td>
      <td>5117788.0</td>
      <td>2020</td>
      <td>8</td>
      <td>30</td>
      <td>6</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-08-31</th>
      <td>1365.0</td>
      <td>4.0</td>
      <td>42583.0</td>
      <td>268663.5</td>
      <td>35508.0</td>
      <td>5160371.0</td>
      <td>2020</td>
      <td>8</td>
      <td>31</td>
      <td>0</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-09-01</th>
      <td>996.0</td>
      <td>6.0</td>
      <td>54395.0</td>
      <td>269659.5</td>
      <td>35514.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-09-02</th>
      <td>975.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>270634.5</td>
      <td>35522.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>2020-09-03</th>
      <td>1326.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>271960.5</td>
      <td>35528.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>3</td>
      <td>3</td>
      <td>Italy</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 11 columns</p>
</div>
```
:::
:::

::: {#e86a6d71-b853-4707-b338-ed987b16d846 .cell .code execution_count="67" tags="[]"}
``` python
merged_df = covid_df.merge(poblacion_df, on="location") ## uso merge para fusinar las tablas con localidad 
## italia
```
:::

::: {#7a8556a0-54d1-4524-8ec3-dfc9fe55141e .cell .code execution_count="68" tags="[]"}
``` python
merged_df
```

::: {.output .execute_result execution_count="68"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_cases</th>
      <th>new_deaths</th>
      <th>new_tests</th>
      <th>total_cases</th>
      <th>total_deaths</th>
      <th>total_tests</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>location</th>
      <th>continent</th>
      <th>population</th>
      <th>life_expectancy</th>
      <th>hospital_beds_per_thousand</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2019</td>
      <td>12</td>
      <td>31</td>
      <td>1</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>935310.0</td>
      <td>2020</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>243</th>
      <td>1444.0</td>
      <td>1.0</td>
      <td>53541.0</td>
      <td>267298.5</td>
      <td>35504.0</td>
      <td>5117788.0</td>
      <td>2020</td>
      <td>8</td>
      <td>30</td>
      <td>6</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>244</th>
      <td>1365.0</td>
      <td>4.0</td>
      <td>42583.0</td>
      <td>268663.5</td>
      <td>35508.0</td>
      <td>5160371.0</td>
      <td>2020</td>
      <td>8</td>
      <td>31</td>
      <td>0</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>245</th>
      <td>996.0</td>
      <td>6.0</td>
      <td>54395.0</td>
      <td>269659.5</td>
      <td>35514.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>246</th>
      <td>975.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>270634.5</td>
      <td>35522.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
    <tr>
      <th>247</th>
      <td>1326.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>271960.5</td>
      <td>35528.0</td>
      <td>5214766.0</td>
      <td>2020</td>
      <td>9</td>
      <td>3</td>
      <td>3</td>
      <td>Italy</td>
      <td>Europe</td>
      <td>60461828.0</td>
      <td>83.51</td>
      <td>3.18</td>
      <td>35220.084</td>
    </tr>
  </tbody>
</table>
<p>248 rows × 16 columns</p>
</div>
```
:::
:::

::: {#9d3b6eaf-e227-4c09-8014-c11cba45844b .cell .markdown}
### ¿Cuantos casos por millon de poblacion hay?
:::

::: {#db4b22b0-a2e9-4846-9082-1b09916c3f10 .cell .code execution_count="69" tags="[]"}
``` python
merged_df['cases_per_million'] = merged_df.total_cases * 1e6 / merged_df.population
merged_df['deaths_per_million'] = merged_df.total_deaths * 1e6 / merged_df.population
merged_df['tests_per_million'] = merged_df.total_tests * 1e6 / merged_df.population
```
:::

::: {#b9ae3019-5658-4bcb-9cc1-ac01e2b017df .cell .code execution_count="70" tags="[]"}
``` python
merged_df[['cases_per_million','deaths_per_million','tests_per_million']].describe()
```

::: {.output .execute_result execution_count="70"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cases_per_million</th>
      <th>deaths_per_million</th>
      <th>tests_per_million</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>248.000000</td>
      <td>248.000000</td>
      <td>248.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2416.042578</td>
      <td>334.560062</td>
      <td>35431.244828</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1788.924950</td>
      <td>255.904827</td>
      <td>22977.951270</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15469.429737</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25.615335</td>
      <td>0.554069</td>
      <td>15469.429737</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3446.438966</td>
      <td>470.925226</td>
      <td>23872.706925</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3993.631321</td>
      <td>576.442214</td>
      <td>55495.568212</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4498.052887</td>
      <td>587.610418</td>
      <td>86248.897403</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#5fe627b1-70c4-461a-8f70-d6d2c7d7e870 .cell .markdown}
# Resumen y concluciones

`<a id="concluciones">`{=html}`</a>`{=html}
:::

::: {#7b4d8c68-8ee5-4f28-a9fa-7085eab5f836 .cell .markdown tags="[]"}
-   El dataset analizado fue sobre la aparicion del covid-19 en Italia
    en 2020
-   Se completaron valores faltantes y se realizaron pequeñas
    concluciones sobre cada una (ver [indice](#index) para mas info)
-   Como valores fueron mayormente registrados en el año 2020, no existe
    la posibilidad de hacer un analisis mas alla de la fecha tomada
-   Aparentemente entre el mes de Marzo y Abril es donde existieron la
    mayor cantidad de valores registrados para Italia
-   4279456 pruebas de covid-19 realizadas, 271960 Casos de covid
    detectados mediante las pruebas y 35528 fallecidos a causa de la
    enfermedad
-   Los primeros dos meses y los ultimos dos meses del año registran muy
    pocos o ningun caso de covid
:::
