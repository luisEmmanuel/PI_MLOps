from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    df = pd.read_csv('langpeli.csv')
    d = df.loc[df.language == idioma]
    c = d['num_movies'].to_list()[0]
    
    return{'Idioma:':idioma, 'cantidad de peliculas:':c}

@app.get('/peliculas_duracion/{pelicula}')
def peliculas_duracion(pelicula:str):
    ##Se ingresa una pelicula y devuelve la duracion y el año
    df = pd.read_csv('pelidur.csv')
    d = df.loc[df.title == pelicula]
    du = d['runtime'].to_list()[0]
    y = d['year'].to_list()[0]
    
    return{'Pelicula:':pelicula, 'duracion:':du, 'anio:':y}

@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    ##Se ingresa la franquicia y la funcion retorna la franquicia, cantidad de peliculas, ganancia total y ganancia promedio
    df = pd.read_csv('franq.csv')
    d = df.loc[df.belongs_to_collection == franquicia]
    c = d['count'].to_list()[0]
    m = d['mean'].to_list()[0]
    s = d['sum'].to_list()[0]
    return {'Franquicia':franquicia, 'cantidad':c, 'ganancia_total':s, 'ganancia_promedio':m}


@app.get('/pelicula_pais/{pais}')
def pelicula_pais(pais:str):
    ##Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo
    df = pd.read_csv('paispeli.csv')
    d = df.loc[df.country == pais]
    d = d['num_movies'].to_list()[0]
    return {'Pais':pais, 'cantidad':d}


@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    ##Ingresa la productora, retornando la ganancia total y la cantidad de peliculas que produjeron
    df = pd.read_csv('prod.csv')
    d = df.loc[df.companies == productora]
    c = d['Number'].to_list()[0]
    m = d['Average'].to_list()[0]
    s = d['Total'].to_list()[0]

    return {'Productora':productora, 'Ganancia_total':s, 'Cantidad':c, 'Promedio':m}


@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    ##Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el exito del mismo medio a traves del retorno.
    ##Ademas, debera devolver el nombre de cada pelicula con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.
    df = pd.read_csv('dir.csv')
    df1 = pd.read_csv('dir_pel.csv')
    d = df.loc[df.director == nombre_director]
    r = d['return'].to_list()

    b = df1[(df1['director'] == nombre_director) & (df1['return'].notnull())]
    g = b['title'].to_list()
    a = b['year'].to_list()
    rr = b['return'].to_list()
    bd = b['budget'].to_list()
    rv = b['revenue'].to_list()
    return {'director':nombre_director, 'retorno_total_director':r,
            'peliculas':g, 'anio':a, 'retorno_pelicula':rr,
            'budget_pelicula':bd, 'revenue_pelicula':rv}


@app.get('/recomendacion/{titulo}')
#ML
def recomendacion(titulo):
    ##Ingresas un nombre de pelicula y te recomienda las similares en una lista
    i = pd.read_csv('titulo.csv')
    tfidf = TfidfVectorizer(stop_words="english")
    i["overview"] = i["overview"].fillna("")

    tfidf_matriz = tfidf.fit_transform(i["overview"])
    coseno_sim = linear_kernel(tfidf_matriz, tfidf_matriz)

    indices = pd.Series(i.index, index=i["title"]).drop_duplicates()
    idx = indices[titulo]
    simil = list(enumerate(coseno_sim[idx]))
    simil = sorted(simil, key=lambda x: x[1], reverse=True)
    simil = simil[1:11]
    movie_index = [i[0] for i in simil]

    lista = i["title"].iloc[movie_index].to_list()[:5]

    return {'lista recomedada': lista}
