CLANews
=======

*CLANews* es un clasificador de titulares de noticias extraídos de Twitter basado
en una Máquina de Soporte Vectorial - Support Vector Machine.

El proyecto fue desarrollado en Java, utilizando el IDE Netbeans y está 
compuesto por cuatro sub-proyectos:

* **TweetsGetter**: utilizado para obtener tweets de la cuenta de Twitter de 
usuarios especificados en un archivo de entrada.
* **TweetsAnalyzer**: es un analizador de sentencias utilizado para pre-procesar los
tweets extraídos antes de utilizarlos para construir el clasificador.
* **Classifier**: es la máquina utilizada para clasificar los tweets en categorías,
por ejemplo: Tecnología, Economía, Deporte, Política y Entretenimiento.

Para hacer uso de cualquiera de los sub-proyectos, es necesario, antes que nada,
generar el archivo ```.jar``` correspondiente.

A continuación se detallará cómo utilizar cada uno de estos sub-proyectos:

## TweetsGetter

La sintaxis para su ejecución es la siguiente:

```
java -cp dist/CLANews.jar clanews.tweetsgetter.TweetsGetter -l <EN|ES> -m <TRAIN|TEST> -n <number of tweets> [-i <input file>]
```

donde:

* ```-l``` es utilizado para especificar el idioma en que están los tweets. 
Actualmente, ```EN``` indica Inglés y ```ES``` español. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar el modo en que será ejecutado 
*TweetsGetter*. Estos modos pueden ser ```TRAIN``` si se quiere extraer datos 
para ser utilizados en la construcción del clasificador o ```TEST``` si se 
quiere extraer datos para validar el clasificador construido.
Para ambos modos, se crea por defecto un directorio ```News``` donde se 
almacenan los datos solicitados.
* ```-n``` es utilizado para especificar el número de tweets que se quiere 
almacenar. Este número debe estar entre 100 y 5000.
Si es indicado el modo ```TRAIN``` entonces se recoletarán el número de tweets 
indicado por cada categoría especificada en el archivo de entrada. 
Si es indicado el modo ```TEST``` entonces, a diferencia del caso anterior, se 
recolectará el número de tweets indicado por cada usuario especificado.
* ```-i``` es un flag opcional y es utilizado para especificar el archivo de
entrada para el algoritmo.
En caso de no ser indicado este flag, se toma el archivo por defecto 
```src/resources/tweetsgetter/train_input.txt``` en caso de estar en modo 
```TRAIN```. La estructura de este archivo es la siguiente:
```
<directorio raiz>
<sub-directorio de categoria 1> <usuario 1 de categoria 1> <usuario 2 de categoria 1> ... <usuario n de categoria 1>
<sub-directorio de categoria 2> <usuario 1 de categoria 2> <usuario 2 de categoria 2> ... <usuario n de categoria 2>
...
<sub-directorio de categoria n> <usuario 1 de categoria n> <usuario 2 de categoria n> ... <usuario n de categoria n>
```
En el caso de estar en modo ```TEST``` y no indicarse este flag, se tomará el 
archivo en ```src/resources/tweetsgetter/test_input.txt``` y su estructura es 
la siguiente:
```
<directorio>
<usuario 1>
<usuario 2>
...
<usuario n>
```

**Nota**: para acceder al API de Twitter se utilizó la libreria *Twitter4J* por
lo que es requerido un archivo ```twitter4j.properties``` para especificar
las credenciales y los permisos necesarios. Para más información puede consultar
el [sitio web de la libreria](http://twitter4j.org/en/configuration.html).

**Otra Nota**: existe un limite máximo de tweets que se pueden recolectar por tiempo,
por ello, se establecio 5000 como límite máximo para el flag ```-n```. Para más
información, puede consultar el 
[sitio web para desarrolladores de Twitter](https://dev.twitter.com/).

## TweetsAnalyzer

La sintaxis para su ejecución es la siguiente:

```
java -cp dist/CLANews.jar clanews.tweetsanalyzer.TweetsAnalyzer -l <EN|ES> -m <message>
```

donde:

* ```-l``` es utilizado para especificar el idioma en que están los tweets. 
Actualmente, ```EN``` indica Inglés y ```ES``` español. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar el mensaje o la sentencia a analizar.
Si consta de más de una palabra, entonces el mensaje debe ser encerrado en
comillas simples.

Este sub-proyecto utiliza la librería *Lucene* para llevar a cabo sus funciones.
Cualquier información puede ser consultada en su 
[sitio web](http://lucene.apache.org/).

De *Lucene* utiliza el analizador _EnglishAnalyzer_ cuando se específica 
```EN``` con el flag ```-l``` o _SpanishAnalyzer_ cuando se específica ```ES```.

Adicionalmente, se utilizan expresiones regulares para eliminar los números
y las urls en las sentencias.

## Classifier

La sintaxis para su ejecución es la siguiente:

```
java -cp dist/CLANews.jar clanews.classifier.CLANews -l <EN|ES> -m <TRAIN|TEST> [-i <test file> (if set TEST mode)]
```

donde:

* ```-l``` es utilizado para especificar el idioma en que están los tweets. 
Actualmente, ```EN``` indica Inglés y ```ES``` español. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar si se desea construir un nuevo
clasificador o si se desea validar un clasificador construido previamente. 
El modo ```TRAIN``` utiliza los tweets almacenados en el directorio 
```CLANews/News/Processed```, separando por categoría según los sub-directorios
especificados aquí, para construir un nuevo clasificador, siempre y cuando no
exista el archivo ```CLANews/src/resources/classifier/all_tweets.arff``` que no
es más que los tweets en ```CLANews/News/Processed``` pero en formato _ARFF_.
Por otro lado, el modo ```TEST``` toma el clasificador construido previamente
y los tweets que se encuentran en el archivo <test file> indicado con el flag 
```-i``` para clasificarlos en sus respectivas categorias. Si el clasificador 
no ha sido construido aún, entonces se construirá uno. Si se omite el flag 
```-i``` entonces se tomarán los tweets que se encuentran en el archivo 
```CLANews/News/Test/tweets.txt``` para clasificarlos.
* ```-i``` es un flag opcional y es utilizado para especificar el archivo de
entrada para la validación de un clasificador. El formato de este archivo debe 
ser simplemente un tweet por linea.

Para la construcción del clasificador se utilizan las librerias *Weka* y
*LibSVM*, especificamente weka 3.6.11 y LibSVM 3.20. Cualquier información
adicional puede ser consultada en sus sitios: 

* [weka](http://www.cs.waikato.ac.nz/ml/weka/)
* [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

