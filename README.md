CLANews
=======

*CLANews* es un clasificador de titulares de noticias extra�dos de Twitter basado
en una M�quina de Soporte Vectorial - Support Vector Machine.

El proyecto fue desarrollado en Java, utilizando el IDE Netbeans y est� 
compuesto por cuatro sub-proyectos:

* **TweetsGetter**: utilizado para obtener tweets de la cuenta de Twitter de 
usuarios especificados en un archivo de entrada.
* **TweetsAnalyzer**: es un analizador de sentencias utilizado para pre-procesar los
tweets extra�dos antes de utilizarlos para construir el clasificador.
* **Classifier**: es la m�quina utilizada para clasificar los tweets en categor�as,
por ejemplo: Tecnolog�a, Econom�a, Deporte, Pol�tica y Entretenimiento.

Para hacer uso de cualquiera de los sub-proyectos, es necesario, antes que nada,
generar el archivo ```.jar``` correspondiente.

A continuaci�n se detallar� c�mo utilizar cada uno de estos sub-proyectos:

## TweetsGetter

La sintaxis para su ejecuci�n es la siguiente:

```
java -cp dist/CLANews.jar clanews.tweetsgetter.TweetsGetter -l <EN|ES> -m <TRAIN|TEST> -n <number of tweets> [-i <input file>]
```

donde:

* ```-l``` es utilizado para especificar el idioma en que est�n los tweets. 
Actualmente, ```EN``` indica Ingl�s y ```ES``` espa�ol. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar el modo en que ser� ejecutado 
*TweetsGetter*. Estos modos pueden ser ```TRAIN``` si se quiere extraer datos 
para ser utilizados en la construcci�n del clasificador o ```TEST``` si se 
quiere extraer datos para validar el clasificador construido.
Para ambos modos, se crea por defecto un directorio ```News``` donde se 
almacenan los datos solicitados.
* ```-n``` es utilizado para especificar el n�mero de tweets que se quiere 
almacenar. Este n�mero debe estar entre 100 y 5000.
Si es indicado el modo ```TRAIN``` entonces se recoletar�n el n�mero de tweets 
indicado por cada categor�a especificada en el archivo de entrada. 
Si es indicado el modo ```TEST``` entonces, a diferencia del caso anterior, se 
recolectar� el n�mero de tweets indicado por cada usuario especificado.
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
En el caso de estar en modo ```TEST``` y no indicarse este flag, se tomar� el 
archivo en ```src/resources/tweetsgetter/test_input.txt``` y su estructura es 
la siguiente:
```
<directorio>
<usuario 1>
<usuario 2>
...
<usuario n>
```

**Nota**: para acceder al API de Twitter se utiliz� la libreria *Twitter4J* por
lo que es requerido un archivo ```twitter4j.properties``` para especificar
las credenciales y los permisos necesarios. Para m�s informaci�n puede consultar
el [sitio web de la libreria](http://twitter4j.org/en/configuration.html).

**Otra Nota**: existe un limite m�ximo de tweets que se pueden recolectar por tiempo,
por ello, se establecio 5000 como l�mite m�ximo para el flag ```-n```. Para m�s
informaci�n, puede consultar el 
[sitio web para desarrolladores de Twitter](https://dev.twitter.com/).

## TweetsAnalyzer

La sintaxis para su ejecuci�n es la siguiente:

```
java -cp dist/CLANews.jar clanews.tweetsanalyzer.TweetsAnalyzer -l <EN|ES> -m <message>
```

donde:

* ```-l``` es utilizado para especificar el idioma en que est�n los tweets. 
Actualmente, ```EN``` indica Ingl�s y ```ES``` espa�ol. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar el mensaje o la sentencia a analizar.
Si consta de m�s de una palabra, entonces el mensaje debe ser encerrado en
comillas simples.

Este sub-proyecto utiliza la librer�a *Lucene* para llevar a cabo sus funciones.
Cualquier informaci�n puede ser consultada en su 
[sitio web](http://lucene.apache.org/).

De *Lucene* utiliza el analizador _EnglishAnalyzer_ cuando se espec�fica 
```EN``` con el flag ```-l``` o _SpanishAnalyzer_ cuando se espec�fica ```ES```.

Adicionalmente, se utilizan expresiones regulares para eliminar los n�meros
y las urls en las sentencias.

## Classifier

La sintaxis para su ejecuci�n es la siguiente:

```
java -cp dist/CLANews.jar clanews.classifier.CLANews -l <EN|ES> -m <TRAIN|TEST> [-i <test file> (if set TEST mode)]
```

donde:

* ```-l``` es utilizado para especificar el idioma en que est�n los tweets. 
Actualmente, ```EN``` indica Ingl�s y ```ES``` espa�ol. Este flag es utilizado 
para crear y utilizar una instancia del analizador *TweetsAnalyzer* adecuada.
* ```-m``` es utilizado para especificar si se desea construir un nuevo
clasificador o si se desea validar un clasificador construido previamente. 
El modo ```TRAIN``` utiliza los tweets almacenados en el directorio 
```CLANews/News/Processed```, separando por categor�a seg�n los sub-directorios
especificados aqu�, para construir un nuevo clasificador, siempre y cuando no
exista el archivo ```CLANews/src/resources/classifier/all_tweets.arff``` que no
es m�s que los tweets en ```CLANews/News/Processed``` pero en formato _ARFF_.
Por otro lado, el modo ```TEST``` toma el clasificador construido previamente
y los tweets que se encuentran en el archivo <test file> indicado con el flag 
```-i``` para clasificarlos en sus respectivas categorias. Si el clasificador 
no ha sido construido a�n, entonces se construir� uno. Si se omite el flag 
```-i``` entonces se tomar�n los tweets que se encuentran en el archivo 
```CLANews/News/Test/tweets.txt``` para clasificarlos.
* ```-i``` es un flag opcional y es utilizado para especificar el archivo de
entrada para la validaci�n de un clasificador. El formato de este archivo debe 
ser simplemente un tweet por linea.

Para la construcci�n del clasificador se utilizan las librerias *Weka* y
*LibSVM*, especificamente weka 3.6.11 y LibSVM 3.20. Cualquier informaci�n
adicional puede ser consultada en sus sitios: 

* [weka](http://www.cs.waikato.ac.nz/ml/weka/)
* [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/)

