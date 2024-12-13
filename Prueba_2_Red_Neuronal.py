
# Necesitaremos únicamente tensorflow y numpy, matplot para el gráfico
import tensorflow as tf  # Importa TensorFlow, una biblioteca de código abierto para aprendizaje automático.
import numpy as np       # Importa NumPy, una biblioteca para trabajar con ARRAYs y matrices numéricas.

# Definir los datos de entrada y salida

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
# Crea un ARRAY de NumPy que contiene temperaturas en grados Celsius.
# np.array crea un ARRAY a partir de la lista proporcionada.
# dtype=float asegura que los elementos del ARRAY sean de tipo flotante (números con decimales).

farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
# Crea un ARRAY de NumPy que contiene las temperaturas correspondientes en grados Fahrenheit.
# Este ARRAY será la salida que el modelo intentará predecir a partir de las entradas en Celsius.

# AHORA VAMOS CON LAS CAPAS:

capa = tf.keras.layers.Dense(units=1, input_shape=[1])  # -> AUTORREGISTRA LA CAPA CON 1 NEURONA
# Define una capa densa (fully connected) de la red neuronal.
# tf.keras.layers.Dense: Crea una capa completamente conectada.
# units=1: Especifica que esta capa tendrá una sola neurona.
# input_shape=[1]: Define la forma de la entrada que recibirá la capa. En este caso, una sola característica o atributo (grados Celsius).

modelo = tf.keras.Sequential([capa])
# Crea un modelo secuencial de Keras.
# tf.keras.Sequential: Indica que el modelo es una secuencia lineal de capas.
# [capa]: Pasa la capa definida anteriormente como la única capa del modelo.
# En modelos más complejos, puedes añadir más capas a esta lista para construir redes neuronales profundas.

# QUIERO INDICARLE COMO QUIERO QUE PROCESE LAS MATEMATICAS PARA APRENDER MEJOR

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
# Compila el modelo, configurando el proceso de aprendizaje.
# optimizer=tf.keras.optimizers.Adam(0.1):
#   - Define el optimizador que actualizará los pesos de la red durante el entrenamiento.
#   - Adam es un optimizador eficiente que adapta las tasas de aprendizaje.
#   - 0.1 es la tasa de aprendizaje (learning rate), que controla qué tan grande son los pasos que da el optimizador al ajustar los pesos.
# loss='mean_squared_error':
#   - Define la función de pérdida (loss function) que el modelo intentará minimizar.
#   - 'mean_squared_error' calcula el promedio de las diferencias al cuadrado entre las predicciones y los valores reales.
#   - Es una métrica común para problemas de regresión.

# Entrenar el modelo

print("Comenzando entrenamiento...")

historial = modelo.fit(celsius, farenheit, epochs=500, verbose=False)
# Entrena el modelo utilizando los datos proporcionados.
# modelo.fit:
#   - Inicia el proceso de entrenamiento del modelo.
# celsius: ARRAY de entradas (temperaturas en Celsius) que el modelo usará para aprender.
# farenheit: ARRAY de salidas (temperaturas en Fahrenheit) que el modelo intentará predecir.
# epochs=1000:
#   - Define el número de veces que el modelo verá cada dato en el conjunto de entrenamiento.
#   - Más epochs pueden llevar a un mejor ajuste, pero también aumentan el riesgo de sobreajuste.
# verbose=False:
#   - Controla la cantidad de información que se muestra durante el entrenamiento.
#   - False significa que no se mostrará ninguna información por cada epoch.
# historial:
#   - Guarda el historial del entrenamiento, incluyendo la pérdida en cada epoch.

print("Modelo entrenado!! :)")

# Graficar la pérdida durante el entrenamiento

import matplotlib.pyplot as plt  # Importa Matplotlib para crear gráficos.

plt.xlabel("€ Epoca")
# Configura la etiqueta del eje x del gráfico.
# "€ Epoca" parece tener un símbolo de euro por error; probablemente quisiste decir "Época".

plt.ylabel("Magnitud de pérdida")
# Configura la etiqueta del eje y del gráfico.
# "Magnitud de pérdida" representa el valor de la función de pérdida en cada epoch.

plt.plot(historial.history["loss"])
# Dibuja una línea en el gráfico que muestra cómo la pérdida (error) disminuye con cada epoch.
# historial.history["loss"] contiene los valores de pérdida registrados en cada epoch durante el entrenamiento.

plt.show()
# Muestra el gráfico generado en una ventana emergente.
# Esto te permite visualizar cómo el modelo está aprendiendo y si la pérdida está disminuyendo adecuadamente.

# Hacer una predicción con el modelo entrenado

print("Hagamos una predicción!")

resultado = modelo.predict(np.array([[512.0]]))
# Utiliza el modelo entrenado para hacer una predicción.
# modelo.predict:
#   - Toma datos de entrada y devuelve las predicciones del modelo.
# np.array([[512.0]]):
#   - Crea un ARRAY de NumPy con una sola muestra y una característica.
#   - En este caso, intenta predecir la temperatura en Fahrenheit para 512°C.
# resultado:
#   - Contendrá la predicción del modelo en forma de ARRAY de NumPy.

print("El resultado es " + str(resultado[0][0]) + " Fahrenheit!")
# str(resultado[0][0]):
#   - Convierte el valor escalar de la predicción en una cadena de texto.
# "El resultado es ... Fahrenheit!":
#   - Muestra el valor predicho junto con una descripción.
