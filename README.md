# Redes Neuronales desde Cero

### Objetivo:

El objetivo principal de este proyecto es programar redes neuronales desde cero en Python, sin utilizar librerías específicas de machine learning como _TensorFlow_ o _PyTorch_. Para los cálculos numéricos se empleará principalmente la librería _NumPy_, junto con otras librerías básicas para tareas complementarias. Este proyecto busca entender los fundamentos matemáticos y computacionales detrás de las redes neuronales.

---

### **Características del Proyecto**

- Implementación de redes neuronales desde cero, incluyendo:
  - Capas densas (`Dense`), convolucionales (`Conv2D`), y de pooling (`MaxPool2D`).
  - Funciones de activación como ReLU, Sigmoid, y Softmax.
  - Algoritmos de optimización como SGD y Adam.
  - Funciones de pérdida como entropía cruzada y error cuadrático medio.
- Propagación hacia adelante y hacia atrás (backpropagation) implementada manualmente.
- Soporte para entrenamiento en lotes y validación.

---

### **Cómo Usar**

1. **Preparar los datos**:

   - Descarga los datasets mencionados en la sección "Datasets usados".
   - Coloca los archivos en la carpeta Data.

2. **Entrenar un modelo**:

   - Usa de ejemplo el notebook `test/test_conv_model.ipynb` para entrenar un modelo convolucional.

3. **Probar un modelo**:
   - Modifica la estructura e hiperparámetros a gusto.

---

### Datasets usados:

- **MNIST**: [Descargar aquí](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Cats & Dogs**: [Descargar aquí](https://www.kaggle.com/datasets/unmoved/30k-cats-and-dogs-150x150-greyscale)
- **Signs Language**: [Descargar aquí](https://drive.google.com/file/d/1QQIxRwUhNp5519xVWK8CYY5V_oqsYOpj/view?usp=drive_link)

> [!IMPORTANT]
> Los archivos .csv descargados debes almacenarse en una carpeta 'Data' dentro de la raiz del proyecto para ser cargados y usados.

---

### Estructura del Proyecto:

<!-- - `data/`
  - Carpeta para almacenar datasets utilizados en el proyecto. -->

- `models/`
  - Contiene las funciones y clases necesarias para la creación de redes neuronales.
- `tests/`
  - Pruebas unitarias para verificar la funcionalidad de las implementaciones.
- `training/`
  - Scripts y funciones relacionadas al entrenamiento de las redes neuronales.
- `utils/`
  - Herramientas y funciones auxiliares para tareas comunes.

--

### Requisitos

Para ejecutar este proyecto, instala las dependencias requeridas:

```bash
pip install -r requirements.txt
```

---

### **Contribuciones**

Si deseas contribuir al proyecto:

1. Haz un fork del repositorio.
2. Crea una rama para tu funcionalidad o corrección de errores:
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. Realiza tus cambios y haz un commit:
   ```bash
   git commit -m "Agrega nueva funcionalidad"
   ```
4. Envía un pull request.

---

### **Licencia**

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
