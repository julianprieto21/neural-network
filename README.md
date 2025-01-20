# Redes Neuronales desde Cero

### Objetivo:

El objetivo principal de este proyecto es programar redes neuronales desde cero en Python, sin utilizar librerías específicas de machine learning como _TensorFlow_ o _PyTorch_. Para los cálculos numéricos se empleará principalmente la librería _NumPy_, junto con otras librerías básicas para tareas complementarias. Este proyecto busca entender los fundamentos matemáticos y computacionales detrás de las redes neuronales.

### ToDo:

- [x] Implementar una Red Neuronal Convolucional (CNN) funcional.
- [ ] Desarrollar una Red Neuronal Recurrente (RNN) funcional.
- [ ] Generalizar la funcionalidad de GradCam (CNN) para diferentes arquitecturas y tipos de datos.
- [ ] Optimizar cálculos recursivos y operaciones matriciales utilizando _Numba_ para mejorar el rendimiento.
- [ ] Incluir más métricas y funciones de pérdida.

### Datasets usados:

- **MNIST**: [Descargar aquí](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Cats & Dogs**: [Descargar aquí](https://www.kaggle.com/datasets/unmoved/30k-cats-and-dogs-150x150-greyscale)

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

### Requisitos

Para ejecutar este proyecto, instala las dependencias requeridas:

```bash
pip install -r requirements.txt
```
