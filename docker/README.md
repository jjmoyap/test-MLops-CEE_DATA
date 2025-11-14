# API de Predicción con MLflow y FastAPI - Team 51

Documentación para Construcción y Uso de la Imagen Docker

---

## 📋 Descripción General

Este proyecto contiene una API REST desarrollada con **FastAPI** que sirve un modelo de Machine Learning para predecir calificaciones de estudiantes basándose en su edad y horas de estudio. El modelo está empaquetado en formato **MLflow** y se despliega mediante **Docker**.

## 📁 Estructura del Proyecto

```
docker/
├── Dockerfile                  # Definición de la imagen Docker
├── docker-requirements.txt     # Dependencias de Python para la API
├── README.md                   # Este archivo
├── api/
│   └── Team51_ML_API.py       # Código fuente de la API FastAPI
└── model/                      # Modelo MLflow (artefactos)
    ├── MLmodel
    ├── conda.yaml
    ├── python_env.yaml
    ├── requirements.txt
    └── registered_model_meta
```

## ⚙️ Requisitos Previos

- Docker instalado en su sistema (versión 20.10 o superior)
- Conexión a internet (para descargar dependencias)
- Puerto 8880 disponible en su máquina host

---

## 🔨 Opción 1: Construir la Imagen desde el Dockerfile

### Paso 1: Navegar al directorio del proyecto

```bash
cd /ruta/a/tu/proyecto/docker
```

### Paso 2: Construir la imagen Docker

```bash
docker build -t team51-api:latest .
```

**Parámetros:**
- `-t team51-api:latest` : Asigna el nombre "team51-api" y la etiqueta "latest"
- `.` : Indica que el Dockerfile está en el directorio actual

⏱️ **Tiempo estimado:** 2-5 minutos (dependiendo de tu conexión a internet)

### Paso 3: Verificar que la imagen se creó correctamente

```bash
docker images | grep team51-api
```

Deberías ver algo como:
```
team51-api    latest    abc123def456    2 minutes ago    500MB
```

### Paso 4: Ejecutar el contenedor

```bash
docker run \
  -p 8880:8880 \
  -v ./model:/ml/model \
  -v ./api:/ml/api \
  c1544c/team51-api
```

**Parámetros:**
- `-v ./model:/model` : monta un volumen de tu equipo local al contenedor
- `-v ./api:/ml/api` : monta un volumen de tu equipo local al contenedor
- `--name team51-api-container` : Nombre del contenedor
- `-p 8880:8880` : Mapea puerto 8880 del host al 8880 del contenedor
- `team51-api:latest` : Imagen a utilizar

### Paso 5: Verificar que el contenedor está ejecutándose

```bash
docker ps
```

Deberías ver el contenedor "team51-api-container" en estado "Up"

### Paso 6: Ver los logs del contenedor

```bash
docker logs team51-api-container
```

Deberías ver:
```
Usando MODEL_URI = /ml/model
Cargando modelo desde MLflow...
✅ Modelo cargado correctamente.
INFO: Uvicorn running on http://0.0.0.0:8880
```

---

## 🌐 Opción 2: Usar la Imagen Pública

### Paso 1: Descargar la imagen pública desde Docker Hub

```bash
docker pull c1544c/team51-api:latest
```

> **Nota:** La imagen pública está disponible en `c1544c/team51-api`

### Paso 2: Ejecutar el contenedor desde la imagen pública

```bash
docker run -d \
  --name team51-api-container \
  -p 8880:8880 \
  c1544c/team51-api:latest
```

### Paso 3: Verificar el funcionamiento

```bash
docker logs team51-api-container
```

---

## 📤 Publicar la Imagen en Docker Hub

### Paso 1: Crear una cuenta en Docker Hub

Visita: [https://hub.docker.com/signup](https://hub.docker.com/signup)

### Paso 2: Iniciar sesión desde la terminal

```bash
docker login
```

Ingresa tu usuario y contraseña de Docker Hub

### Paso 3: Etiquetar la imagen con tu usuario

```bash
docker tag team51-api:latest <tu-usuario>/team51-api:latest
```

**Ejemplo:**
```bash
docker tag team51-api:latest c1544c/team51-api:latest
```

### Paso 4: Subir la imagen a Docker Hub

```bash
docker push <tu-usuario>/team51-api:latest
```

⏱️ **Tiempo estimado:** 5-15 minutos (dependiendo de tu conexión a internet)

### Paso 5: Verificar en Docker Hub

Visita: `https://hub.docker.com/r/<tu-usuario>/team51-api`

---

## 🧪 Probar la API

### Opción A: Desde el navegador

#### 1. Health Check
```
http://localhost:8880/health
```
**Respuesta esperada:** `{"status":"ok"}`

#### 2. Hola Mundo
```
http://localhost:8880/hola_mundo
```
**Respuesta esperada:** 
```json
{"mensaje":"Hola Mundo desde la API de Team 51 con MLflow! y FastAPI"}
```

#### 3. Documentación interactiva
```
http://localhost:8880/docs
```
Podrás probar todos los endpoints desde la interfaz **Swagger UI**

### Opción B: Desde la terminal con curl

#### 1. Health Check
```bash
curl http://localhost:8880/health
```

#### 2. Predicción individual
```bash
curl -X POST "http://localhost:8880/predict_one" \
  -H "Content-Type: application/json" \
  -d '{"edad": 20, "horas_estudio": 5.5}'
```

**Respuesta esperada:**
```json
{
  "input": {"edad": 20, "horas_estudio": 5.5},
  "calificacion_predicha": 75.3
}
```

#### 3. Predicción por lotes
```bash
curl -X POST "http://localhost:8880/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {"edad": 18, "horas_estudio": 3.0},
      {"edad": 22, "horas_estudio": 7.5},
      {"edad": 19, "horas_estudio": 4.2}
    ]
  }'
```

### Opción C: Desde Python

```python
import requests
import json

# Health check
response = requests.get("http://localhost:8880/health")
print(response.json())

# Predicción individual
data = {"edad": 20, "horas_estudio": 5.5}
response = requests.post("http://localhost:8880/predict_one", json=data)
print(response.json())

# Predicción por lotes
batch_data = {
    "students": [
        {"edad": 18, "horas_estudio": 3.0},
        {"edad": 22, "horas_estudio": 7.5}
    ]
}
response = requests.post("http://localhost:8880/predict_batch", json=batch_data)
print(response.json())
```

---

## 🛠️ Gestión del Contenedor

### Detener el contenedor
```bash
docker stop team51-api-container
```

### Iniciar el contenedor detenido
```bash
docker start team51-api-container
```

### Reiniciar el contenedor
```bash
docker restart team51-api-container
```

### Ver logs en tiempo real
```bash
docker logs -f team51-api-container
```

### Ejecutar comandos dentro del contenedor
```bash
docker exec -it team51-api-container /bin/bash
```

### Eliminar el contenedor
```bash
docker rm -f team51-api-container
```

### Eliminar la imagen
```bash
docker rmi team51-api:latest
```

### Ver estadísticas de uso del contenedor
```bash
docker stats team51-api-container
```

---

## 🔧 Variables de Entorno Configurables

Al ejecutar el contenedor, puedes sobrescribir las variables de entorno:

```bash
docker run -d \
  --name team51-api-container \
  -p 8880:8880 \
  -e PORT=9000 \
  -e MLFLOW_TRACKING_URI=http://mi-servidor-mlflow:5000 \
  team51-api:latest
```

### Variables disponibles:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `PORT` | Puerto donde se ejecuta la API | `8880` |
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://host.docker.internal:8080` |
| `MODEL_URI` | URI del modelo MLflow | `models:/student_grade_regressor/Production` |
| `MODEL_PATH` | Ruta del modelo en el contenedor | `model/` |
| `API_PATH` | Ruta de la API en el contenedor | `api/` |

---

## 💻 Volúmenes para Desarrollo

Si deseas modificar el código sin reconstruir la imagen, puedes montar volúmenes:

```bash
docker run -d \
  --name team51-api-container \
  -p 8880:8880 \
  -v $(pwd)/api:/ml/api \
  -v $(pwd)/model:/ml/model \
  team51-api:latest
```

Esto permite editar el código en tiempo real y reiniciar el contenedor para aplicar los cambios sin necesidad de reconstruir la imagen.

---

## 🐛 Solución de Problemas

### Problema: El contenedor no inicia

**Solución 1:** Verificar logs
```bash
docker logs team51-api-container
```

**Solución 2:** Verificar que el puerto 8880 no esté en uso
```bash
# macOS/Linux
lsof -i :8880

# Windows
netstat -ano | findstr :8880
```

**Solución 3:** Verificar que los archivos del modelo existen
```bash
ls -la model/
```

### Problema: Error "RESOURCE_DOES_NOT_EXIST: Run with id=model not found"

**Solución:** Verificar que `MODEL_URI` apunte a la ruta correcta del modelo

En `Team51_ML_API.py`, asegúrate de que:
```python
MODEL_URI = "/ml/model"
```

### Problema: La API responde lento

**Solución:** Asignar más recursos al contenedor Docker
```bash
docker run -d \
  --name team51-api-container \
  --memory="2g" \
  --cpus="2.0" \
  -p 8880:8880 \
  team51-api:latest
```

### Problema: No puedo conectarme a la API desde fuera del host

**Solución:** Verificar que el puerto está mapeado correctamente
```bash
docker ps  # Verifica que aparezca 0.0.0.0:8880->8880/tcp
```

### Problema: Error al instalar dependencias durante la construcción

**Solución 1:** Limpiar la caché de Docker
```bash
docker builder prune
```

**Solución 2:** Construir sin caché
```bash
docker build --no-cache -t team51-api:latest .
```

---

## 🏗️ Arquitectura del Proyecto

### Flujo de ejecución:

1. Dockerfile copia el modelo y el código de la API al contenedor
2. Se instalan las dependencias de Python
3. Se crea un usuario no-root para seguridad
4. Uvicorn inicia el servidor FastAPI en el puerto 8880
5. MLflow carga el modelo desde `/ml/model`
6. La API queda lista para recibir peticiones

### Endpoints disponibles:

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Verificación del estado del servicio |
| `GET` | `/hola_mundo` | Mensaje de bienvenida |
| `GET` | `/docs` | Documentación interactiva Swagger |
| `POST` | `/predict_one` | Predicción para un estudiante |
| `POST` | `/predict_batch` | Predicción para múltiples estudiantes |

### Modelo de datos:

**Input:**
```json
{"edad": int, "horas_estudio": float}
```

**Output:**
```json
{
  "input": {...},
  "calificacion_predicha": float
}
```

---

## ✨ Mejores Prácticas

### 1. Seguridad:
- ✅ La imagen ejecuta el servicio con usuario no-root (`team51`)
- ✅ No incluir credenciales en el Dockerfile
- ✅ Usar variables de entorno para configuración sensible

### 2. Optimización:
- ✅ Usar imágenes base slim para reducir el tamaño
- ✅ Multi-stage builds si el proyecto crece
- ✅ Aprovechar la caché de Docker organizando comandos correctamente

### 3. Monitoreo:
- ✅ Implementar health checks
- ✅ Centralizar logs
- ✅ Usar herramientas de monitoreo (Prometheus, Grafana)

### 4. CI/CD:
- ✅ Automatizar construcción de imágenes
- ✅ Versionar las imágenes (tags semánticos)
- ✅ Ejecutar pruebas antes de publicar

---

## 📚 Recursos Adicionales

### Documentación oficial:
- [Docker](https://docs.docker.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://www.mlflow.org/docs/latest/index.html)
- [Uvicorn](https://www.uvicorn.org/)

### Tutoriales recomendados:
- [Docker para principiantes](https://docker-curriculum.com/)
- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html)

---

## 👥 Contacto y Soporte

- **Equipo:** Team 51
- **Proyecto:** MLOps - Sistema de Predicción de Calificaciones
- **Repositorio:** test-MLops-CEE_DATA
- **Imagen Docker Hub:** [c1544c/team51-api](https://hub.docker.com/r/c1544c/team51-api)

Para reportar problemas o sugerencias, por favor crear un issue en el repositorio.

---

## 📝 Changelog

### Versión 1.0.0 (14 de noviembre de 2025)

- ✨ Implementación inicial de la API con FastAPI
- 🔗 Integración con MLflow para carga de modelos
- 🐳 Dockerización del servicio
- 📊 Endpoints de predicción individual y por lotes
- 📖 Documentación completa

---

## 📄 Licencia

Este proyecto es parte del curso de MLOps - Team 51

---

**¡Gracias por usar nuestra API! 🚀**
