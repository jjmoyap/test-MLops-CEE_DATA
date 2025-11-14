# Imagen oficial recomendada (ligera)
FROM python:3.11-slim

# Buenas prácticas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Constantes del entorno, estás pueden ser modificadas al ejecutar el contenedor
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:8080
ENV MODEL_URI=models:/student_grade_regressor/Production
ENV PORT=8880
ENV MODEL_PATH=model/
ENV API_PATH=api/

# Puerto por defecto del servidor de modelos, ojo no expone el puerto fuera del contenedor
# solo avisa que internamente en el contenedor el servicio de API 
# estará ejecutándose en el puerto 8880
EXPOSE 8880

# Directorio de trabajo dentro del contenedor
WORKDIR /ml

# Dependencias del sistema si alguna lib de Python las requiere
RUN apt-get update && apt-get install -y --no-install-recommends 

# 1) Copiar del path local al contenedor la carpeta del modelo 
COPY ${MODEL_PATH} /ml/model
# 2) Copiar del path local al contenedor la carpeta del modelo
COPY docker-requirements.txt /tmp/docker-requirements.txt
# 3) Copiar el 'Hola Mundo' de la API al contenedor
COPY ${API_PATH} /ml/api

# 4) Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r /tmp/docker-requirements.txt
RUN pip install -r /ml/model/requirements.txt

# 5) Crear usuario no-root por seguridad
RUN useradd -m team51 && chown -R team51 /ml
USER team51

# 6) Para para exponer API cuando se ejecute con docker run
CMD ["uvicorn", "api.Team51_ML_API:app", "--host", "0.0.0.0", "--port", "8880"]