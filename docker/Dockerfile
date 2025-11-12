# Imagen oficial recomendada (ligera)
FROM python:3.11-slim

# Buenas prácticas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

    # Constantes del entorno
ENV MLFLOW_TRACKING_URI=https://mlflow.miempresa.com
ENV MODEL_URI=models:/my-model/Production
ENV PORT=8880

# Puerto por defecto del servidor de modelos, ojo no expone el puerto fuera del contenedor
# solo avisa que internamente en el contenedor el servicio de API 
# estará ejecutándose en el puerto 8880
EXPOSE 8880

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Dependencias del sistema si alguna lib de Python las requiere
RUN apt-get update && apt-get install -y --no-install-recommends 

# 1) Copiar solo requirements primero (mejor caché de Docker)
COPY requirements.txt /tmp/requirements.txt

# 2) Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# 3) Copiar tu código fuente (carpetas y archivos) al contenedor
#    Ajusta según tu estructura (src/, app/, etc.)
COPY . /app

# 5) Crear usuario no-root por seguridad
RUN useradd -m team51 && chown -R team51 /app
USER team51

# 6) Exponer el API
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8880"]