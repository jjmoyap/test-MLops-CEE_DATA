# Imagen oficial recomendada (ligera)
FROM python:3.11-slim

# Buenas prácticas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Constantes del entorno, estás pueden ser modificadas al ejecutar el contenedor
ENV PORT=8880

# Puerto por defecto del servidor de modelos, ojo no expone el puerto fuera del contenedor
# solo avisa que internamente en el contenedor el servicio de API 
# estará ejecutándose en el puerto 8880
EXPOSE 8880

# Directorio de trabajo dentro del contenedor
WORKDIR /ml

# Dependencias del sistema si alguna lib de Python las requiere
RUN apt-get update && apt-get install -y --no-install-recommends 

# 1) Copiar del path local al contenedor la carpeta del modelo 
COPY models/ /ml/models/
COPY docker/docker-requirements.txt /tmp/docker-requirements.txt
COPY serving/ /ml/serving/
COPY src/ /ml/src/

# 4) Instalar dependencias de Python y configurar locales
RUN pip install --upgrade pip
RUN pip install -r /tmp/docker-requirements.txt

RUN apt-get install -y locales \
    && locale-gen C.UTF-8 \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=C.UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8


# 5) Crear usuario no-root por seguridad
RUN useradd -m team51 && chown -R team51 /ml
USER team51

# 6) Para para exponer API cuando se ejecute con docker run
CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8880"]