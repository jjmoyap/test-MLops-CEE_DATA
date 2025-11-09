# Imagen oficial recomendada (ligera)
FROM python:3.11-slim

# Buenas prácticas para Python en contenedores
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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