# Ruta raíz del proyecto (ajusta si no estás en la raíz)
$root = "C:\Users\Jorge Moya\Documents\Academic\I.T.E.S.M\M.E. Inteligencia Artificial Aplicada\4to Trimestre\Operaciones de aprendizaje automático\Tareas o Trabajos\ml-CEE_DATA-project"
Set-Location $root

# 1️⃣ Crear carpetas faltantes
$folders = @(
    "src\data",
    "src\features",
    "src\models",
    "src\visualization",
    "reports\figures",
    "reports\models_summary",
    "references",
    "tests"
)

foreach ($f in $folders) {
    if (!(Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force
        Write-Host "Creada carpeta: $f"
    }
}

# 2️⃣ Mover modelos_summary a reports/models_summary
if (Test-Path "models_summary") {
    Get-ChildItem -Path "models_summary" -File | ForEach-Object {
        Move-Item $_.FullName "reports\models_summary\"
    }
    # Eliminar la carpeta vacía
    Remove-Item "models_summary" -Recurse -Force
    Write-Host "Carpeta models_summary movida a reports/models_summary"
}

# 3️⃣ Verificar archivos .csv u otros reportes en raíz que quieras mover
# (opcional: agrega aquí otras reglas de movimiento si quieres)

# 4️⃣ Git: agregar cambios, commit y push
git add .
git commit -m "Reorganize project to Cookiecutter Data Science structure"
git push
Write-Host "✅ Proyecto reorganizado y push realizado a Git."
