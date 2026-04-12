# Dockerfile — Scan2Plan
# Exécution du pipeline en conteneur Linux.
#
# Build :
#   docker build -t scan2plan:latest .
#
# Utilisation :
#   docker run --rm \
#     -v /chemin/vers/scan.e57:/data/input.e57 \
#     -v /chemin/vers/output:/output \
#     scan2plan:latest /data/input.e57 /output/plan.dxf

# ---------------------------------------------------------------------------
# Étape 1 — Builder : installe les dépendances et construit le wheel
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Outils système nécessaires pour open3d et scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement les fichiers nécessaires au build
COPY pyproject.toml README.md ./
COPY src/ src/
COPY config/ config/

# Mettre à jour pip et construire le wheel
RUN pip install --upgrade pip build && \
    python -m build --wheel --outdir /dist

# ---------------------------------------------------------------------------
# Étape 2 — Runtime : image finale légère
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Métadonnées
LABEL org.opencontainers.image.title="Scan2Plan"
LABEL org.opencontainers.image.description="Transformation nuage de points 3D → plan 2D DXF"
LABEL org.opencontainers.image.authors="AIT BARANE Said"
LABEL org.opencontainers.image.version="0.2.0"

# Dépendances système runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer le wheel depuis le builder
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm /tmp/*.whl

# Copier les fichiers de configuration par défaut
COPY config/ /app/config/

# Répertoires de travail I/O
RUN mkdir -p /data /output

WORKDIR /app

# Healthcheck : la CLI doit répondre
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD scan2plan --help || exit 1

# Point d'entrée par défaut : la CLI
ENTRYPOINT ["scan2plan"]
CMD ["--help"]
