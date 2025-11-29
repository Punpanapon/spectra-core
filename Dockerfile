# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:1.5.7

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PYTHONUNBUFFERED=1

# Create env from the same environment.yml we use locally
COPY --chown=$MAMBA_USER:$MAMBA_USER env/environment.yml /tmp/environment.yml
RUN micromamba create -y -n spectra -f /tmp/environment.yml && micromamba clean -a -y

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Streamlit config inside image
RUN mkdir -p /home/$MAMBA_USER/.streamlit && \
    cp .streamlit/config.toml /home/$MAMBA_USER/.streamlit/config.toml

EXPOSE 8501

# Use flexible entrypoint that honors PORT environment variable
CMD ["micromamba","run","-n","spectra","bash","-lc","/app/scripts/entrypoint.sh"]