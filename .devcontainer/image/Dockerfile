FROM python:3.12-slim as seismometer

RUN useradd --create-home --uid 1000 --shell /bin/bash seismo
ENV PATH=${PATH}:/home/seismo/.local/bin

COPY ./stagefiles /env/
RUN /bin/sh /env/layer.sh

USER seismo
RUN pip install --no-cache-dir -r /env/python/requirements.txt

# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL \
    org.opencontainers.image.vendor="Epic" \
    org.opencontainers.image.title="Seismometer" \
    org.opencontainers.image.description="Development container for AI Trust & Assurance Toolset"

WORKDIR /home/seismo/workspace
CMD /bin/bash
