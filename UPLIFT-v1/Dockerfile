FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace

RUN useradd --create-home --shell /bin/bash uplift

USER uplift

CMD ["/bin/bash"]
