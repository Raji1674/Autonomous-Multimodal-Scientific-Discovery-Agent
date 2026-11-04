
FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn

USER 65534:65534

CMD ["python", "--version"]
