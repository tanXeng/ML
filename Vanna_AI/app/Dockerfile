FROM python:3.11.9

WORKDIR /app

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends curl gcc g++ gnupg unixodbc-dev ca-certificates

# Copy custom CA cert into the container
COPY aie-root-ca.pem /usr/local/share/ca-certificates/aie-root-ca.crt
RUN update-ca-certificates

ENV ACCEPT_EULA=Y
# set cert env vars
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
  && curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends --allow-unauthenticated msodbcsql17 mssql-tools

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN pip install pyodbc openai chromadb loguru iso-639 fastapi
COPY * /app/

RUN python /app/onnx_download.py

CMD [ "python3", "vanna-flask.py" ]
