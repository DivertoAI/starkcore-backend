FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install supervisor

COPY . .

# Add supervisord config
COPY supervisord.conf /etc/supervisord.conf

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisord.conf"]