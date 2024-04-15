FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]