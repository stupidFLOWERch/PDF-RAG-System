FROM python:3.12

WORKDIR /app

# Linux system libraries required by OpenCV / PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

# CPU-only PaddlePaddle
RUN pip install --no-cache-dir \
    paddlepaddle==3.2.2 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]