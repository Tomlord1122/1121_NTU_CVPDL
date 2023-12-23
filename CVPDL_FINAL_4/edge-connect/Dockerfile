# 選擇一個基礎映像檔，最好與您的本地 Python 版本相匹配
FROM python:3.7.7

# 將工作目錄設置為 /app
WORKDIR /app

# 將本地的 requirements.txt 文件複製到容器中
COPY requirements.txt /app

# 安裝依賴項
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx


# 將您的代碼複製到容器中
COPY . /app

# 您可以指定容器啟動時要運行的命令
CMD ["python", "test.py", "--checkpoints", "./checkpoints/places2", "--input", "./examples/tom/images", "--mask", "./examples/tom/masks", "--output", "./checkpoints/results"]