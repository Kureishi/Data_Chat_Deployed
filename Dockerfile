FROM python:3.9.15

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y build-essential libgtk-3-dev
RUN pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/debian-9 wxPython==4.2.1

RUN git clone https://github.com/Kureishi/Data_Chat_Deployed.git .

RUN pip install -r app/requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app/app_v6.py", "--server.port=8501", "--server.address=0.0.0.0"]