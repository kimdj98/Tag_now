FROM python:3.10.5

WORKDIR /app
COPY . .

RUN pip3 install flask && pip3 install pymongo[srv] && pip3 install python-dotenv &&\
    pip3 install pandas && pip3 install numpy && pip3 install scipy

EXPOSE 5000

CMD ["python", "./app.py"]
