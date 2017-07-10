from continuumio/anaconda3
RUN apt-get update
RUN conda install -c anaconda flask=0.12.2
RUN apt-get install -y libgl1-mesa-glx
RUN pip install haversine
RUN echo 'Docker image for ML deployment with flask API'
COPY ./ /app
WORKDIR ./app

EXPOSE 8080

ENTRYPOINT ["python"]
CMD ["./main.py"]
