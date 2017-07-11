This an experimental setup involving a base Nginx image and 2 modified Anaconda Docker images deployed with the help of docker-compose.
The Nginx image is setup to load balance the models deployed in the docker cntainers.

The models have been trained as a part of the NYC green-taxi data analysis from Sept' 2015.
The aim is to predict the tip as the percentage of the total trip amount for unseen records.

./app folder contains the following files
- dockerfile: the modified (anaconda image) dockerfile, 
- rfr-model.pkl: pre-treined pickled RandomForestRegressor scikit-learn model,
- grm-model.pkl: pre-treined pickled GradientBoostedRegressor scikit-learn model,   
- preprocess.py: the preprocessing helper functions
- main.py: model prediction and Flask REST API implementation

./nginx folder contains the following files
- dockerfile: the nginx dockerfile, 
- nginx.conf: nginx config file.

To test the system the following command could be used:
A curl request issued to the nginx container with a sample datapoint is provided below.

curl -H "Content-Type: application/json" -X POST -d '{"VendorID": 2,"lpep_pickup_datetime": "2015-09-21 21:58:22","Lpep_dropoff_datetime": "2015-09-21 22:07:24","Store_and_fwd_flag": "N","RateCodeID": 1,"Pickup_longitude": -73.9604263305664,"Pickup_latitude": 40.80978012084961,"Dropoff_longitude": -73.95526123046875,"Dropoff_latitude": 40.78600692749024,"Passenger_count": 1,"Trip_distance": 2.28,"Fare_amount": 9.5,"Extra": 0.5,"MTA_tax": 0.5,"Tip_amount": 2.16,"Tolls_amount": 0,"Ehail_fee": "","improvement_surcharge": 0.3,"Total_amount": 12.96,"Payment_type": 1,"Trip_type": 1}' http://172.19.0.4/predict

Please replace the ip at the end of the command with the ip of your Nginx container. 
This can be obtained via the "docker inspect imageID" command. 
The system only supports the POST HTTP methods or and application/json content-type currently.
