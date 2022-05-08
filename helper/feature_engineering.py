import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import pickle 

def feature_engineering(cars_data, state):
    cars_data.rename(columns={"dateCreated": "ad_created",
                  "dateCrawled": "date_crawled",
                  "fuelType": "fuel_type",
                  "lastSeen": "last_seen",
                  "monthOfRegistration": "registration_month",
                  "notRepairedDamage": "unrepaired_damage",
                  "nrOfPictures": "num_of_pictures",
                  "offerType": "offer_type",
                  "postalCode": "postal_code",
                  "powerPS": "power_ps",
                  "vehicleType": "vehicle_type",
                  "yearOfRegistration": "registration_year"}, inplace=True)

    cars_data[["ad_created", "date_crawled", "last_seen"]] = cars_data[["ad_created", "date_crawled", "last_seen"]].astype('datetime64')

    cars_data['odometer'] = cars_data['odometer'].str.rstrip('km')
    cars_data['odometer'] = cars_data['odometer'].str.replace(',','')
    cars_data['odometer'] = cars_data['odometer'].astype('int64')

    cars_data['price'] = cars_data['price'].str.lstrip('$')
    cars_data['price'] = cars_data['price'].str.replace(',','')
    cars_data['price'] = cars_data['price'].astype('int64')

    cars_data.drop(['name', 'postal_code', 'seller', 'offer_type', 'num_of_pictures'], axis=1, inplace=True)

    cars_data = cars_data[(cars_data['price']>=500) & (cars_data['price']<=40000)].reset_index()

    for i in cars_data.columns:
        if cars_data[i].dtype == 'object':
            cars_data[i] = cars_data[i].fillna(cars_data[i].mode()[0])
        if cars_data[i].dtype == 'int64':
            cars_data[i] = cars_data[i].fillna(cars_data[i].median())

    num_cols = [col for col in cars_data.columns if (cars_data[col].dtype == 'int64')]
    num_cols.remove('price')

    cars_data[num_cols] = normalize(cars_data[num_cols])

    obj_cols = [col for col in cars_data.columns if (cars_data[col].dtype == 'object')]

    oh_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    cars_data_ohe = pd.DataFrame(oh_enc.fit_transform(cars_data[obj_cols]).toarray())
    cars_data = cars_data[num_cols].join(cars_data_ohe)