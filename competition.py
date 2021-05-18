"""
Method Description: 
    I used Xgboost method to predict the rating. 
    In the hw3, I have done following: 
        I used review_count, average_stars from user.json file. 
        I also used review count, stars, RestaurantsPriceRange, photo_id from business.json file.
        For the photo_id, I calculated the number of photos per business.
    In addition to these steps, I improved the prediction by calculating sentiment intensity score
    of text (comments) in tip.json file. This helps to identify the score customers gave
    to the business.

Error Distribution:
>=0 and < 1: 102176
>=1 and <2: 32876
>=2 and <3: 6191
>=3 and <4: 800
>=4: 1


RMSE:
    0.979    
Execution Time:
    Execution Time on Validation Set is 126s

"""
import os
import sys
import json
import pyspark
import numpy as np
import xgboost as xgb
from pyspark import SparkContext
import time
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Take care of missing price 
def price_range(biz_attr, price):
    if biz_attr:
        if price in biz_attr.keys():
            return float(biz_attr[price])
    else:
        return 0.0

# Take care of missing photo
def photo_count(biz_id, photos):

    if biz_id not in photos.keys():
        return 0
    else:
        return photos[biz_id]

# Combine training features
def train_combine(user, biz, star, Biz_photo, Biz_price, Biz_features, User, Tip):
    global sentiment
    if biz not in Biz_photo.keys() or biz not in Biz_price.keys() or biz not in Biz_features.keys() or user not in User.keys():
    # in case prediction biz or user not in feature dictionaries
        if biz not in Biz_photo.keys():
            Biz_photo[biz] = 0
        if biz not in Biz_price.keys():
            Biz_price[biz] = 0.0
        if biz not in Biz_features.keys():
            Biz_features[biz] = [0, 0.0]
        if user not in User.keys():
            User[user] = [0, 0.0]
    if (user, biz) not in Tip.keys():
        feature = [Biz_photo[biz], Biz_price[biz], Biz_features[biz][0], Biz_features[biz][1], User[user][0], User[user][1], 0, star]
    else:
        text = Tip[(user, biz)]
        s_dict = sentiment.polarity_scores(text)
        feature = [Biz_photo[biz], Biz_price[biz], Biz_features[biz][0], Biz_features[biz][1], User[user][0], User[user][1], s_dict['compound'], star]

    return ((user, biz), feature) 

 
# Combine test features 
def test_combine(user, biz, Biz_photo, Biz_price, Biz_features, User, Tip): 
    if biz not in Biz_photo.keys() or biz not in Biz_price.keys() or biz not in Biz_features.keys() or user not in User.keys():

        if biz not in Biz_photo.keys():
            Biz_photo[biz] = 0
        if biz not in Biz_price.keys():
            Biz_price[biz] = 0.0
        if biz not in Biz_features.keys():
            Biz_features[biz] = [0, 0.0]
        if user not in User.keys():
            User[user] = [0, 0.0]
    if (user, biz) not in Tip.keys():
        feature = [Biz_photo[biz], Biz_price[biz], Biz_features[biz][0], Biz_features[biz][1], User[user][0], User[user][1], 0]
    else:
        text = Tip[(user, biz)]
        s_dict = sentiment.polarity_scores(text)
        feature = [Biz_photo[biz], Biz_price[biz], Biz_features[biz][0], Biz_features[biz][1], User[user][0], User[user][1], s_dict['compound']]

    return ((user, biz), feature) 

if __name__ == '__main__':
    # Initiating a SparkContext
    sc = pyspark.SparkContext('local[*]', 'Task2_2')
    sc.setLogLevel("WARN")
    
    # Define paths to input foler, training file, output file
    folder = sys.argv[1]
    output_task22 = sys.argv[3] 
    val_filepath  = sys.argv[2] 
    
    
    start = time.time()
    # read data
    train_filepath = folder + '/yelp_train.csv'#sys.argv[1]
    
    train_file = sc.textFile(train_filepath)
    val_file = sc.textFile(val_filepath)
    
    user_file_path = folder + '/user.json'
    user_file = sc.textFile(user_file_path)
    
    biz_file_path = folder + '/business.json'
    biz_file = sc.textFile(biz_file_path)
    
    photo_file_path = folder + '/photo.json'
    photo_file = sc.textFile(photo_file_path)
    
    tip_file_path = folder + '/tip.json'
    tip_file = sc.textFile(tip_file_path)
  
    header_train = train_file.first()
    header_val = val_file.first()
    
    # Loading CSV data 
    yelpTrain = train_file.filter(lambda line: line != header_train).map(lambda f: f.split(","))
    yelpVal = val_file.filter(lambda line: line != header_val).map(lambda f: f.split(","))
    
    # Create dictionaries of User, Biz, and other features
    # user: review_count, average_stars
    User = user_file.map(lambda f: json.loads(f)) \
        .map(lambda f : ((f['user_id'], (f['review_count'], f['average_stars'])))).collectAsMap()
    # Biz: review_count, stars
    Biz = biz_file.map(lambda f: json.loads(f))
    Biz_reviewCount_stars = Biz.map(lambda f : ((f['business_id'], (f['review_count'], f['stars'])))).collectAsMap()
    
    # find price of the business (some business might not have price)
    Biz_price = Biz.map(lambda biz : (biz['business_id'], price_range(biz['attributes'],'RestaurantsPriceRange2'))).collectAsMap()
    
    #  find business - num of photo they have
    Photo_existed = photo_file.map(lambda f: json.loads(f)) \
        .map(lambda f : (f['business_id'], f['photo_id'])).groupByKey() \
        .map(lambda x: (x[0], len(x[1]))).collectAsMap()
    
    Biz_photo = Biz.map(lambda biz: (biz['business_id'],photo_count(biz['business_id'], Photo_existed))).collectAsMap()
    
    tip = tip_file.map(lambda f: json.loads(f)) \
        .map(lambda f: ((f['user_id'], f['business_id']), f['text'])).collectAsMap()

    
    
    sentiment = SentimentIntensityAnalyzer()
    # combine the features to form train set
    train = yelpTrain.map(lambda t: train_combine(t[0], t[1], float(t[2]), Biz_photo, Biz_price, Biz_reviewCount_stars, User, tip)) \
    .map(lambda x: x[1]).collect()
    
#     print(train[:3])
    train_mat = np.array(train)
    X_train, Y_train = train_mat[:,:-1], train_mat[:,-1]
    
    # combine the features to form test set
    val_tuples = yelpVal.map(lambda t: test_combine(t[0], t[1], Biz_photo, Biz_price, Biz_reviewCount_stars, User, tip)) 
    
    val = val_tuples.map(lambda x: x[1]).collect()
    user_biz = val_tuples.map(lambda x: x[0]).collect()
    
    
    X_val = np.array(val)
    # Build a XGBoost model
    xgbModel = xgb.XGBRegressor()
    xgbModel.fit(X_train, Y_train)
     
    prediction = xgbModel.predict(data=X_val)
    
    # Write data into csv
    with open(output_task22, "w+") as task22_output:
        task22_output.write("user_id, business_id, prediction" + "\n")
        for o in range(len(prediction)):
            task22_output.write(user_biz[o][0] + "," + user_biz[o][1] + "," + str(prediction[o]) + "\n")

        
    
    end = time.time() - start
    print('Duration:',round(end,2))