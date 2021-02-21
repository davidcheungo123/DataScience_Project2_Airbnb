# Airbnb Machine Learning
## Introduction:
The goal of this project is to predict acceptable price range given particular properties of a house on Airbnb so that Airbnb can provide appropiate pricing for hosts. Therefore, hosts can optimize their incomes by setting the price into acceptable price range. Therefore, we are here to answer the question, *"How to price a new listing on Airbnb?* Some feature engineering, data cleaning and model selection techniques are employed in this project. This documentation will explain algorithms used for extracting information to improve accuracy and learning methods to get the best result.

The dataset is obtained from <a href="https://www.kaggle.com/airbnb/seattle">kaggle</a> which contains data in Seattle. There are many attributes (92) for houses including their own ids. 

## Data Prepocessing:
We use a number of features to do the prediction, the selected features are depicted in following coding:
```
cols = ['accommodates', 'bathrooms', 'bedrooms', 'beds', "price", "number_of_reviews", "room_type", "host_listings_count", 
        "review_scores_location", "review_scores_rating", "minimum_nights", "guests_included", "property_type", "amenities"]
```
This results came from *PCA*, which will analyze the importance index of features. We selected around 10-20 attributes so that the model will not overfit. First due to the format of attri "price" and "amenities", ```apply``` function was required to change their formats.

<p align="center"><img  src="./image/1.png" alt="price and amenities" width="400"/></p>

For "price", we used ```apply``` function to replace the *$*, "," signs and changed them to float instead of string:
```
df_1["price"] = df_1["price"].apply(lambda x : float(x.replace("$", "").replace(",", "")))
```
For better interpretation, we defined a specific function to make "price" becomes categorical.
```
def priceCat(x):
    if x>=0 and x<=75:
        return 0
    elif x>=76 and x<=120:
        return 1
    elif x>=121 and x<=200:
        return 2
    elif x >= 201 and x<= 300:
        return 3
    elif x>=301:
        return 4
    else:
        return np.nan
    
df_1["price"] = df_1["price"].apply(priceCat)
```
*note : More research is required to determine better price ranges for analysis/ business' usage.*

