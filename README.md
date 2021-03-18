# Product Personalization Engine (Recommendation-System)
This repository comprises of two different recommendation algorithms([ALS](https://github.com/krishangi-deka/Recommendation-System-for-Products#collaborative-filtering-using-alternating-least-squaresals) and [K-Means Clustering](https://github.com/krishangi-deka/Recommendation-System-for-Products#content-based-filtering-using-k-means)) at a retailer’s end which could help recognize similar and popular products and suggest recommendations for users.

<p align="center">
  <img width="460" height="400" src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/Recc%20System.jpg">
</p>

## Contents
1. [Data Collection](https://github.com/krishangi-deka/Recommendation-System-for-Products#data-collection)
2. [Techniques Used](https://github.com/krishangi-deka/Recommendation-System-for-Products#techniques-used)
3. [Results](https://github.com/krishangi-deka/Recommendation-System-for-Products#results)
4. [Conclusion and Future Scopes](https://github.com/krishangi-deka/Recommendation-System-for-Products#conclusion-and-future-scopes)
5. [References](https://github.com/krishangi-deka/Recommendation-System-for-Products#references)

## Data Collection
Data for this project was collected from [Jianmo Ni’s](https://nijianmo.github.io/amazon/index.html) github profile. The data deals with sports and outdoor equipment. The dataset contains 435924 rows and 12 columns. Originally, the dataset was contained as a .json file zipped into .gz which was extracted using pyspark shell script.

## Techniques Used
### 1. Collaborative Filtering using Alternating Least Squares(ALS) 

Collaborative Filtering:  A filtering technique which analyses information about users with similar tastes to assess the probability of what a target individual will enjoy (a similar product, movie, book etc.). It depends on past interactions recorded between users and items in order to produce new recommendations. The more users interact with products, the more new recommendations become accurate. But because it only considers past interactions to make recommendations, this method of filtering suffers from the “cold start problem” - it is impossible to recommend anything to new users or recommend a new item to any users and many users or items have too few interactions to be efficiently handled. However, there are ways to address the cold-start problem.

Alternating Least Squares (ALS): In this method we have a large matrix and we factor it into smaller matrices through alternating least squares. We end up with two or more lower dimensional matrices whose product equals the original one. ALS comes inbuilt in Apache Spark.
To pre-process the data, the productID and reviewerID columns had to be converted from string to integer using a string indexer in order to prepare them for the ALS regression. While building the recommendation model using ALS on the training data, the ‘cold-start-strategy’ was dropped to avoid getting NaN value in evaluation metrics. Cold starts occur when it is attempted to predict a rating for a user-item pair but there were no ratings for this user/item in the training set. Dropping the cold-start-strategy simply removes those rows/columns from the predictions and from the test set. The result will therefore only contain valid numbers that can be used for evaluation.  

### 2. Content Based Filtering using K-Means

Content Based Filtering: This technique uses features of items to recommend other similar items to users based on their actions and searches. This method therefore requires a good amount of information about the items' own features. For example, it can be features like the product’s manufacturer, quality, size, price etc.

K-Means: K- means is a popularly used unsupervised learning algorithm. K refers to the number of centroids and allocates each centroid with data points closest to it. Optimal value for k can be found by 2 methods: 1) Silhouette score and 2)Elbow method. We have used the Silhouette method to find the optimal value of K.
The reviewText column data had to be pre-processed in order to provide more accuracy in text search. All punctuations and numbers were removed using python regular expressions. Stopwords were also removed. Moreover, in order to convert words into numerical statistics, term frequency–inverse document frequency(TF-IDF) was used. Tf in tf-idf weight measures frequency of terms in a document, and idf measure importance of that given term in given corpus. After performing the k-means algorithm, silhouette method was implemented in order to find the best k value in the range 3-17.     

## Results
### 1. ALS 
Columns Used - productID(asin), reviewerID, rating(overall). 
The dataset was divided using a train-test split of 80-20%. An ALS matrix regression was performed on the training set to predict on the test data with an initial rmse of 1.86
<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/initial_rmse.jpg">
</p>
<p align="center">Fig: RMSE on initial model</p>

To tune the model, a 10 fold cross validation and a parameter grid builder was used where the model rank was increased to 25 to get better results. A 10% improvement in the rmse value was found in the best model whose rmse value reduced to 1.76.
<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/final_rmse.jpg">
</p>
<p align="center">Fig: RMSE after Cross Validation</p>

The output of the final recommendation by the best model , which predicts the top 10 products for the first 10 users is shown below.  
<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/top10als.jpg">
</p>
<p align="center">Fig: Top 10 products for first 10 users</p>

### 2. K-Means
The columns used for this analysis were - productID(asin), reviewerID, reviewText. The best k chosen for this analysis was 10. The figure below shows the silhouette scores for our model for k in range (3-17). Even though the highest score is 0.77 for 3 clusters, considering the size of our dataset, we chose our optimal k as 10.  

<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/silhoutte.jpg">
</p>
<p align="center">Fig: Various Silhouette scores for optimal k value</p>

<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/cluster10.jpg">
</p>
<p align="center">Fig:Showing clusters for k=10</p>

<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/cluster_head.jpg">
</p>
<p align="center">Fig: Showing the head of dataframe with unique productIDs in each cluster</p>

The top 10 products for the search keyword "Skateboard" is shown below:
<p align="center">
  <img src="https://github.com/krishangi-deka/Recommendation-System-for-Products/blob/main/images/top10kmeans.jpg">
</p>
<p align="center">Fig: Recommendations</p>

## Conclusion and Future Scopes
Amongst the models implemented, ALS gives us better results and a more accurate recommendation system. ALS keeps into account user preferences and past history of users to customize better prediction of products to each user. Moreover, ALS can be generated using both user-user and item-item similarity. On the contrary, k-means is good when user data is not available to us and we still wish to provide recommendations to users. This can be used initially by new retailers to provide recommendations to their users since they do not have much customer data.
However, the implementation of recommendation in this project is very raw and primitive in nature and there are numerous improvements and possibilities to improve the accuracy of the models. First of all, to make the predictions more accurate and realistic, data can be scrapped directly through retail sites using scraping tools for web pages. We could also collect data for product names, which is a limitation when it comes to the dataset used in this project. Product names would give us a more tangible recommendation system. Apart from that, more sophisticated models like topic modelling or LDA(Latent Dirichlet Allocation) can be implemented to assign and identify words in k-means clusters, in which the assigned topics and their associated score with words can act as a prediction logic ground.      

## References
1. [https://realpython.com/build-recommendation-engine-collaborative-filtering/#:~:text=Remove%20ads-,What%20Is%20Collaborative%20Filtering%3F,similar%20to%20a%20particular%20user.](https://realpython.com/build-recommendation-engine-collaborative-filtering/#:~:text=Remove%20ads-,What%20Is%20Collaborative%20Filtering%3F,similar%20to%20a%20particular%20user.) 

2. [https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada) 

3. [https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html](https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html)

4. [https://stackoverflow.com/questions/56642128/how-to-use-k-means-for-a-product-recommendation-dataset](https://stackoverflow.com/questions/56642128/how-to-use-k-means-for-a-product-recommendation-dataset)

5. [https://www.kaggle.com/shawamar/product-recommendation-system-for-e-commerce](https://www.kaggle.com/shawamar/product-recommendation-system-for-e-commerce) 

6. [https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/)

7. [https://stanford.edu/~cpiech/cs221/handouts/kmeans.html](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html

8. [https://github.com/LaxmiChaudhary/Amzon-Product-Recommendation ](https://github.com/LaxmiChaudhary/Amzon-Product-Recommendation)

9. [https://spark.apache.org/docs/latest/ml-clustering.html#k-means](https://spark.apache.org/docs/latest/ml-clustering.html#k-means)

10. [https://spark.apache.org/docs/latest/mllib-feature-extraction.html#tf-idf](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#tf-idf) 
