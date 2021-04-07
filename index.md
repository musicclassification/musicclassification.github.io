### [Link to proposal video](https://drive.google.com/file/d/1kwqb7UsB2iS4vsniqZ7w89nSrMh75e7L/view?usp=sharing)

### Overview
In this project, we aim to use supervised and unsupervised learning for music classification purposes.

### Introduction

As avid music listeners, we so often find ourselves listening to the same genres and types of music on a consistent basis. Creating a tool that would allow music enthusiasts to discover, curate, share, and compare pieces of music would allow listeners to foray into a whole new musical experience -- one that would drastically improve the diversity of musical tracks they listen to. As such, we aim to create a music classification tool to allow for the grouping together of similar types of music based on acoustic and non-acoustic features as well as the recognition of music genre from a given piece of music.

### Problem Definition
    
Given a piece of music, we would like to use a supervised approach to accurately classify the genre to which it belongs. For the unsupervised approach, we aim to group the music with other similar types of music based on both acoustic and non-acoustic features and provide recommendations for other songs which are similar to the sample in question. 

### Data Collection

We began by looking for datasets involving music. As stated in our project proposal, we found the datasets GTZAN and Million Songs as the perfect match. We started by downloading these datasets and we began by using the 30-second songs feature dataset. We wanted to see correlations between features and thus we used a heatmap to visually identify correlations between different features.
![Correlation Heatmap](/assets/correlation_matrix.png "Correlation Heatmap")
As seen, certain features were highly correlated with others. Thus, this inspired us to transform our data from 58 features to 2 engineered features using PCA as this will simplify our dataset while withholding important information. 

Here, we will explain some of the features within this dataset:
Zero Crossing Rate: the rate at which the the audio signal modulates between positive and negative 
Mel-Frequency Cepstral Coefficients (MFCC): a set of features which essentially capture the human envelope of speech (in the dataset, we use 20 of these features)
Spectral Centroid: the weighted average of frequencies samples at a specific point in time
Spectral Rolloff: shape of the signal that represents a threshold below which most of the signal lies
Tempo BPM: beats per minute in the audio piece
Harmonics: overtones which are present in the sound sample (very hard to distinguish using the human ear)
Perceptual: features that encodes the rhythm and beat of the piece

### Methods

The first step will involve constructing a music dataset and extracting features about different songs -- both acoustic features and non-acoustic features. From that point onward, we will separate our analysis into two parts -- an unsupervised component and a supervised component. 
We will use supervised learning to predict the genre of a piece of music, comparing the performance of models such as Support Vector Machines, K-Nearest Neighbors, and Naive Bayes. 
We can employ unsupervised learning to cluster the dataset based on acoustic and non-acoustic features. These clusters can be used to give suggestions for other songs that the user may enjoy. 
These models will be trained and tested using the GTZAN and Million Song datasets, and we will use grid search in order to fine-tune our hyperparameters. Employing principal component analysis will allow us to reduce the dimensionality of our dataset. 

Unsupervised

The original dataset had 58 features and thus we decided to transform our dataset using PCA. We wanted to determine the optimal number of principal components, thus we used the explained variance ratio.

![Explained Variance](/assets/explained_variance.png "Explained Variance")

We can see that running PCA on our dataset provides a greatly simplified feature set that is still able to capture the vast majority of variance present in the dataset. With just num_features=2, we are able to capture > 0.99 of the variance in the original dataset, prompting us to utilize this as our threshold value for our features post-PCA.

Thus, we successfully transformed our original dataset of 58 features into a simplified feature set of just 2 engineered features. We wanted to use both the transformed and pre-transformed data when employing our approaches for exploratory reasons as we could compare the two feature sets.

Now, we were ready to begin employing our unsupervised/supervised learning approaches. 

We began by employing unsupervised learning where we used GMM and K-Means to cluster our data. 
For K-Means, we used the elbow method to determine the optimal number of clusters : 
For the pre-transformed data (with 58 features) we received this when utilizing the elbow method:

![Pre-Transformed Elbow](/assets/pre_transformed_elbow.png "Pre Transformed Elbow Method")

As seen, the optimal number of clusters determined by the elbow method was 5. 

For the transformed data post PCA, the output was as follows:

![Transformed Elbow](/assets/transformed_elbow.png "Transformed Elbow Method")

As seen, the post PCA data which had 2 engineered features, was extremely similar to the first when utilizing the elbow method. The optimal number of clusters, as determined by the elbow method, was 5. 

We then ran K-Means with 5 clusters and the output and analysis is listed in the results section. 

Afterwards, we utilized our other clustering method: GMM. For GMM, we clustered the data using several different numbers of clusters and then used the silhouette coefficient to analyze the optimal number of clusters as well as the purity scores to evaluate the quality of clustering. The outputs are in the Results section. 

For the supervised portion of this project, we will use supervised learning to predict the genre of a piece of music, comparing the performance of models such as Support Vector Machines, K-Nearest Neighbors, and Naive Bayes. We may use grid search to tune the hyperparameters for our supervised learning approaches.



### Results
    
Unsupervised

For the unsupervised portion of this project, we utilized two clustering methods -- KMeans clustering and a Gaussian Mixture Model. 

![Pre-Transformed Purity](/assets/pre_transformed_purity.png "Pre Transformed Purity Method")

![Pre-Transformed Silhouette](/assets/pre_transformed_silhouette.png "Pre Transformed Silhouette Method")

We know that the canonical number of genres in the GTZAN dataset is 10. Any result above this number is likely the cause of overfitting. The silhouette coefficient suggests that the optimal number of clusters is indeed 10, the number of genres.

However, the purity score tells a different story. For all the numbers we tested, the purity is low, which implies that this unsupervised clustering is not a good way to identify different genres. This suggests that tracks of a single genre do not necessarily share many features in common. To find a track which closely resembles an input track, it may be necessary to look in other genres. 

For the transformed data GMM output, we received:
![Transformed Purity](/assets/transformed_purity.png "Transformed Purity Method")

![Transformed Silhouette](/assets/transformed_silhouette.png "Transformed Silhouette Method")

Again, the silhouette coefficient suggests that either 8 or 10 is the optimal number of clusters, but the purity score also seems to confirm, albeit with a low score, that 10 is indeed the optimal number of clusters. There was a noticeable deviation between the model’s performance on the PCA-transformed data and the original data. With both purity and silhouette coefficient scores, we hypothesize that this was due to the simplification of the data. Since so many features were removed, the boundaries between different classes were more malleable and may have blended closer together, increasing the ambiguity in the clustering and decreasing the scores of the models compared to the original features.

For the KMeans clustering approach, we plotted the silhouette coefficient for both the non-transformed data and the transformed data to validate our result of 5 clusters as determined by the elbow method. 

Pre-transformed, original data:

![Pre-Transformed Silhouette on KMeans](/assets/kmeans_pre_silhouette.png "Pre-Transformed Silhouette Method on Kmeans")

Transformed Data, post PCA:

![Transformed Silhouette on KMeans](/assets/kmeans_transform_silhouette.png "Transformed Silhouette Method on KMeans")

Using the Silhouette method to validate the results of the analysis was slightly less favorable. Typically, Silhouette values closer to 1 indicate less ambiguity when making decisions using a clustering algorithm, since a higher value on the scale of [-1, 1] indicates less ambiguity in terms of classifying points and more decisiveness by the algorithm, while lower values are a sign of potential conflict for data points, as they more closely to the line to the decision boundaries in the clustering algorithm. As we increase the number of clusters, we can see that the scores slightly decrease, from a high of around 0.6 towards < 0.5 with a higher number of clusters. This can partially be attributed to the increasing granularity of clusters (i.e. as the number increases, more data points fall into conflict between being part of one cluster or another, since decision boundaries are tighter, therefore increasing the Silhouette value).


Although the silhouette coefficient didn’t fully support our optimal number of clusters as determined by the elbow approach, we decided to go with 5 as the optimal number of clusters due to the elbow approach.
We have also included a visual breakdown of what each of these clusters are composed of. 
Original, pre-transformed data K-Means:

![Cluster 1](/assets/pie1_kmeans_pre.png "Pre-Transformed Clustering Results")

![Cluster 2](/assets/pie2_kmeans_pre.png "Pre-Transformed Clustering Results")

![Cluster 3](/assets/pie3_kmeans_pre.png "Pre-Transformed Clustering Results")

![Cluster 4](/assets/pie4_kmeans_pre.png "Pre-Transformed Clustering Results")

![Cluster 5](/assets/pie5_kmeans_pre.png "Pre-Transformed Clustering Results")

As can be seen, a cluster number of 5 does group similar genres, such as classical & blues and pop, hip hop, & reggae together.

Now, we will run K-Means on the transformed data, post PCA:

![Cluster 1](/assets/pie1_kmeans_transform.png "Transformed Clustering Results")

![Cluster 2](/assets/pie2_kmeans_transform.png "Transformed Clustering Results")

![Cluster 3](/assets/pie3_kmeans_transform.png "Transformed Clustering Results")

![Cluster 4](/assets/pie4_kmeans_transform.png "Transformed Clustering Results")

![Cluster 5](/assets/pie5_kmeans_transform.png "Transformed Clustering Results")

In the post-transformed feature set, KMeans seems to give some interesting results such as grouping classical and metal together. It does a good job grouping similar genres such as pop, hip hop, and reggae together. It also understandably groups classical and blues together, but also strangely groups country and disco together. It is possible that instead of grouping genres together it is actually grouping certain structures of songs together. 
Soon, we will work to get results for our supervised learning approach. 
    
Our desired outcome is to be able to input in a series of unknown pieces of music as audio files and have them grouped together based on similarity to form usable “recommendation groups.” For our supervised component, our desired outcome is to be able to identify the genre of a given piece of music. For our stretch goal, we define our desired outcome as having a user input a musical composition and the supervised model successfully outputs the corresponding genre, while the unsupervised model outputs a few related musical compositions. 

### Discussion
Thus far, we have collected the data, transformed the data using PCA, and completed our unsupervised learning approaches of GMM and K-Means and evaluated the results. We have also delved deeply into the features themselves to find similarities using heatmaps of correlations between the features. We have noticed some differences in KMeans and GMM and analyzed them such as the optimal number of clusters between the two. Next, we will be employing our supervised approaches and evaluating those results and perhaps running grid search to effectively tune the hyperparameters for our models.
By doing this project, we hope to further strengthen our understanding of both supervised and unsupervised learning along with a deeper understanding of similarities and dissimilarities between musical pieces. In terms of difficulties with our project, two of the most difficult aspects will likely be audio feature extraction since we have to generate quantifiable features from relatively short audio clips and implementation of our supervised learning algorithms. Our goal with this project is to make it easier to categorize music and group music effectively, but our overall aim and future goal is to utilize this in a way so that users can get quality music recommendations based on the specific types of songs they like. From our analysis, most modern music recommendation tools are based on surface level comparisons like the artist rather than actually looking into the audio features of sounds, so our research will be useful for creating a more powerful recommendation system.




### References
Changsheng Xu, N. C. Maddage, Xi Shao, Fang Cao and Qi Tian, "Musical genre classification using support vector machines," 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings. (ICASSP '03)., Hong Kong, China, 2003, pp. V-429, doi: 10.1109/ICASSP.2003.1199998.  

D. Kim, K. Kim, K. Park, J. Lee and K. M. Lee, "A music recommendation system with a dynamic k-means clustering algorithm," Sixth International Conference on Machine Learning and Applications (ICMLA 2007), Cincinnati, OH, USA, 2007, pp. 399-403, doi: 10.1109/ICMLA.2007.97.  

Lidy, Thomas & Rauber, Andreas. (2005). Evaluation of Feature Extractors and Psycho-Acoustic Transformations for Music Genre Classification.. 34-41.   

Olteanu, A. (2020, March 24). GTZAN dataset - music genre classification. Retrieved from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification  


Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.  
