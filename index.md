### Overview
In this project, we aim to use supervised and unsupervised learning for music classification purposes.

### Introduction

As avid music listeners, we so often find ourselves listening to the same genres and types of music on a consistent basis. Creating a tool that would allow music enthusiasts to discover, curate, share, and compare pieces of music would allow listeners to foray into a whole new musical experience -- one that would drastically improve the diversity of musical tracks they listen to. As such, we aim to create a music classification tool to allow for the grouping together of similar types of music based on acoustic and non-acoustic features as well as the recognition of music genre from a given piece of music.

### Problem Definition
    
Given a piece of music, we would like to use a supervised approach to accurately classify the genre to which it belongs. For the unsupervised approach, we would like to group the music with other similar types of music based on both acoustic and non-acoustic features and provide recommendations for other songs which are similar to the sample in question. 

### Methods

The first step will involve constructing a music dataset and extracting features about different songs -- both acoustic features and non-acoustic features. From that point onward, we will separate our analysis into two parts -- an unsupervised component and a supervised component. 
We will use supervised learning to predict the genre of a piece of music, comparing the performance of models such as Support Vector Machines, K-Nearest Neighbors, and Naive Bayes. 
We can employ unsupervised learning to cluster the dataset based on acoustic and non-acoustic features. These clusters can be used to give suggestions for other songs that the user may enjoy. 
These models will be trained and tested using the GTZAN and Million Song datasets, and we will use grid search in order to fine-tune our hyperparameters. Employing principal component analysis will allow us to reduce the dimensionality of our dataset. 

### Potential Results
    
Our desired outcome is to be able to input in a series of unknown pieces of music as audio files and have them grouped together based on similarity to form usable “recommendation groups.” For our supervised component, our desired outcome is to be able to output the corresponding genre based on a given piece of music. For our stretch goal, we define our desired outcome as having a user input a musical composition and the supervised model successfully outputs the corresponding genre, while the unsupervised model outputs a few related musical compositions. 

### Discussion
By doing this project, we hope to further strengthen our understanding of both supervised and unsupervised learning along with a deeper understanding of similarities and dissimilarities between musical pieces. We can use the methods described by Lidy & Rauber (2005) to extract features from the tracks in the GTZAN and Million Song datasets. We expect the k-nearest neighbor method to perform the best in regard to classifying music due to its “voting-based” classification approach, allowing proximal songs to be grouped most closely together.


### References
Changsheng Xu, N. C. Maddage, Xi Shao, Fang Cao and Qi Tian, "Musical genre classification using support vector machines," 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings. (ICASSP '03)., Hong Kong, China, 2003, pp. V-429, doi: 10.1109/ICASSP.2003.1199998.  

D. Kim, K. Kim, K. Park, J. Lee and K. M. Lee, "A music recommendation system with a dynamic k-means clustering algorithm," Sixth International Conference on Machine Learning and Applications (ICMLA 2007), Cincinnati, OH, USA, 2007, pp. 399-403, doi: 10.1109/ICMLA.2007.97.  

Lidy, Thomas & Rauber, Andreas. (2005). Evaluation of Feature Extractors and Psycho-Acoustic Transformations for Music Genre Classification.. 34-41.   

Olteanu, A. (2020, March 24). GTZAN dataset - music genre classification. Retrieved from https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification  


Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.  
