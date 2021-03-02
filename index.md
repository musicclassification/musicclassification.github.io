## Music Classification

#### Pranay Agrawal, Sidhesh Desai, Pranal Madria, Ayush Nene, Manoj Niverthi

### Overview
In this project, we aim to use supervised and unsupervised learning for music classification purposes.

### Introduction

As avid music listeners, we so often find ourselves listening to the same genres and types of music on a consistent basis. Creating a tool that would allow music enthusiasts to discover, curate, share, and compare pieces of music would allow listeners to foray into a whole new musical experience -- one that would drastically improve the diversity of musical tracks they listen to. As such, we aim to create a music classification tool to allow for the grouping together of similar types of music based on audio features as well as the recognition of music genre from a given piece of music.

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
https://ieeexplore.ieee.org/document/4457263   

http://www.ifs.tuwien.ac.at/~andi/publications/pdf/lid_ismir05.pdf  

http://millionsongdataset.com/  

https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification  

https://ieeexplore.ieee.org/document/1199998  
