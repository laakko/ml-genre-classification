# genre-classification

Final project for *Machine Learning* course:  

Identifying music genres of songs using classification methods for a [Kaggle](https://www.kaggle.com/) competition.   

Training data set is a custom subset of Million Song Database and training labels
are gotten from AllMusic.com. The features contain 3 main components of music: timbre, pitch and rhythm. Genres corresponding to the 10 labels are:  
1   'Pop_Rock'  
2   'Electronic'  
3   'Rap'  
4   'Jazz'  
5   'Latin'  
6   'RnB'  
7   'International'  
8   'Country'  
9   'Reggae'  
10  'Blues  

The problem was divided into 10 classification subproblems for each 10 genres (for example "Blues" and "Not Blues"), and the subproblems along with training data were fed to logistic regression function, which output the weight vectors. Test data and the obtained weight vectors were passed to sigmoid function, which then passed the final predicted probabilities for each genre.
  

Resulting accuracy in the Kaggle competition was 0.610, which is sufficient but has plenty of room for improvement.
