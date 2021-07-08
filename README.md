# Dataset description:  
The dataset contains 60000 32x32 RGB images (50000 are used for training and 10000 for testing) labeled over 10 different categories.  
The categories are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  
# An example of each category:  
![image](https://user-images.githubusercontent.com/72389636/124950678-ecb12e80-e012-11eb-9ab3-7d412bfb452a.png)  
# Neural network's structure:  
2x Conv2D, BatchNormalization, MaxPool2D, Dropout -> 2x Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten -> Dense, Dropout, Dense  
# Neural network's score and loss after each epoch:  
![image](https://user-images.githubusercontent.com/72389636/124956375-351f1b00-e018-11eb-946c-e46dc53d9fe2.png)  
# Requirements:  
```tensorflow==2.5.0```  
```matplotlib==3.4.1```  
```numpy==1.19.5```  
