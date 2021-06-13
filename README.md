# We use 5 kinds of model to build our model for the social bot decetion 

- Yifeng Luo

---
## how to see the code

**For each jupyter notebook file represent a step during our experiment**
>> each index represent one notebook file

1. Data Preprocessing and Data simple Expore
2. Clean Data using Regular expression
3. Traditional machine learning algorithms
    - Logistic Regression
    - Random Forests
4. Deep Learning method
    - Simple Neural network
5. Transformer architecture: Bert
    - Simple Bert Model
    - Bert + BiLSTM 

>> See the result plots in each jupyter notebook



>> 5, 6 is in the folder `5,6_Bert_and_Bert_BiLstm`


## Result
<best>
- Logistic Regression : c(正则惩罚项)=100
    - recall = 0.939071749738425
    - test recall = 0.8805970149253731 , test acc = 0.86
>> Logistic Regression is not the best model cuz i have used c parameter to tweak it     
- Random Forests
    - test acc = 0.86
    - test recall = 0.8756218905472637

- Simple NN 
    - test acc = 0.6272974610328674 , test loss = 0.6595907807350159

- Bert only after 6 epochs
    -  Test Loss: 0.23, Test Acc:92.74%

- Bert Bilstm only after 6 epochs
    - Test Loss: 0.23, Test Acc:91.28%

# BERT is the best 
