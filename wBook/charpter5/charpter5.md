# Neural networks model

## Perceptron Model

1. *Data Sets*

   Or problem data set:

   | $(x_1,x_2)$ |  Y   |
   | :---------: | :--: |
   |    (0,0)    |  0   |
   |    (0,1)    |  1   |
   |    (1,0)    |  1   |
   |    (1,1)    |  1   |

   And problem data set

   | $(x_1,x_2)$ |  Y   |
   | :---------: | :--: |
   |    (0,0)    |  0   |
   |    (0,1)    |  0   |
   |    (1,0)    |  0   |
   |    (1,1)    |  1   |

   Not problem dataSet

   | $(x_1)$ |  Y   |
   | :-----: | :--: |
   |   (0)   |  1   |
   |   (1)   |  0   |

2. *Experimental Setup*

   win10

   vsCode

   learning_rate = 0.5

3. **Quantitative Results*

   The result of perceptron solve the Or problem is here:

   ![or](模板.assets/or.png)

   The result of perceptron solve the And problem is here:

   ![And](模板.assets/And.png)

   The result of perceptron solve the Not problem is here:

   ![not](模板.assets/not.png)

4. *Further Discussion*

   ​	The perceptron model although can solve linear problem, the linear hyperplane is too close to the positive samples. I think it is necessaly to adjust the orginal model. we need to add Regularization item. So the linear hyperplane won't too close to positive samples.

## Netural Networks Model

1. *Data Sets*

   Handwritten numeral data set:

   A vector with 20*20=400 dimensions, each region corresponds to a dimension, with color represented by a float number (ink) and no color represented by 0 (ink)

   A number range from 1 to 10 represented  the number written in the picture

2. *Experimental Setup*

   hidden_size = 30

   num_labels = 10

   learning_rate = 0.1

3. *Quantitative Results*

   

   | Hdden_size | learning_rate | cost    | accuracy |
   | ---------- | ------------- | ------- | -------- |
   | 25         | 0.05          | 6788.26 | 9.04%    |
   | 30         | 0.05          | 5880.49 | 10.0%    |
   | 25         | 0.1           | 6009.81 | 9.84%    |
   | 30         | 0.1           | 6317.12 | 10.28%   |

   

4. *Further Discussion*

   In my experiment, I think there is sothing wrong in my code. After a long try, I still can't make the model converge. The cost of my alogrithm is very large. And the accuracy of my alogrithm is very small. I think that I need to rebuild my code to try again.

