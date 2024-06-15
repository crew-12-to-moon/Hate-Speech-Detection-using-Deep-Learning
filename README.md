<img width="1422" alt="Screenshot 2024-06-11 at 3 13 58 PM" src="https://github.com/crew-12-to-moon/Hate-Speech-Detection-using-Deep-Learning/assets/106720341/0d091a3f-12f3-4478-8b56-9923441aa94b"># Hate-Speech-Detection-using-Deep-Learning
There must be times when you have come across some social media post whose main aim is to spread hate and controversies or use abusive language on social media platforms. As the post consists of textual information to filter out such Hate Speeches NLP comes in handy. This is one of the main applications of NLP which is known as Sentence Classification tasks.

In this article, we will learn how to build an NLP-based Sequence Classification model which can predict Tweets as Hate Speech, Offensive Language, and Normal.

Importing Libraries and Dataset
Python libraries make it very easy for us to handle the data and perform typical and complex tasks with a single line of code.

1.Pandas – This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.

2.Numpy – Numpy arrays are very fast and can perform large computations in a very short time.

3.Matplotlib/Seaborn/Wordcloud– This library is used to draw visualizations.

4.NLTK – Natural Language Tool Kit provides various functions to process the raw textual data.

## Pie chart distribution of classes
![Capture3](https://github.com/crew-12-to-moon/Hate-Speech-Detection-using-Deep-Learning/assets/106720341/c42d0a8a-b572-4444-bed5-459609801c01)
0 - Hate Speech
1 - Offensive Language
2 - Neither

##plotting the graph of loss and accuracy epoch-by-epoch.
![Capture9](https://github.com/crew-12-to-moon/Hate-Speech-Detection-using-Deep-Learning/assets/106720341/b955e2a2-0456-4d2e-896a-2b4228dc39b4)

##Conclusion
The model we have trained is a little over fitting the training data but we can handle this by using different regularization techniques. But still, we had achieved 90% accuracy on the validation data which is quite sufficient to prove the power of LSTM models in NLP-related tasks. 
