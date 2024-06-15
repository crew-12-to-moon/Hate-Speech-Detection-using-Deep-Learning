# Lower case all the words of the tweet before any preprocessing
df['tweet'] = df['tweet'].str.lower()

# Removing punctuations present in the text
punctuations_list = string.punctuation
def remove_punctuations(text):
	temp = str.maketrans('', '', punctuations_list)
	return text.translate(temp)

df['tweet']= df['tweet'].apply(lambda x: remove_punctuations(x))
df.head()
def remove_stopwords(text):
	stop_words = stopwords.words('english')

	imp_words = []

	# Storing the important words
	for word in str(text).split():

		if word not in stop_words:

			# Let's Lemmatize the word as well
			# before appending to the imp_words list.

			lemmatizer = WordNetLemmatizer()
			lemmatizer.lemmatize(word)

			imp_words.append(word)

	output = " ".join(imp_words)

	return output


df['tweet'] = df['tweet'].apply(lambda text: remove_stopwords(text))
df.head()
def plot_word_cloud(data, typ):
# Joining all the tweets to get the corpus
email_corpus = " ".join(data['tweet'])

plt.figure(figsize = (10,10))

# Forming the word cloud
wc = WordCloud(max_words = 100,
				width = 200,
				height = 100,
				collocations = False).generate(email_corpus)

# Plotting the wordcloud obtained above
plt.title(f'WordCloud for {typ} emails.', fontsize = 15)
plt.axis('off')
plt.imshow(wc)
plt.show()
print()

plot_word_cloud(df[df['class']==2], typ='Neither')
class_2 = df[df['class'] == 2]
class_1 = df[df['class'] == 1].sample(n=3500)
class_0 = df[df['class'] == 0]

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)
plt.pie(balanced_df['class'].value_counts().values,
		labels=balanced_df['class'].value_counts().index,
		autopct='%1.1f%%')
plt.show()
