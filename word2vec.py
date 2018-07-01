import pickle
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


with open('df_train_api.pk', 'rb') as pklfile:
    df = pickle.load(pklfile)

sentences = []

for i, row in df.iterrows():
    text = row['type'][0].strip()
    sentences.append(text.split(' '))

model = Word2Vec(sentences, size=300, window=5, min_count=3, workers=4)
model.save('model.bin')
print(model)

X = model[model.wv.vocab]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
