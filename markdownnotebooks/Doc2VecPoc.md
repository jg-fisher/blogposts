```
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import seaborn as sns
import pandas as pd
```


```
common_texts
```




    [['human', 'interface', 'computer'],
     ['survey', 'user', 'computer', 'system', 'response', 'time'],
     ['eps', 'user', 'interface', 'system'],
     ['system', 'human', 'system', 'eps'],
     ['user', 'response', 'time'],
     ['trees'],
     ['graph', 'trees'],
     ['graph', 'minors', 'trees'],
     ['graph', 'minors', 'survey']]




```
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
documents
```




    [TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]),
     TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]),
     TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]),
     TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]),
     TaggedDocument(words=['user', 'response', 'time'], tags=[4]),
     TaggedDocument(words=['trees'], tags=[5]),
     TaggedDocument(words=['graph', 'trees'], tags=[6]),
     TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]),
     TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]




```
model = Doc2Vec(documents, vector_size=2, window=2, min_count=1, workers=4)
```


```
model.docvecs.vectors_docs
```




    array([[ 0.05046562,  0.16214591],
           [-0.0852842 , -0.14003737],
           [-0.09103508, -0.03952483],
           [ 0.06617976,  0.14716847],
           [-0.24916454, -0.07607701],
           [ 0.00068843,  0.12861392],
           [ 0.20287043, -0.03062222],
           [-0.10221263,  0.1934656 ],
           [ 0.21184532, -0.08516461]], dtype=float32)




```
df = pd.DataFrame(data=model.docvecs.vectors_docs, columns=['x', 'y'])
```


```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.050466</td>
      <td>0.162146</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.085284</td>
      <td>-0.140037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.091035</td>
      <td>-0.039525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.066180</td>
      <td>0.147168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.249165</td>
      <td>-0.076077</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000688</td>
      <td>0.128614</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.202870</td>
      <td>-0.030622</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.102213</td>
      <td>0.193466</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.211845</td>
      <td>-0.085165</td>
    </tr>
  </tbody>
</table>
</div>




```
import matplotlib.pyplot as plt
```


```
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for word, pos in df.iterrows():
    ax.annotate(word, pos)

ax.scatter(df['x'], df['y'])
```




    <matplotlib.collections.PathCollection at 0x7f24f440bcf8>




![png](Doc2VecPoc_files/Doc2VecPoc_8_1.png)

