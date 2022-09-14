import math
from numpy import int64
import pandas
import warnings
from sklearn.model_selection import train_test_split
from src.rscf import RecommenderEngine, evaluator
from src.rscf.algorithms import SVD

# Ignore deprecation warning from numpy
warnings.simplefilter('ignore', FutureWarning)

# Load articles and interactions data as DataFrame object
articles = pandas.read_json('articles.json')
interactions = pandas.read_json('interactions.json').set_index('user_id')

interactions = interactions \
    .groupby(['user_id', 'article_id'])['weight'].sum() \
    .apply(lambda x: math.log(1 + x, 2)) \
    .reset_index()

# Split the dataset into training and testing set using hold-out method.
interactions_train, interactions_test = train_test_split(
    interactions,
    stratify=interactions['user_id'],
    test_size=0.20,
    random_state=42
)

interactions = interactions.set_index('user_id')
interactions_train = interactions_train.set_index('user_id')
interactions_test = interactions_test.set_index('user_id')

# Instantiate the engine, feed your data, and begin the training
engine = RecommenderEngine(articles, interactions_train) \
    .set_algorithm(SVD) \
        .train()

# evaluator = evaluator.recall(
#     interactions=interactions, 
#     interaction_train=interactions_train, 
#     interaction_test=interactions_test,
#     trained_engine=engine)

# print(evaluator)

while True:
    try:
        user_id = int(input('User ID: '))
    except ValueError:
        print('Input is not a number')
        continue
    
    rec_articles = pandas.DataFrame(engine.get_recommendation(user_id, top_n=10), columns=['article_id', 'strength'])
    rec_articles['article_id'] = rec_articles['article_id'].astype(int64)
    print(rec_articles)