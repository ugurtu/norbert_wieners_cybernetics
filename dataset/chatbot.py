import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a dict of label_encoders
label_encoders: dict = {}

# Define a model with the Class DecisionTreeClassifier
model = DecisionTreeClassifier()

#  We define a list of features. The features are:
#  [genre, stars, year, rating, duration, ...].
features: list = ['genre', 'stars', 'year']
target: str = 'movie_title'


def train_model():
    """
    This function trains reads the data from movies_clean.csv.

    """
    # Load data
    data = pd.read_csv('movies_clean.csv', low_memory=False)

    # Encoding categorical variables
    for feature in features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le

    # Preparing data for training
    X = data[features]
    y = data[target]

    model.fit(X, y)

    # Save the trained model and label encoders
    dump(model, 'trained_decision_tree_model.joblib')
    for feature, le in label_encoders.items():
        dump(le, f'label_encoder_{feature}.joblib')

    for genre_label in label_encoders['stars'].classes_:
        print(genre_label)

    return model, label_encoders


# Load or train the model and label encoders
model, label_encoders = train_model()


# Function for movie recommendations
def recommend_movie(genre, stars, year):
    processed_preferences = {}

    # Process each preference
    features = ['genre', 'stars', 'year']
    for feature, value in zip(features, [genre, stars, year]):
        try:
            le = label_encoders.get(feature)
            if le:
                if value in le.classes_:
                    processed_preferences[feature] = le.transform([value])[0]
                else:
                    print(f"Unseen label for {feature}: {value}")
                    return f"ChatBot: Unseen {feature}: {value} in my dataset."
            else:
                processed_preferences[feature] = None
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}")
            return None

    user_input = pd.DataFrame([processed_preferences])
    try:
        recommendation = model.predict(user_input)
        return recommendation[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def retrieve_information(text):
    # Text preprocessing
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Remove punctuation and lowercase
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Example: Simple rule-based response generation
    if 'movie' in tokens or 'actor' in tokens:
        return "It seems like you need assistance. for a movie?"

    elif len(text.split(",")) == 3:
        genre, stars, year = text.split(",")
        genre.replace(" ", "")
        year.replace(" ", "")
        print(genre, stars, year)
        return recommend_movie(genre, stars, int(year)) + "\n"
    else:
        return ("ChatBot: I am chatbot programmed by Ugur and Mario, I just can recommend movies. "
                "Please use me like that: \n"
                "Genre,Actor,Year\n")
