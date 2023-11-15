import pandas as pd
import nltk
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load or train the model
def load_or_train_model():
    print("Training a new model.")
    # Load data
    data = pd.read_csv('movies_clean.csv', low_memory=False)
    #    mlb = MultiLabelBinarizer()
    #    encoded_data = mlb.fit_transform(data['stars'])

    # Create a DataFrame from the encoded data
    #    encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_)

    # Join the encoded data with the original DataFrame
    #    data = data.join(encoded_df)
    features = ['genre', 'stars', 'year']
    target = 'movie_title'
    data.to_csv("test.csv")
    # Encoding categorical variables
    label_encoders = {}
    for feature in features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le

    # Preparing data for training
    X = data[features]
    y = data[target]

    # Creating and training the decision tree model
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Save the trained model and label encoders
    dump(model, 'trained_decision_tree_model.joblib')
    for feature, le in label_encoders.items():
        dump(le, f'label_encoder_{feature}.joblib')

    for genre_label in label_encoders['stars'].classes_:
        print(genre_label)

    return model, label_encoders


# Load or train the model and label encoders
model, label_encoders = load_or_train_model()


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
                    return "I do not know this combinaiton"
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
        return recommend_movie(genre,stars,int(year))+"\n"
    else:
        return ("I am chatbot programmed by Ugur and Mario, I just can recommend movies.\n"
                "Please use me like that: \n"
                "Genre,Actor,Year\n")