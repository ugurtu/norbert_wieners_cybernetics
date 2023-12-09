import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a dict of label_encoders
label_encoders = {}

# Define a model with the Class DecisionTreeClassifier
model = DecisionTreeClassifier()

# Define a list of features
features = ['genre', 'director', 'stars', 'year', 'runtime']
target = 'metascore'  # Changed to 'rating_category'
data = pd.read_csv('../v2/data/movies_clean.csv', low_memory=False)
print(len(data))

recommended_movies = []


def categorize_rating(rating):
    if rating < 25.0:
        return 'Low'
    elif 25.0 <= rating < 50.0:  # Ratings from 25 (inclusive) to 50 (exclusive)
        return 'Medium'
    elif 50.0 <= rating < 75.0:  # Ratings from 50 (inclusive) to 75 (exclusive)
        return 'High'
    else:  # Ratings from 75 (inclusive) to 100 (inclusive)
        return 'Very High'


def test_model(model, X, y):
    print("Testing")
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualize the confusion matrix
    if not os.path.exists("evaluation"):
        os.mkdir("evaluation")
    else:
        os.chdir("evaluation")
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual labels')
        plt.xlabel('Predicted labels')
        plt.savefig('confusion_matrix.png')
        plt.show()

    report = classification_report(y_test, y_pred)
    lines = report.split('\n')
    report_data = []
    for line in lines:  # Skip the header and the last few lines
        row = {}
        row_data = line.split()  # Splitting by whitespace
        if len(row_data) >= 5:  # To ensure it's not an empty or malformed line
            row['class'] = row_data[0]
            row['precision'] = row_data[1]
            row['recall'] = row_data[2]
            row['f1_score'] = row_data[3]
            row['support'] = row_data[4]  # Support is typically an integer
            report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report_optimized.csv', index=False)
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=features, class_names=True, rounded=True,
              max_depth=True)  # None to visualize the full tree
    plt.title('Decision Tree (Partial View)')
    plt.savefig("Decision Tree optimized")
    plt.show()
    os.chdir("..")


def train_model():
    # Encoding categorical variables
    for feature in features:
        if data[feature].dtype == 'object':

            le = LabelEncoder()
            data[feature] = le.fit_transform(data[feature].astype(str))
            label_encoders[feature] = le

    # Categorize the 'rating' column
    data['metascore'] = data['metascore'].apply(categorize_rating)

    # Encode the 'rating' column into numerical values
    le_rating = LabelEncoder()
    data['metascore'] = le_rating.fit_transform(data['metascore'])

    # Preparing data for training
    X = data[features]
    y = data[target]

    model.fit(X, y)
    if not os.path.exists("joblib"):
        os.mkdir("joblib")
    else:
        os.chdir("joblib")
        dump(model, 'trained_decision_tree_model.joblib')
        for feature, le in label_encoders.items():
            dump(le, f'label_encoder_{feature}.joblib')
        os.chdir("..")
    test_model(model, X, y)
    return model, label_encoders


# Load or train the model and label encoders
train_model()


def recommend_movie(movie_attributes):
    try:
        processed_preferences = {feature: None for feature in features}
        print(processed_preferences)
        for feature, value in movie_attributes.items():
            if feature in label_encoders:
                le = label_encoders[feature]
                processed_preferences[feature] = le.transform([str(value)])[0]
                print(processed_preferences[feature])
            else:
                processed_preferences[feature] = value

        user_input = pd.DataFrame([processed_preferences])
        predicted_rating = model.predict(user_input)[0]

        # Filter movies with the predicted rating and other attributes
        filtered_movies = data[(data['metascore'] == predicted_rating)]
        for feature, value in movie_attributes.items():
            if feature in label_encoders:
                filtered_movies = filtered_movies[filtered_movies[feature] == processed_preferences[feature]]

        if not filtered_movies.empty:
            recommended_movie = filtered_movies.sample(n=1)['movie_title'].iloc[0]
            counter = 4
            while recommended_movie in recommended_movies:
                recommended_movie = filtered_movies.sample(n=1)['movie_title'].iloc[0]
                counter -= 1
                if counter == 0:
                    recommended_movie = " No other movies "
                    recommended_movies.clear()

            recommended_movies.append(recommended_movie)
            # This is a satabase for storing the movies

            return "I recommend: the movie:" + recommended_movie
        else:
            return "No movie recommendation found for the given attributes."
    except:
        return f"Unseen {feature} in my data."


def retrieve_information(text):
    """
    Processes user text and generates responses.
    """
    # Text preprocessing
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemma_tizer = WordNetLemmatizer()
    tokens = [lemma_tizer.lemmatize(word) for word in tokens]

    # Extracting movie attributes from the input text
    movie_attributes = {}
    print(tokens)
    if 'I want a movie' in tokens:
        return "Great I am a movie recommendation bot, I can recommend you a movie based on your preferences."

    elif ('ivan') in tokens or ('University of Basel') in tokens or ('Pattern recognition') in tokens or ('Cybernetics') in tokens:
        return "I only know Ivan Dokmanic and he is a great professor."

    elif ('ugur') in tokens or ('turhal') in tokens:
        return "Ugur is a great TA."

    elif ('mario') in tokens or ('tachikawa') in tokens:
        return "Mario is a great TA."

    elif ('boston') in tokens or ('dynamics') in tokens:
        print(tokens)
        return "ANYmal beats the robots of boston dynamics."

    else:
        text_elements = text.split(',')
        for element in text_elements:
            key_value = element.split(':')
            if len(key_value) == 2:
                key, value = key_value
                if value.isnumeric():
                    value = int(value)
                movie_attributes[key.strip().lower()] = value.strip()
                print(value.title().strip())
        if movie_attributes:
            return recommend_movie(movie_attributes)
        else:
            return "ChatBot: Please specify movie attributes like Genre, Actor, Year, etc."

print(retrieve_information("boston dynamics"))