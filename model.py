import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Path to save the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_model.pkl')

class FakeNewsModel:
    """Class for training and using fake news detection model."""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        # Try to load pre-trained model if it exists
        try:
            if os.path.exists(MODEL_PATH):
                self._load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def train(self, texts, labels, test_size=0.2, random_state=42):
        """
        Train the fake news detection model.
        
        Args:
            texts: List of text articles/headlines
            labels: List of labels (1 for fake, 0 for real)
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        
        Returns:
            accuracy: Model accuracy on test set
        """
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        # Create a pipeline with TF-IDF vectorizer and Logistic Regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(C=1.0, max_iter=1000, random_state=random_state))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print performance metrics
        print("Model Performance:")
        print(classification_report(y_test, y_pred))
        
        # Save the model
        self._save_model()
        
        self.is_trained = True
        return accuracy
    
    def predict(self, text):
        """
        Predict if a given text is fake news.
        
        Args:
            text: The news text to classify
        
        Returns:
            prediction: 1 for fake, 0 for real
            probability: Confidence score (probability of the predicted class)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Make prediction
        prediction = self.model.predict([text])[0]
        
        # Get probability
        proba = self.model.predict_proba([text])[0]
        probability = proba[1] if prediction == 1 else proba[0]
        
        return prediction, probability
    
    def _save_model(self):
        """Save the trained model to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, MODEL_PATH)
    
    def _load_model(self):
        """Load a trained model from disk."""
        self.model = joblib.load(MODEL_PATH)
        self.is_trained = True


# Training function that can be called with a dataset
def train_model_with_dataset(dataset_path):
    """
    Train the model with a dataset from a CSV file.
    Expected format: text column, label column (1 for fake, 0 for real)
    
    Args:
        dataset_path: Path to the CSV dataset
    
    Returns:
        model: Trained FakeNewsModel instance
    """
    import pandas as pd
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    
    # Initialize and train model
    model = FakeNewsModel()
    accuracy = model.train(df['text'].tolist(), df['label'].tolist())
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    return model 