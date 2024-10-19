import joblib
import json
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import warnings
import re

# Suppress specific InconsistentVersionWarning from scikit-learn
warnings.filterwarnings("ignore", message="Trying to unpickle estimator TfidfTransformer", category=UserWarning)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator TfidfVectorizer", category=UserWarning)

class ErrorClassifier:
    def __init__(self, model_path):
        # Load the trained model components (DataFrame, vectorizer, tfidf_matrix)
        self.df, self.vectorizer, self.tfidf_matrix = joblib.load(model_path)

    def preprocess_text(self, text):
        # Replace any custom tokens in the text with regex-like patterns
        return text.replace('<*>', '.*')

    def fuzzy_match_score(self, str1, str2):
        # Use rapidfuzz to calculate the fuzzy match ratio between two strings
        return fuzz.ratio(str1, str2) / 100.0

    def semantic_similarity(self, query):
        # Use TF-IDF vectorizer to transform the query and compute cosine similarity
        query_vector = self.vectorizer.transform([query])
        return cosine_similarity(query_vector, self.tfidf_matrix).flatten()

    def combined_similarity(self, query, word_weight=0.8, semantic_weight=0.2):
        # Preprocess query and compute both fuzzy match and semantic similarity scores
        query = self.preprocess_text(query)
        fuzzy_scores = np.array([self.fuzzy_match_score(query, event) for event in self.df['EventTemplate']])
        semantic_scores = self.semantic_similarity(query)
        # Return the weighted combination of fuzzy and semantic similarity scores
        return word_weight * fuzzy_scores + semantic_weight * semantic_scores

    def find_top_matches(self, query, n=1):
        # Get combined similarity scores for the query and find the top N matches
        combined_scores = self.combined_similarity(query)
        top_indices = combined_scores.argsort()[-n:][::-1]
        return self.df.iloc[top_indices]

class ErrorAnalysisSystem:
    def __init__(self, model_path, api_key):
        # Initialize the ErrorClassifier and configure Google Generative AI
        self.classifier = ErrorClassifier(model_path)
        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
        self.genai_model = genai.GenerativeModel('gemini-1.5-pro')

    def generate_ai_response(self, error_message, model_output):
        # Prepare the prompt for the generative AI model
        prompt = f"""
        # Error Analysis Prompt Template

        You are an AI assistant specialized in analyzing error messages and providing detailed insights. Given an error message and its classification, you need to provide a comprehensive analysis in JSON format. Here's an example followed by a new query:

        ## Example:

        Input Error Message: "10.251.35.1:50010:Got exception while serving blk_7940316270494947483 to /10.251.122.38:"

        Model Output:
        ```
        Unnamed: 0 Level Component EventTemplate type
        789 WARN dfs.DataNode Got exception while serving blk_<*> to /<*> HDFS
        ```

        Expected Output:
        {{
          "analysis": {{
            "coreIssue": "An exception occurred while the DataNode was serving a specific block to a client."
          }},
          "classification": "HDFS",
          "severity": "WARN",
          "likelyCause": "There could be network issues, disk I/O problems, or the block might be corrupted. It's also possible that the client disconnected unexpectedly during data transfer.",
          "suggestedSolution": [
            "Check the DataNode logs for more detailed error messages",
            "Verify the health of the HDFS cluster",
            "Ensure that the block is not corrupted by running fsck",
            "Check network connectivity between the DataNode and the client"
          ],
          "tips": [
            "Regularly monitor DataNode health and performance",
            "Implement proper error handling and retry mechanisms in HDFS clients",
            "Keep HDFS software up-to-date to benefit from bug fixes and performance improvements"
          ],
          "actionableRecommendations": [
            "Run 'hdfs fsck' to check for any corrupted blocks",
            "Review DataNode logs for any recurring issues or patterns",
            "Monitor network performance between DataNodes and clients",
            "Consider increasing the number of replicas for important data to improve fault tolerance"
          ]
        }}

        ## New Query:

        Input Error Message: "{error_message}"

        Model Output:
        {model_output}

        Based on the input error message and model output, provide a detailed analysis in the same JSON format as the example above. Include relevant information for all fields: analysis, classification, severity, likelyCause, suggestedSolution, tips, and actionableRecommendations. Do not include any markdown formatting or code blocks in your response, just the raw JSON object.
        """

        # Generate the AI response
        result = self.genai_model.generate_content(prompt)

        # Clean up the result and extract JSON
        cleaned_result = result.text.strip()
        json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
        
        if json_match:
            try:
                ai_response = json.loads(json_match.group())
                return ai_response
            except json.JSONDecodeError as e:
                print(f'Failed to parse AI response as JSON: {e}')
                return {"error": "Invalid JSON format from AI", "rawResponse": cleaned_result}
        else:
            print('No JSON object found in AI response')
            return {"error": "No JSON object found in AI response", "rawResponse": cleaned_result}

    def process_error(self, error_message):
        # Find the top matches for the error message using the classifier
        top_matches = self.classifier.find_top_matches(error_message)
        model_output = top_matches[['Level', 'Component', 'EventTemplate', 'type']].to_dict(orient='records')

        # Generate AI response based on the error message and model output
        ai_response = self.generate_ai_response(error_message, json.dumps(model_output))

        # Return the final result including the error message, model classification, and AI analysis
        return {
            "errorMessage": error_message,
            "modelClassification": model_output,
            "aiAnalysis": ai_response
        }