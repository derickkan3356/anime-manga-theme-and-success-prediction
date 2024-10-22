# Anime & Manga Success Prediction Project

## Overview
This project aims to explore the factors that influence the success of anime and manga using machine learning techniques. The dataset is obtained from MyAnimeList and includes user ratings, synopses, titles, and other features. The analysis involves both unsupervised and supervised learning to derive thematic labels and predict user-rated scores.

### Key Objectives:
- Use **unsupervised learning** to derive a single thematic label for anime and manga titles.
- Use **supervised learning** to predict the user scores based on various features, including those derived from topic modeling.

## Project Structure
The project is divided into the following major sections:

1. **Data Source and Feature Engineering**: Pre-process the dataset to combine anime and manga data, handle missing values, and engineer useful features for the model.

2. **Unsupervised Learning**: Use **BERTopic** and other topic modeling approaches to derive a single theme for each title. This step also involves evaluating and tuning hyperparameters to optimize topic coherence.

3. **Supervised Learning**: Train a neural network to predict user scores. We utilized various techniques such as **embedding layers**, **cross-validation**, **Optuna hyperparameter tuning**, and **early stopping** to ensure robustness.

4. **Evaluation and Analysis**: Analyze residuals, feature importances, hyperparameter sensitivity, and learning curves to gain insights into model performance.

## Results
- **Unsupervised Learning**: Derived 9 distinct themes, including **'High School Life and Romance'** and **'Fantasy Adventure / Isekai (alternate world)'**, reflecting popular genres in the dataset.
- **Supervised Learning**: Achieved an RMSE of **0.626** for user score prediction, demonstrating the model's effectiveness in understanding user ratings.

## Requirements
Please refer to `requirements.txt` for a full list of dependencies.