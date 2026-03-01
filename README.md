# Bilibili Emotion Detection AI (MIS300 Project)

This repository contains the source code for an AI-driven sentiment analysis system designed to detect and categorize user emotions from Bilibili video comments. By leveraging Deep Learning, this project helps creators and researchers understand audience sentiment in real-time.

## 📝 Description
The **Bilibili Emotion Detection AI** focuses on Natural Language Processing (NLP) to process Chinese-language comments. It fetches data from Bilibili, preprocesses the text, and utilizes a Reinforcement Learning-based approach (PPO) or specialized Transformers to classify emotions such as Joy, Anger, Sadness, and Neutrality.

## 🌟 Key Features
* **Real-time Crawling:** Automatically extracts comments from specific Bilibili Video IDs.
* **Emotion Classification:** Detects nuanced emotions beyond simple positive/negative polarities.
* **Dockerized Environment:** Fully containerized setup for consistent performance across different machines.
* **Automated Training Pipeline:** Includes scripts for training the model and normalizing environment statistics.
* **Pre-trained Model:** Comes with a ready-to-use model (`vizdoom_ppo_model.zip`) for immediate testing.

## 📂 Project Structure
```text
vizdoom_ppo/
├── Dockerfile                  # Defines the Docker image for the application
├── docker-compose.yml          # Configures the Docker build and service run
├── requirements.txt            # Required Python libraries (PyTorch, Pandas, etc.)
├── README.md                   # Project documentation
├── scenarios/
│   ├── basic.cfg               # Scenario configuration for the AI environment
│   └── basic.wad               # Environment data file
├── vizdoom_ppo/
│   ├── __init__.py             # Marks this as a Python package
│   ├── train.py                # Script to train the emotion detection agent
│   └── demo.py                 # Script to watch the trained agent in action
├── logs/                       # Directory for training logs and statistics
│   └── vec_normalize.pkl       # Environment normalization statistics
└── vizdoom_ppo_model.zip         # The trained and saved AI model
