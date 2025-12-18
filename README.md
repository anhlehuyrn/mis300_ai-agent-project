# MIS300 Project

This is a placeholder README file for the MIS300 project.
It should be updated with relevant information about the project, including:

*   **Project Title:** A clear and concise name for your project.
*   **Description:** A brief overview of what your project does and its purpose.
*   **Features:** A list of key functionalities or capabilities of your project.
*   **Installation:** Instructions on how to set up and run the project locally.
*   **Usage:** Examples or guidelines on how to interact with the project.
*   **Contributing:** Information on how others can contribute to the project (if applicable).
*   **License:** The licensing information for your project.
*   **Contact:** How to reach the project maintainers or team.
*   **Project Structure:**
vizdoom_ppo/
├── Dockerfile                  # Defines the Docker image for the application
├── docker-compose.yml          # Configures the Docker build and service run
├── requirements.txt            # Required Python libraries
├── README.md                   # Project documentation (this file)
├── scenarios/
│   ├── basic.cfg               # Basic scenario configuration for ViZDoom
│   └── basic.wad               # Doom game world (scenario) file
├── vizdoom_ppo/
│   ├── __init__.py             # Marks this as a Python package
│   ├── train.py                # Script to train the agent
│   └── demo.py                 # Script to watch the trained agent play
├── logs/                         # Directory for logs and generated files
│   └── vec_normalize.pkl       # File to save environment normalization statistics
└── vizdoom_ppo_model.zip         # The trained and saved model
