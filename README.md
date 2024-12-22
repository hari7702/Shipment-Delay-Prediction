# Shipment Delay Prediction

This project is a *Shipment Delay Prediction System* designed to predict whether a shipment will be delayed or delivered on time. The project combines machine learning, Flask API development, and a user-friendly web interface for real-time predictions.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [How to Set Up the Project](#how-to-set-up-the-project)
5. [How to Use the Project](#how-to-use-the-project)
6. [Directory Structure](#directory-structure)
7. [API Endpoints](#api-endpoints)
8. [Web Interface](#web-interface)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

The Shipment Delay Prediction project helps logistics companies forecast shipment delays using historical data and external factors such as weather, traffic, and shipment distance. This prediction enables better planning and decision-making.

---

## Features

- *Machine Learning*: Uses a Random Forest model for high accuracy.
- *RESTful API*: Built with Flask to provide predictions via /predict endpoint.
- *Web Interface*: A user-friendly interface to input shipment details and get predictions in real-time.
- *Deployable*: Can be deployed locally or on cloud platforms.

---

## Technologies Used

- *Python*: Core programming language for data processing and model building.
- *Flask*: API development and backend server.
- *HTML/CSS/JavaScript*: Frontend web interface.
- *Scikit-learn*: Machine learning models.
- *Pandas & NumPy*: Data preprocessing and manipulation.
- *Matplotlib*: Visualizations for exploratory data analysis.

---

## How to Set Up the Project

### Prerequisites
1. Install *Python (>=3.8)*.
2. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt

Installation

1. Clone this repository:

git clone [https://github.com/your_username/shipment-delay-prediction.git](https://github.com/hari7702/Shipment-Delay-Prediction)
cd shipment-delay-prediction


2. Place your random_forest_model.pkl and scaler.pkl files in the root directory.


3. Run the Flask server:

python app.py


4. Open your browser and go to:

API: http://127.0.0.1:5000/predict

Web Interface: http://127.0.0.1:5500/index.html





---

How to Use the Project

API Usage

1. Use tools like Postman or Python’s requests library to make POST requests to:

Endpoint: /predict

Example Input:

{
    "Origin": 3,
    "Destination": 2,
    "Vehicle Type": 1,
    "Distance (km)": 1500,
    "Weather Conditions": 1,
    "Traffic Conditions": 2,
    "Delivery Delay": 3
}

Example Output:

{
    "prediction": "Delayed"
}




Web Interface

1. Access the interface at http://127.0.0.1:5500/index.html.


2. Fill in the shipment details in the form fields.


3. Click the Predict button to see the prediction.




---

API Endpoints

/predict (POST)

Description: Accepts shipment details in JSON format and returns the prediction.

Input:

{
    "Origin": 3,
    "Destination": 2,
    "Vehicle Type": 1,
    "Distance (km)": 1500,
    "Weather Conditions": 1,
    "Traffic Conditions": 2,
    "Delivery Delay": 3
}

Output:

{
    "prediction": "Delayed"
}



---

Web Interface

The web interface allows users to:

Input shipment details through a simple form.

Submit the form to make a prediction using the Flask API.

View the prediction directly on the page.



---

Results

Accuracy: The Random Forest model achieved 99% accuracy on the test data.

Impact: Predicting delays improves delivery planning, reduces costs, and enhances customer satisfaction.



---

Contributing

Feel free to contribute to this project by submitting issues or pull requests. Make sure to follow the contribution guidelines.


---

License

This project is licensed under the MIT License. See the LICENSE file for details.


---

Happy Predicting!
