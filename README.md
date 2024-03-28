# Car Price Prediction
This project aims to predict the prices of used cars based on various features such as the car's name, company, manufacturing year, kilometers driven, and fuel type. The dataset used for training and testing the machine learning models contains information about different cars along with their corresponding prices.

## Dataset
The dataset used for this project is stored in the file `cars_dataset.csv`. It contains the following columns:

- `name`: Name of the car.
- `company`: Company or manufacturer of the car.
- `year`: Manufacturing year of the car.
- `Price`: Price of the car (target variable).
- `kms_driven`: Total kilometers driven by the car.
- `fuel_type`: Type of fuel used by the car (e.g., petrol, diesel).

## Data Preprocessing
- Removed entries with missing or invalid values.
- Converted categorical variables into numerical format using one-hot encoding.
- Cleaned and formatted numerical variables.
- Split the dataset into training and testing sets.

## Model Training
- Utilized linear regression for predicting car prices.
- Implemented a pipeline to preprocess data and fit the model.
- Evaluated the model's performance using the R-squared (R2) score.

## Results
The trained linear regression model achieved an R-squared score of approximately 0.92 on the test set, indicating a good fit to the data. This model can be used to predict the prices of used cars based on their features.

## Usage
1. Ensure you have the required Python libraries installed by running: 
   ```
   pip install pandas numpy matplotlib scikit-learn
   ```
2. Clone the repository to your local machine: 
   ```
   git clone https: //github.com/hassan3014/Car-Price-Prediction.git
   ```
3. Run the Jupyter Notebook `car_price_prediction.ipynb` to train the model and analyze the results.

## Files
- `car_price_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `cars_dataset.csv`: Dataset containing information about used cars.

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn

## Contact
For any inquiries or support, please contact [hassan.malik1574@gmail.com].
