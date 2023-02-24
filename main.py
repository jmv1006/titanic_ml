import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt


passenger_data = pd.read_csv("./input/train.csv")


mean_age = passenger_data['Age'].mean()
passenger_data["Age_imputated"] = np.where(passenger_data['Age'].isnull(), mean_age, passenger_data['Age'])
passenger_data['Sex_numerical'] = np.where(passenger_data['Sex'] == 'male', 1, 0)

columnsInclude = ["Survived", "Age_imputated", "Fare", "Sex_numerical"]
passenger_data = passenger_data[columnsInclude]

train_dataset = passenger_data.sample(frac=0.8, random_state=0)
test_dataset = passenger_data.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Survived')
test_labels = test_features.pop('Survived')

linear_model = tf.keras.Sequential([
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.005
    ),
    loss="mean_absolute_error"
)

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=500,
    verbose=1,
    validation_split=0.2,
)

competition_data_raw = pd.read_csv("./input/test.csv")
mean_age = competition_data_raw['Age'].mean()

competition_data_raw["Age_imputated"] = np.where(competition_data_raw['Age'].isnull(), mean_age, competition_data_raw['Age'])
competition_data_raw['Sex_numerical'] = np.where(competition_data_raw['Sex'] == 'male', 1, 0)

features = ["Age_imputated", "Fare", "Sex_numerical"]
competition_test_data = competition_data_raw[features]

predictions = linear_model.predict(train_features)

def handleRound(val):
    if val < 0.5:
        return 0
    else: return 1


def generateOuput():
    predictions = [handleRound(x) for x in predictions]

    output = pd.DataFrame({
        "PassengerId": competition_data_raw.PassengerId,
        "Survived": predictions
    })
    output.to_csv("submission.csv", index=False)
