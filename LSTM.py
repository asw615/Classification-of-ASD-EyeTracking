import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data - replace 'your_data.csv' with the path to your CSV file
df = pd.read_csv('../data/Eye-tracking Output/combined.csv', low_memory=False)
print("reading csv done")

# Preprocessing
# You may need to convert time columns to a suitable format or drop if not needed
# Normalize the numerical features
scaler = StandardScaler()
features_to_scale = df.columns.difference(['Trial', 'Stimulus', 'Participant', 'Color', 'Category Group', 'Category Right', 'Category Left', 'Annotation Name', 'Annotation Description', 'Annotation Tags', 'Content'])
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("scaling of features done")

# Convert categorical columns to encoded labels if necessary
# For example:
# df['Stimulus'] = df['Stimulus'].astype('category').cat.codes

# Handle missing data if necessary
# df = df.dropna() # or other imputation methods

# Feature selection - Select relevant columns for input to LSTM
# For example, if you want to use pupil sizes and gaze points
features = ['Pupil Diameter Right [mm]', 'Pupil Diameter Left [mm]', 'Point of Regard Right X [px]', 'Point of Regard Right Y [px]']
X = df[features].values
y = df['Participant'].apply(lambda x: 1 if x == 'ASD' else 0).values
print("feature selection done")

# Reshape input to be [samples, time steps, features]
# This depends on how you structure your time-series data
# For example, if each trial is a sample and you have 10 time steps
# X = np.reshape(X, (X.shape[0], 10, -1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions (as probabilities)
predictions = model.predict(X_test)

# You can then round these probabilities to get binary classification
# predicted_classes = (predictions > 0.5).astype(int)
