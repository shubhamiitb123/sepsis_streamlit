import streamlit as st
import numpy as np
from keras.models import load_model,Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, BatchNormalization, Flatten, Reshape, Add, Concatenate, LSTM, Masking, GRU
import pandas as pd
input_dim = 40
batch_size = 100
mask_value = -9999.0
# Assuming you've defined batch_size earlier

# Define the model architecture
input = Input(batch_shape=(batch_size, None, input_dim))
x = Masking(mask_value=mask_value)(input)
x = GRU(32, stateful=True, return_sequences=True)(x)
x = GRU(32, stateful=True)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input, outputs=output)

# Load the weights
model.load_weights('model_weights.h5')

# print(model.summary())


# Function to generate predictions from a PSV file
def generate_predictions_from_psv_file(data, model):
    # Read data from PSV file
    

    # Preprocess the data (similar to what was done during training)
    data[np.isnan(data)] = 0
    slen = data.shape[0]
    batch_n = int(np.ceil(data.shape[0] / batch_size))
    data = np.concatenate([data, np.full([batch_n * batch_size - data.shape[0], data.shape[1]], mask_value)])
    data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

    # Use the trained model to predict the output
    predictions = model.predict(data[:, :, :-1], batch_size=batch_size, verbose=0)
    predictions = [predictions[i][0] for i in range(slen)]

    # Post-process the predictions if necessary

    return predictions

# Load the trained model

# Streamlit UI
st.title('Sepsis Prediction App')

# File uploader
uploaded_file = st.file_uploader("Upload a PSV file", type=["psv"])

import pandas as pd


if uploaded_file is not None:
    # Display uploaded file
    ip=pd.read_csv(uploaded_file,sep='|')
    st.write(ip.head())
    st.write('File Uploaded Successfully!')

    # Button to generate predictions
    if st.button('Generate Predictions'):
        # Generate predictions from the uploaded file
        data=ip.values
        predictions = generate_predictions_from_psv_file(data, model)
        
        
        # Display predictions
        st.write('Prediction Probabilities for Each Timestamp:')

        df=pd.DataFrame({'pred':predictions})
        st.write(df)
