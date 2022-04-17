from tabnanny import verbose
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

def NN(X_train, X_test, y_train, y_test, fig):
    nn1 = 64
    nn2 = 32
    # nn3 = 32
    input_dim = len(X_train.columns)
    
    model = Sequential()
    model.add(Dense(nn1, activation='relu', input_dim=input_dim))
    model.add(Dense(nn2, activation='relu'))
    # model.add(Dense(nn3, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=None,
                )

    train_history  = model.fit(X_train, y_train,
            batch_size=50,
            epochs=1000,
            verbose=1  # type: ignore
            )
    
    if fig==1:
        plt.plot(train_history.history['loss'])
        plt.xlabel('epochs')
        plt.ylabel('RMSE')
        plt.show()
        
    return model
    
    
    
    
    
    
    
    

