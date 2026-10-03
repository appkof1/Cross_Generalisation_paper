import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

from build_model import build_model
from utils.preprocess_data import balance_datasets, augment_data

def train_and_test(i, setA_X, setA_y, setB_X, setB_y):
    """
    Runs a single iteration of the bias experiment. Splits Dataset A into training, 
    validation, and test data. Balances Test Set A against Dataset B to produce two
    balanced test sets. Applies image augmentation to all sets seperately.

    Trains the model with Dataset A training and validation data. Tests the model with
    balanced and unbalanced Dataset A test data, and with Dataset B test data.
    Computes metrics for each test set. 

    Args:
        i: Iteration number
        setA_X (NumPy array): Array of images in Dataset A, each image has shape (224, 224, 3)
        setA_y (NumPy array): Array of labels in Dataset A, one-hot encoded as [0, 0, 1]
        setB_X (NumPy array): Array of images in Dataset B, each image has shape (224, 224, 3)
        setB_y (NumPy array): Array of labels in Dataset B, one-hot encoded as [0, 0, 1]

    Returns:
        Pandas DataFrame: DataFrame containing performance metrics for balanced Test Set A
        Pandas DataFrame: DataFrame containing performance metrics for unbalanced Test Set A
        Pandas DataFrame: DataFrame containing performance metrics for Test Set B
    """
    # Split Set A into training, validation, and test sets
    X_train_temp, setA_X_test_prebalance_preaug, y_train_temp, setA_y_test_prebalance_preaug = train_test_split(setA_X, setA_y, test_size=0.15, random_state=42)
    setA_X_train_preaug, setA_X_val_preaug, setA_y_train_preaug, setA_y_val_preaug = train_test_split(X_train_temp, y_train_temp, test_size=0.18, random_state=42)

    # Balance A & B test sets
    setA_X_test_preaug, setA_y_test_preaug, setB_X_test_preaug, setB_y_test_preaug = balance_datasets(setA_X_test_prebalance_preaug, setA_y_test_prebalance_preaug, setB_X, setB_y)

    # Augment training, validation, and test sets separately
    setA_X_train, setA_y_train = augment_data(setA_X_train_preaug, setA_y_train_preaug)
    setA_X_val, setA_y_val = augment_data(setA_X_val_preaug, setA_y_val_preaug)
    setA_X_test, setA_y_test = augment_data(setA_X_test_preaug, setA_y_test_preaug)
    setB_X_test, setB_y_test = augment_data(setB_X_test_preaug, setB_y_test_preaug)
    setA_X_prebal_test, setA_y_prebal_test = augment_data(setA_X_test_prebalance_preaug, setA_y_test_prebalance_preaug)
    
    # Build & train model
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'auc'])
    model.fit(setA_X_train, setA_y_train, epochs=20, batch_size=16, validation_data=(setA_X_val, setA_y_val))

    # Make predictions
    setA_y_true = np.argmax(setA_y_test, axis=1)
    setA_predict_y = model.predict(setA_X_test)
    setA_y_pred = np.argmax(setA_predict_y,axis=1)

    setB_y_true = np.argmax(setB_y_test, axis=1)
    setB_predict_y = model.predict(setB_X_test)
    setB_y_pred = np.argmax(setB_predict_y,axis=1)

    # Predict pre-balanced datasets (control)
    setAprebal_y_true = np.argmax(setA_y_prebal_test, axis=1)
    setAprebal_predict_y = model.predict(setA_X_prebal_test)
    setAprebal_y_pred = np.argmax(setAprebal_predict_y,axis=1)

    # Compute metrics
    setA_metrics = _compute_metrics(i+1, setA_predict_y, setA_y_true, setA_y_pred)
    setB_metrics = _compute_metrics(i+1, setB_predict_y, setB_y_true, setB_y_pred)
    setAprebal_metrics = _compute_metrics(i+1, setAprebal_predict_y, setAprebal_y_true, setAprebal_y_pred)

    return setA_metrics, setB_metrics, setAprebal_metrics

def _compute_metrics(iteration, predict_y, y_true, y_pred):
    # Compute metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    auc = metrics.roc_auc_score(y_true, predict_y, multi_class='ovr', average='weighted')
    
    # Convert to DF to store as CSV
    title = 'Iteration ' + str(iteration)
    df_metrics = pd.DataFrame({title: [accuracy, precision, recall, auc]})

    return df_metrics