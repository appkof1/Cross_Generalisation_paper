import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split

from build_model import build_model
from utils.load_data import load_ghana_dataset
from utils.preprocess_data import augment_data

# Load datasets
spectra_X, spectra_y = load_ghana_dataset('spectra', 'ultrasound')
dcl_X, dcl_y = load_ghana_dataset('dcl', 'ultrasound')
X = np.concatenate((spectra_X, dcl_X), axis=0)
y = np.concatenate((spectra_y, dcl_y), axis=0)

# Split training, validation, and test sets
X_train_temp, X_test_preaug, y_train_temp, y_test_preaug = train_test_split(X, y, test_size=0.15, random_state=42)
X_train_preaug, X_val_preaug, y_train_preaug, y_val_preaug = train_test_split(X_train_temp, y_train_temp, test_size=0.18, random_state=42)

# Augment training, validation, and test sets separately
X_train, y_train = augment_data(X_train_preaug, y_train_preaug)
X_val, y_val = augment_data(X_val_preaug, y_val_preaug)
X_test, y_test = augment_data(X_test_preaug, y_test_preaug)

# Build & train model
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'auc'])
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

# Make predictions
y_true = np.argmax(y_test, axis=1)
predict_y = model.predict(X_test)
y_pred = np.argmax(predict_y,axis=1)

# Compute metrics
accuracy = metrics.accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred, average='weighted')
recall = metrics.recall_score(y_true, y_pred, average='weighted')
auc = metrics.roc_auc_score(y_true, predict_y, multi_class='ovr', average='weighted')
df_metrics = pd.DataFrame({'Metrics': [accuracy, precision, recall, auc]})

# Save metrics
df_metrics.to_csv('metrics_report.csv', index=False)