"""
SSVEP Classification Model Architecture
1D CNN for classifying 8Hz vs 14Hz SSVEP responses
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_ssvep_cnn(input_shape, num_classes=2):
    """
    Build 1D CNN for SSVEP classification

    Args:
        input_shape: Tuple (n_frequency_bins, 1)
        num_classes: Number of output classes (default: 2 for binary classification)

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same',
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Second convolutional block
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Third convolutional block
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.4),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lightweight_ssvep_cnn(input_shape, num_classes=2):
    """
    Build a lighter 1D CNN for faster real-time inference

    Args:
        input_shape: Tuple (n_frequency_bins, 1)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(16, kernel_size=5, activation='relu', padding='same',
                     input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Second convolutional block
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_training_callbacks(model_save_path, patience=10):
    """
    Get training callbacks for model optimization

    Args:
        model_save_path: Path to save best model
        patience: Patience for early stopping

    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, model_save_path,
                epochs=100, batch_size=32):
    """
    Train the SSVEP model

    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_save_path: Path to save the model
        epochs: Maximum number of epochs
        batch_size: Batch size

    Returns:
        Training history
    """
    callbacks = get_training_callbacks(model_save_path)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history


def fine_tune_model(base_model_path, X_train, y_train, X_val, y_val,
                    output_model_path, epochs=50, learning_rate=0.0001):
    """
    Fine-tune a pre-trained model on user-specific data

    Args:
        base_model_path: Path to pre-trained model
        X_train: User's training data
        y_train: User's training labels
        X_val: User's validation data
        y_val: User's validation labels
        output_model_path: Path to save fine-tuned model
        epochs: Number of fine-tuning epochs
        learning_rate: Lower learning rate for fine-tuning

    Returns:
        Fine-tuned model and training history
    """
    # Load pre-trained model
    model = keras.models.load_model(base_model_path)

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train on user data
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        output_model_path, epochs=epochs, batch_size=16
    )

    return model, history
