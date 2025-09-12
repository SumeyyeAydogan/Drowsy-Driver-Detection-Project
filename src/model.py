# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     GlobalAveragePooling2D, Dense)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import Model

def build_model(input_shape=(224, 224, 3), weight_decay=1e-4, use_inception=True):
    """
    Build model with InceptionV3 transfer learning or custom CNN:
      - InceptionV3: Pre-trained on ImageNet, better for face/eye detection
      - Custom CNN: Original architecture for comparison
    """
    
    if use_inception:
        return build_inception_model(input_shape, weight_decay)
    else:
        return build_custom_cnn(input_shape, weight_decay)


def build_inception_model(input_shape=(224, 224, 3), weight_decay=1e-4):
    """
    InceptionV3 transfer learning model for drowsy driver detection
    """
    # Base model - pre-trained on ImageNet
    base_model = InceptionV3(
        input_shape=input_shape,
        include_top=False,  # Remove the original classifier
        weights='imagenet'  # Use pre-trained weights
    )
    
    # Freeze base model layers (optional - can unfreeze for fine-tuning)
    base_model.trainable = False
    
    # Add custom classifier head
    inputs = base_model.input
    x = base_model.output
    
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout
    x = Dense(512, activation='relu', 
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu',
              kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.3)(x)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    return model


def build_custom_cnn(input_shape=(224, 224, 3), weight_decay=1e-4):
    """
    Original custom CNN architecture for comparison
    """
    model = Sequential()

    # --- Block 1 ---
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Block 2 ---
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Block 3 ---
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=l2(weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # --- Classifier Head ---
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(128,
                    kernel_regularizer=l2(weight_decay),
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification: 1 output

    return model


if __name__ == "__main__":
    # Quick summary:
    m = build_model()
    m.summary()
