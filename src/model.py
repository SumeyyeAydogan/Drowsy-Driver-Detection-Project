# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,
                                     BatchNormalization, Dropout,
                                     GlobalAveragePooling2D, Dense,
                                     Add, Multiply, Activation,
                                     GlobalMaxPooling2D, Concatenate)
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(224, 224, 3), weight_decay=1e-4):
    """
    Enhanced Deep CNN architecture for binary classification:
      - 5 convolutional blocks with increasing filters
      - Spatial attention mechanism
      - Residual connections
      - Enhanced classifier with multiple dense layers
      - Advanced regularization techniques
    """
    from tensorflow.keras import Model, Input
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # --- Block 1: Initial Feature Extraction ---
    x = Conv2D(32, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # --- Block 2: Deeper Features ---
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # --- Block 3: Mid-level Features ---
    x = Conv2D(128, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # --- Block 4: High-level Features ---
    x = Conv2D(256, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # --- Block 5: Deep Features with Spatial Attention ---
    # Spatial attention mechanism
    attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = Multiply()([x, attention])
    
    x = Conv2D(512, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same',
               kernel_regularizer=l2(weight_decay),
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    # --- Global Feature Aggregation ---
    # Use both GlobalAveragePooling and GlobalMaxPooling
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    
    # Concatenate both pooling methods
    x = Concatenate()([gap, gmp])
    
    # --- Enhanced Classifier Head ---
    x = Dense(512, kernel_regularizer=l2(weight_decay),
              activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, kernel_regularizer=l2(weight_decay),
              activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, kernel_regularizer=l2(weight_decay),
              activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


if __name__ == "__main__":
    # Quick summary:
    m = build_model()
    m.summary()
