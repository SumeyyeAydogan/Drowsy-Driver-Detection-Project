import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC

callbacks = [
  EarlyStopping(patience=5, restore_best_weights=True),
  ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6),
  ModelCheckpoint('best_model.h5', save_best_only=True)
]
def train_model(model, train_ds, val_ds, epochs=50):
    model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-4),
      loss='binary_crossentropy',
      metrics=[
          BinaryAccuracy(name='accuracy'),
          Precision(name='precision'),
          Recall(name='recall'),
          AUC(name='auc')
        ]
    )
    '''
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=callbacks
    )
    '''
    train_small = train_ds.take(4)  # 4*16=64
    val_small = val_ds.take(4)
    history = model.fit(train_small, epochs=4, validation_data=val_small, class_weight=None)
    return history
