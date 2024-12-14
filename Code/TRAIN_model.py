from CNN_model import define_model
from config import DATASETgen 
from Process_data import load_data
import tensorflow as tf

train_data, val_data, total_train_samples, total_val_samples = load_data(
    DATASETgen, 
    target_size=(32, 32), 
    batch_size=32,        
    num_classes=2         
)

for x, y in train_data.take(1):
    print(f"Input batch shape: {x.shape}, Label batch shape: {y.shape}")

model = define_model(input_shape=(32, 32, 3), num_classes=2)

batch_size = 32
steps_per_epoch = (total_train_samples + batch_size - 1) // batch_size  
validation_steps = (total_val_samples + batch_size - 1) // batch_size  

train_data = train_data.prefetch(tf.data.AUTOTUNE) 
val_data = val_data.prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_data,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=validation_steps
)

model.save('model.keras')  
