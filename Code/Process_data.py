from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(train_dir, val_dir, target_size=(32, 32), batch_size=8):  
    datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size, 
        class_mode="categorical"  
    )
    
    val_data = datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size, 
        class_mode="categorical"
    )
    
    return train_data, val_data