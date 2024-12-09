from CNN_model import define_model
from config import TRAIN_DIR, VAL_DIR
from CNN_model import define_model
from Process_data import load_data

train_data, val_data = load_data(TRAIN_DIR, VAL_DIR)

model = define_model(input_shape=(32, 32, 3), num_classes=2)

history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

#model.save('model.h5')  asta e legacy conform terminalului
model.save('model.keras')