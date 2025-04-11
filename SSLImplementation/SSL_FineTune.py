import tensorflow as tf
from Utils.Data_Loader import load_labeled_data
from Utils.SSL_Models import add_classification_head
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load pre-trained encoder
encoder = tf.keras.models.load_model("models/ssl_pretrained_encoder.keras")

# Add classification head for IDS
model = add_classification_head(encoder, num_classes=2)  # 0=normal, 1=attack

# Load labeled data (small subset)
(X_train, y_train), (X_val, y_val) = load_labeled_data(config["labeled_data_path"])

# Fine-tune
model.compile(
    optimizer=tf.keras.optimizers.Adam(config["finetune_lr"]),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=config["finetune_epochs"],
    batch_size=config["finetune_batch_size"]
)

# Save final IDS model
model.save("models/ssl_finetuned_ids_model.keras")
print(" Fine-tuning complete!")