import tensorflow as tf
from tensorflow.keras import layers
from Utils.Data_Loader import load_unlabeled_data
from Utils.SSL_Models import create_ssl_encoder
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load unlabeled data (e.g., normal network traffic)
unlabeled_data = load_unlabeled_data(config["unlabeled_data_path"])

# Create SSL model
encoder = create_ssl_encoder(input_shape=(100,))  # Replace with your feature size
ssl_model = tf.keras.Sequential([
    encoder,
    layers.Dense(64, activation='relu'),  # Projection head
    layers.Dense(32)                      # Latent space
])

# Contrastive loss (SimCLR-style)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ssl_model.compile(optimizer='adam', loss=loss_fn)

# Train on unlabeled data
ssl_model.fit(
    unlabeled_data,
    epochs=config["ssl_epochs"],
    batch_size=config["ssl_batch_size"]
)

# Save pre-trained encoder
encoder.save("models/ssl_pretrained_encoder.keras")
print(" SSL pre-training complete!")