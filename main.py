import os
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.features import create_features
from src.model import train_model, predict
from src.evaluate import evaluate_model
from src.visualize import plot_results

# Absolute path (fixes FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "energy.csv")

# Load data
df = load_data(data_path)

# Preprocess
df = preprocess_data(df)

# Feature engineering
df = create_features(df)

# Train-test split
split_index = int(len(df) * 0.8)
train = df[:split_index]
test = df[split_index:]

# Train model
model = train_model(train)

# Predict
predictions = predict(model, test)

# Evaluate
evaluate_model(test['energy'], predictions)

# Save predictions (IMPORTANT for GitHub proof)
output_path = os.path.join(BASE_DIR, "outputs", "predictions.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

test_result = test.copy()
test_result['predicted'] = predictions
test_result.to_csv(output_path, index=False)

print(f"Predictions saved at: {output_path}")

# Plot
image_path = os.path.join(BASE_DIR, "images", "result.png")
os.makedirs(os.path.dirname(image_path), exist_ok=True)

plot_results(test['energy'], predictions, image_path)