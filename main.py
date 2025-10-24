# ======================================
# Step 1: Imports
# ======================================
import pandas as pd
import numpy as np
import os
from src.utils import download_images
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm


# ======================================
# Step 2: Main Code
# ======================================
if __name__ == "__main__":
    # --------------------------
    # Load CSV files
    # --------------------------
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df["catalog_content"] = train_df["catalog_content"].fillna("")
    test_df["catalog_content"] = test_df["catalog_content"].fillna("")

    download_images(train_df["image_link"].tolist(), "images/train/")
    download_images(test_df["image_link"].tolist(), "images/test/")

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    # --------------------------
    # TF-IDF
    # --------------------------
    tfidf = TfidfVectorizer(max_features=5000)
    X_text_train = tfidf.fit_transform(train_df['catalog_content']).toarray()
    X_text_test = tfidf.transform(test_df['catalog_content']).toarray()

    print("Text features shape:", X_text_train.shape)

    # --------------------------
    # Image Feature Extraction (ResNet50)
    # --------------------------
    resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def batch_extract_features(img_paths, batch_size=32):
        features = []
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Extracting image features"):
            batch_imgs = []
            for path in img_paths[i:i+batch_size]:
                if not os.path.exists(path):
                    batch_imgs.append(np.zeros((224,224,3)))
                else:
                    img = image.load_img(path, target_size=(224,224))
                    x = image.img_to_array(img)
                    batch_imgs.append(x)
            x_batch = preprocess_input(np.array(batch_imgs))
            feats = resnet.predict(x_batch, verbose=0)
            features.append(feats)
        return np.vstack(features)

    train_img_paths = [f"images/train/{sid}.jpg" for sid in train_df['sample_id']]
    test_img_paths = [f"images/test/{sid}.jpg" for sid in test_df['sample_id']]

    X_img_train = batch_extract_features(train_img_paths)
    X_img_test = batch_extract_features(test_img_paths)

    print("Image features shape:", X_img_train.shape)

    # --------------------------
    # Combine Text + Image Features
    # --------------------------
    X_train_full = np.hstack([X_text_train, X_img_train])
    X_test = np.hstack([X_text_test, X_img_test])
    y_train_full = train_df['price'].values

    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    print("Combined features shape:", X_train_full.shape)

    # --------------------------
    # Train/Validation Split
    # --------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # --------------------------
    # SMAPE Metric
    # --------------------------
    def smape(y_true, y_pred):
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0
        return np.mean(diff) * 100

    # --------------------------
    # Train XGBoost
    # --------------------------
    model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mae" 
    )

    print("Training model...")
    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
    )

    # --------------------------
    # Evaluate
    # --------------------------
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    smape_val = smape(y_val, y_val_pred)

    print(f"\nValidation MAE: {mae:.4f}")
    print(f"Validation SMAPE: {smape_val:.2f}%")

    # --------------------------
    # Predict on Test Set
    # --------------------------
    y_pred = model.predict(X_test)

    # --------------------------
    # Save Output
    # --------------------------
    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": y_pred
    })
    submission.to_csv("test_out.csv", index=False)
    print("✅ Submission saved to test_out.csv")
