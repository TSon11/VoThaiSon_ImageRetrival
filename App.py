import streamlit as st
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract HOG features from an image
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    # Resize the image to reduce computational load
    image_resized = cv2.resize(image, (128, 128))
    hog_features = hog.compute(image_resized)
    return hog_features.flatten()

# Function to adjust the size of HOG feature vectors
def adjust_hog_features_size(features, target_length):
    current_length = len(features)
    if current_length < target_length:
        # If the current length is shorter than the target length, pad zeros to the end
        features = np.pad(features, (0, target_length - current_length), 'constant')
    elif current_length > target_length:
        # If the current length is longer than the target length, truncate the vector
        features = features[:target_length]
    return features

# Function to load images from a folder
def load_images(folder_path):
    images = []
    names = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                # Extract the name before the underscore
                name = filename.split('_')[0]
                names.append(name)
    return images, names

# Main function to run Streamlit app
def main():
    st.title("CBIR")
    st.markdown("Vo Thai Son - 21521388")

    # Select the query image from the query folder
    st.write("Chọn ảnh truy vấn:")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if uploaded_file is not None:
        # Read the uploaded image
        uploaded_name = uploaded_file.name.split('_')[0]
        query_image = np.array(cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1))
        st.image(query_image, caption="Ảnh truy vấn", use_column_width=True)

        # Extract HOG features from the query image
        query_features = extract_hog_features(query_image)
        
        # Adjust the size of HOG feature vector
        target_length = 1000  # You can adjust this value according to your needs
        query_features = adjust_hog_features_size(query_features, target_length)

        # Load images from the database folder
        database_folder = 'CBIR/database'
        database_images, database_names = load_images(database_folder)

        # Calculate true positives and false negatives based on the query image
        true_positives = 0
        false_negatives = 0
        for name in database_names:
            if name == uploaded_name:
                true_positives += 1
            else:
                false_negatives += 1

        # Calculate cosine similarity between query features and database images
        similarities = []
        for image in database_images:
            database_features = extract_hog_features(image)
            database_features = adjust_hog_features_size(database_features, target_length)
            similarity = cosine_similarity([query_features], [database_features])[0][0]
            similarities.append(similarity)

        # Get top similar images based on user's choice
        top_k = st.selectbox("Chọn số lượng top ảnh hiển thị:", [3, 5, 10], index=2)

        # Get top k similar images
        top_similar_indices = np.argsort(similarities)[::-1][:top_k]

        # Display top k similar images and calculate precision, recall, and AP
        true_positive_count = 0
        precision_scores = []
        recall_scores = []
        count = 0
        for index in top_similar_indices:
            count += 1
            if database_names[index] == uploaded_name:
                true_positive_count += 1
            recall_score = true_positive_count / true_positives if true_positives != 0 else 0
            recall_scores.append(recall_score)
            precision_score = true_positive_count / (true_positive_count + (count - true_positive_count))
            precision_scores.append(precision_score)

            st.image(database_images[index], caption=f"Độ tương đồng: {similarities[index]:.2f}", use_column_width=True)
            st.write(f"Precision: {precision_score:.2f}")
            st.write(f"Recall: {recall_score:.2f}")

        # Calculate AP
        sum_tmp = 0
        arr_length = len(precision_scores)
        for i in range(0, arr_length):
            sum_tmp += precision_scores[i]
        AP_score = sum_tmp / arr_length
        st.write(f"AP Score: {AP_score:.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
