import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import mediapipe as mp

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Load pre-trained ResNet18 model
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final classification layer to get feature embeddings
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()  # Set model to evaluation mode
    return model

# Singleton model instance
_model = None

def get_model_instance():
    global _model
    if _model is None:
        _model = get_model()
    return _model

def extract_palm_with_mediapipe(img):
    """
    Use MediaPipe to detect the hand and extract the palm region from the image.
    Returns a mask with the palm region and the cropped palm image.
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None, None
        shape = img.shape
        if len(shape) == 3:
            h, w = shape[:2]
        else:
            h, w = shape[0], shape[1]
        hand_landmarks = results.multi_hand_landmarks[0]
        # Use only palm landmarks for a tighter crop (landmarks 0, 1, 2, 5, 9, 13, 17)
        palm_indices = [0, 1, 2, 5, 9, 13, 17]
        x_coords = [hand_landmarks.landmark[i].x for i in palm_indices]
        y_coords = [hand_landmarks.landmark[i].y for i in palm_indices]
        x_min = int(min(x_coords) * w) - 20
        x_max = int(max(x_coords) * w) + 20
        y_min = int(min(y_coords) * h) - 20
        y_max = int(max(y_coords) * h) + 20
        # Clamp to image bounds
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, w)
        y_max = min(y_max, h)
        # Create palm mask as a filled rectangle
        palm_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(palm_mask, (x_min, y_min), (x_max, y_max), 255, -1)
        # Crop palm region
        palm_crop = cv2.bitwise_and(img, img, mask=palm_mask)[y_min:y_max, x_min:x_max]
        return palm_mask, palm_crop

def process_palm_lines(img):
    """
    Process palm image to highlight palm lines in the palm region only, using MediaPipe for palm detection.
    """
    palm_mask, _ = extract_palm_with_mediapipe(img)
    if palm_mask is None:
        return np.zeros_like(img)
    # Enhance contrast and detect lines only in palm region
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = cv2.Canny(clahe_img, 30, 100)
    kernel2 = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel2, iterations=1)
    main_lines = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel2, iterations=1)
    secondary_lines = cv2.subtract(dilated, main_lines)
    color_img = np.zeros_like(img)
    color_img[(main_lines > 0) & (palm_mask > 0)] = [0, 0, 255]
    color_img[(secondary_lines > 0) & (palm_mask > 0)] = [0, 255, 0]
    palm_region = cv2.bitwise_and(img, img, mask=palm_mask)
    color_img = cv2.add(color_img, palm_region)
    return color_img

def preprocess_palm_image(image_data):
    """
    Preprocess the palm image for feature extraction, using only the palm region (background removed) via MediaPipe.
    Args:
        image_data (numpy.ndarray): Raw image data
    Returns:
        tuple: (torch.Tensor: Preprocessed image tensor, 
                numpy.ndarray: Original image, 
                numpy.ndarray: Processed image with highlighted palm lines)
    """
    # Decode the image
    original_img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if original_img is None:
        raise ValueError("Failed to decode image")
    # Use MediaPipe to extract palm
    palm_mask, palm_crop = extract_palm_with_mediapipe(original_img)
    if palm_mask is None or palm_crop is None or palm_crop.size == 0:
        raise ValueError("No palm detected in image")
    # Resize to standard size (224x224 for ResNet)
    palm_crop = cv2.resize(palm_crop, (224, 224))
    img_pil = Image.fromarray(cv2.cvtColor(palm_crop, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img_pil)
    img_tensor = img_tensor.unsqueeze(0)
    # For visualization, use process_palm_lines
    processed_img = process_palm_lines(original_img.copy())
    return img_tensor, original_img, processed_img

def extract_features(img_tensor):
    """
    Extract feature embeddings from the preprocessed palm image
    
    Args:
        img_tensor (torch.Tensor): Preprocessed image tensor
        
    Returns:
        numpy.ndarray: Feature embedding vector
    """
    model = get_model_instance()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        features = model(img_tensor)
    
    # Flatten the features
    features = features.squeeze().flatten().numpy()
    
    return features

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1 (numpy.ndarray): First feature embedding
        embedding2 (numpy.ndarray): Second feature embedding
        
    Returns:
        float: Cosine similarity score (0-1)
    """
    # Reshape embeddings to 2D arrays required by sklearn
    embedding1_reshaped = embedding1.reshape(1, -1)
    embedding2_reshaped = embedding2.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1_reshaped, embedding2_reshaped)
    
    return similarity[0][0]

def verify_palm_identity(palm_embedding, claimed_user_id, all_palm_prints, threshold=0.90):
    """
    Strictly verify that the palm print belongs to the claimed user only, by comparing only with the claimed user's registered palm prints.
    Args:
        palm_embedding (numpy.ndarray): Extracted palm embedding
        claimed_user_id (int): User ID claiming to be the owner of the palm print
        all_palm_prints (list): List of palm print database records to check against
        threshold (float): Similarity threshold for authentication
    Returns:
        tuple: (bool, float, int, str) - (is_authenticated, highest_similarity, matched_user_id, error_message)
    """
    if not all_palm_prints:
        logging.error("No registered palm prints found.")
        return False, 0, None, "No registered palm prints found"
    
    if claimed_user_id is None:
        logging.error("No user ID provided.")
        return False, 0, None, "No user ID provided"
    
    # Only compare with the claimed user's palm prints
    user_palm_prints = [p for p in all_palm_prints if p.user_id == claimed_user_id]
    if not user_palm_prints:
        logging.error(f"No palm prints registered for user {claimed_user_id}.")
        return False, 0, None, "No palm prints registered for this user"
    
    highest_similarity = 0
    matched_user_id = claimed_user_id
    is_authenticated = False
    for palm_print in user_palm_prints:
        stored_embedding = np.array(palm_print.get_embedding())
        similarity = calculate_cosine_similarity(palm_embedding, stored_embedding)
        logging.debug(f"Similarity with user {claimed_user_id}: {similarity:.4f}")
        if similarity > highest_similarity:
            highest_similarity = similarity
    
    logging.info(f"Claimed user {claimed_user_id} similarity: {highest_similarity:.4f}")
    if highest_similarity >= threshold:
        is_authenticated = True
        return is_authenticated, highest_similarity, claimed_user_id, "Authentication successful"
    else:
        return False, highest_similarity, claimed_user_id, "Palm print does not match registered palm"

def get_hand_type(palm_embedding):
    """
    Determine if the palm is from left or right hand based on embedding features.
    This is a simplified version - you should implement a more sophisticated method.
    """
    # This is a placeholder - implement actual hand type detection
    # You might want to use specific features or a separate classifier
    return "right"  # or "left"
