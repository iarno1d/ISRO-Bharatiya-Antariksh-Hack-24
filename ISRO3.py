import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_broken_parts(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Approximate the contour shape
        shape = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        # Identify the shape
        if len(shape) == 3:
            shape_name = "Triangle"
        elif len(shape) == 4:
            # Determine if the shape is a square or rectangle
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        elif len(shape) > 4:
            shape_name = "Polygon"
        else:
            shape_name = "Circle"
        
        # Put the shape name on the image
        cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the original image with contours and rectangles
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'Moon sam1.png'  # Replace with the path to your image
detect_broken_parts(image_path)
