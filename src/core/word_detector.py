"""
Word detection and cropping module using YOLO.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from dataclasses import dataclass
import os


@dataclass
class WordBox:
    """Represents a detected word bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    image: np.ndarray
    index: int  # Sequential index after sorting
    
    @property
    def center_y(self) -> float:
        """Get vertical center of the box."""
        return (self.y1 + self.y2) / 2
    
    @property
    def center_x(self) -> float:
        """Get horizontal center of the box."""
        return (self.x1 + self.x2) / 2
    
    @property
    def area(self) -> int:
        """Get box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class WordDetector:
    """Detects and crops words from images using YOLO."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, target_class_id: int = 0):
        """
        Initialize WordDetector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
            target_class_id: Class ID to filter (0 for word class)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.target_class_id = target_class_id
    
    def detect_and_crop(self, image_path: str) -> List[WordBox]:
        """
        Detect words in an image and return sorted cropped regions.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of WordBox objects sorted in reading order (top-to-bottom, left-to-right)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get predictions
        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            save=False,
            verbose=False
        )
        
        # Extract boxes
        boxes = results[0].boxes
        word_boxes = []
        
        for box in boxes:
            cls_id = int(box.cls[0])
            
            # Filter by class ID
            if cls_id != self.target_class_id:
                continue
            
            # Extract coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop the region
            cropped_img = img[y1:y2, x1:x2]
            
            # Create WordBox object
            word_box = WordBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                image=cropped_img,
                index=-1  # Will be set after sorting
            )
            word_boxes.append(word_box)
        
        # Sort boxes in reading order
        sorted_boxes = self._sort_reading_order(word_boxes)
        
        # Assign sequential indices
        for idx, box in enumerate(sorted_boxes):
            box.index = idx
        
        return sorted_boxes
    
    def _sort_reading_order(self, boxes: List[WordBox]) -> List[WordBox]:
        """
        Sort boxes in reading order: top-to-bottom, left-to-right.
        
        Uses line-based sorting: groups boxes into lines based on vertical overlap,
        then sorts left-to-right within each line.
        
        Args:
            boxes: List of WordBox objects
            
        Returns:
            Sorted list of WordBox objects
        """
        if not boxes:
            return []
        
        # Sort by vertical position first
        boxes_sorted_y = sorted(boxes, key=lambda b: b.center_y)
        
        # Group into lines based on vertical overlap
        lines = []
        current_line = [boxes_sorted_y[0]]
        
        for box in boxes_sorted_y[1:]:
            # Check if box overlaps vertically with current line
            last_box = current_line[-1]
            
            # Calculate vertical overlap threshold (mean height of boxes in current line)
            mean_height = np.mean([b.y2 - b.y1 for b in current_line])
            
            # If centers are within half the mean height, consider same line
            if abs(box.center_y - last_box.center_y) < mean_height * 0.5:
                current_line.append(box)
            else:
                # Start new line
                lines.append(current_line)
                current_line = [box]
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        # Sort each line left-to-right
        sorted_boxes = []
        for line in lines:
            line_sorted = sorted(line, key=lambda b: b.center_x)
            sorted_boxes.extend(line_sorted)
        
        return sorted_boxes
    
    def save_crops(self, word_boxes: List[WordBox], output_dir: str, image_name: str):
        """
        Save cropped word images to disk.
        
        Args:
            word_boxes: List of WordBox objects
            output_dir: Directory to save crops
            image_name: Base name for the image (without extension)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for box in word_boxes:
            filename = os.path.join(output_dir, f"{image_name}_word_{box.index:04d}.jpg")
            cv2.imwrite(filename, box.image)
    
    def visualize_detections(self, image_path: str, word_boxes: List[WordBox], output_path: str):
        """
        Visualize detections with bounding boxes and indices.
        
        Args:
            image_path: Path to original image
            word_boxes: List of WordBox objects
            output_path: Path to save visualization
        """
        img = cv2.imread(image_path)
        
        for box in word_boxes:
            # Draw rectangle
            cv2.rectangle(img, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
            
            # Draw index
            label = f"{box.index}"
            cv2.putText(img, label, (box.x1, box.y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
