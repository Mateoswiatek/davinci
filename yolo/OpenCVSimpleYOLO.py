#!/usr/bin/env python3
"""
OpenCV-based Simple YOLO Detector
Drop-in replacement for SimpleYOLO that uses only OpenCV DNN + raw weights
NO PyTorch, NO Ultralytics needed!

Compatible with the existing raspberrypi-new.py interface
"""

import cv2
import numpy as np
import logging
import os
import time
from typing import List, Dict

logger = logging.getLogger(__name__)

class SimpleYOLODetector:
    """
    OpenCV-based YOLO detector - drop-in replacement for ultralytics version
    Uses raw YOLO weights with OpenCV DNN module
    """
    
    def __init__(self, 
                 config_path: str = "yolo/yolov4-tiny.cfg",
                 weights_path: str = "yolo/yolov4-tiny.weights",
                 names_path: str = "yolo/coco.names",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 input_size: int = 320):  # Smaller for better Pi performance
        """
        Initialize OpenCV YOLO detector
        
        Args:
            config_path: Path to YOLO config file (.cfg)
            weights_path: Path to YOLO weights file (.weights)
            names_path: Path to class names file (.names)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            input_size: Input image size (320 recommended for Pi)
        """
        self.config_path = config_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
        # Model components
        self.net = None
        self.output_layers = []
        self.classes = []
        self.model_loaded = False
        
        # Performance tracking
        self.inference_times = []
        
        # Try to load model
        self._try_load_yolo()
    
    def _try_load_yolo(self):
        """Try to load YOLO model, fail gracefully if not available"""
        try:
            logger.info("Loading OpenCV YOLO model...")
            logger.info(f"  Config: {self.config_path}")
            logger.info(f"  Weights: {self.weights_path}")
            logger.info(f"  Classes: {self.names_path}")
            
            # Check if files exist
            if not os.path.exists(self.config_path):
                logger.error(f"Config file not found: {self.config_path}")
                logger.error("Download with: wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg")
                return False
                
            if not os.path.exists(self.weights_path):
                logger.error(f"Weights file not found: {self.weights_path}")
                logger.error("Download with: wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")
                return False
                
            if not os.path.exists(self.names_path):
                logger.error(f"Names file not found: {self.names_path}")
                logger.error("Download with: wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
                return False
            
            # Load network
            logger.info("Loading YOLO network...")
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Set backend and target for CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            try:
                # OpenCV 4.x format
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                # OpenCV 3.x format
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.model_loaded = True
            logger.info("✅ OpenCV YOLO model loaded successfully!")
            logger.info(f"  Classes: {len(self.classes)}")
            logger.info(f"  Output layers: {len(self.output_layers)}")
            logger.info(f"  Input size: {self.input_size}x{self.input_size}")
            
            return True
            
        except Exception as e:
            logger.warning(f"❌ Could not load OpenCV YOLO model: {e}")
            logger.info("YOLO detection will be disabled")
            self.model_loaded = False
            return False
    
    def is_available(self):
        """Check if YOLO model is available and loaded"""
        return self.model_loaded
    
    def detect_humans(self, frame):
        """
        Detect humans in frame and return detections
        Compatible with existing SimpleYOLO interface
        
        Args:
            frame: OpenCV image (BGR format)
            
        Returns:
            List of detection dictionaries with 'bbox' and 'confidence'
        """
        if not self.model_loaded or frame is None:
            return []
        
        start_time = time.time()
        
        try:
            height, width = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                frame,
                1/255.0,  # Scale factor
                (self.input_size, self.input_size),  # Size
                swapRB=True,  # Swap R and B channels (BGR -> RGB)
                crop=False
            )
            
            # Set input to network
            self.net.setInput(blob)
            
            # Run forward pass
            outputs = self.net.forward(self.output_layers)
            
            # Parse detections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]  # Class probabilities
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # Scale bounding box back to image size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Calculate top-left corner
                        x = int(center_x - w/2)
                        y = int(center_y - h/2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                self.confidence_threshold,
                self.nms_threshold
            )
            
            # Build final detections - filter for humans only (class_id = 0)
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    if class_ids[i] == 0:  # person class
                        x, y, w, h = boxes[i]
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, x)
                        y = max(0, y)
                        x2 = min(width, x + w)
                        y2 = min(height, y + h)
                        
                        # Compatible format with existing code
                        detections.append({
                            'bbox': (x, y, x2, y2),
                            'confidence': confidences[i]
                        })
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV YOLO detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes on frame
        Compatible with existing SimpleYOLO interface
        
        Args:
            frame: OpenCV image (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Frame with drawn detections
        """
        if not detections:
            return frame
        
        result_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw green rectangle for humans
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(
                result_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            
            # Label text
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result_frame
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_ms': np.mean(self.inference_times),
            'min_inference_ms': np.min(self.inference_times),
            'max_inference_ms': np.max(self.inference_times),
            'fps': 1000 / np.mean(self.inference_times) if self.inference_times else 0,
            'model_loaded': self.model_loaded,
            'total_classes': len(self.classes)
        }


def test_opencv_simple_yolo():
    """Test function for OpenCV Simple YOLO detector"""
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing OpenCV Simple YOLO detector...")
    
    # Initialize detector
    detector = SimpleYOLODetector()
    
    if not detector.is_available():
        logger.error("❌ YOLO model not available!")
        logger.info("Make sure you have downloaded to ~/davinci/yolo/:")
        logger.info("  - yolov4-tiny.cfg")
        logger.info("  - yolov4-tiny.weights") 
        logger.info("  - coco.names")
        return
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info("✅ Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = detector.detect_humans(frame)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Add performance info
            stats = detector.get_performance_stats()
            if stats:
                info_text = f"FPS: {stats['fps']:.1f} | Inference: {stats['avg_inference_ms']:.1f}ms | Humans: {len(detections)}"
                cv2.putText(
                    result_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display
            cv2.imshow('OpenCV Simple YOLO Test', result_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = detector.get_performance_stats()
        if stats:
            logger.info("=" * 50)
            logger.info("Final Performance Stats:")
            logger.info(f"  Average FPS: {stats['fps']:.1f}")
            logger.info(f"  Average inference: {stats['avg_inference_ms']:.1f}ms")
            logger.info("=" * 50)


if __name__ == "__main__":
    test_opencv_simple_yolo()
