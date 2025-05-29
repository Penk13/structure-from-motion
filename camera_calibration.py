import numpy as np
import cv2
import glob
import os

def calibrate_camera_from_video(video_path, chessboard_size=(10, 7), frame_skip=30):
    """
    Calibrate the camera using chessboard frames extracted from a video.
    
    Args:
        video_path: Path to the video file
        chessboard_size: Tuple of (width, height) - number of inner corners
        frame_skip: Process every nth frame
        
    Returns:
        Camera matrix or None if calibration fails
    """
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)

            print(f"Frame {frame_count} - Chessboard found (saved as frame {saved_frame_count})")
            saved_frame_count += 1
        else:
            print(f"Frame {frame_count} - Chessboard NOT found")

    cap.release()

    if not objpoints:
        print("No chessboard patterns detected in video.")
        return None

    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save only camera matrix
    np.savetxt('camera_matrix.txt', mtx)

    print(f"Calibration complete! RMS re-projection error: {ret}")
    print("Camera matrix saved to camera_matrix.txt")
    return mtx


def calibrate_camera_from_images(images_path, chessboard_size=(10, 7)):
    """
    Calibrate the camera using chessboard images.
    
    Args:
        images_path: Path to directory containing calibration images
        chessboard_size: Tuple of (width, height) - number of inner corners
        
    Returns:
        Camera matrix or None if calibration fails
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images from directory
    if not os.path.exists(images_path):
        print(f"Directory not found: {images_path}")
        return None
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files from the directory
    all_files = os.listdir(images_path)
    images = []
    for file in all_files:
        if os.path.splitext(file.lower())[1] in image_extensions:
            images.append(os.path.join(images_path, file))
    
    if not images:
        print(f"No calibration images found in directory: {images_path}")
        return None
    
    print(f"Found {len(images)} calibration images")
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard found")
        else:
            print(f"Processed image {idx+1}/{len(images)}: {fname} - Chessboard NOT found")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None
    
    print("Calibrating camera...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Save only camera matrix
    np.savetxt('camera_matrix.txt', mtx)
    
    print(f"Calibration complete! RMS re-projection error: {ret}")
    print("Camera matrix saved to camera_matrix.txt")
    
    return mtx


def main():
    """
    Main function to run the camera calibration process.
    """
    print("Starting camera calibration for SfM...")
    
    # Example usage - modify these paths as needed
    images_path = 'example/calibration_images'
    video_path = 'example/calibration.mp4'
    chessboard_size = (10, 7)
    frame_skip = 30
    source = 'images'
    
    # Choose calibration source:
    mtx = None
    if source == 'images':
        mtx = calibrate_camera_from_images(images_path, chessboard_size)
    elif source == 'video':
        mtx = calibrate_camera_from_video(video_path, chessboard_size, frame_skip)
    
    if mtx is None:
        print("Calibration failed. Exiting.")
        return
    
    print("Camera calibration completed successfully!")


if __name__ == "__main__":
    main()