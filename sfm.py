import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm


class StructureFromMotion:
    """
    A class for performing Structure from Motion (SfM) reconstruction from multiple images.
    
    This class provides methods to:
    - Load and preprocess images
    - Extract and match features between images
    - Estimate camera poses
    - Triangulate 3D points
    - Export results to PLY or OBJ format
    """
    
    def __init__(self, image_directory: str, k_path: str, downscale_factor: int = 2):
        """
        Initialize the Structure from Motion pipeline.
        
        Args:
            image_directory (str): Path to directory containing images
            k_path (str): Path to file containing camera intrinsic parameters
            downscale_factor (int): Factor by which to downscale images (default: 2)
        """
        self.image_directory = image_directory
        self.k_path = k_path
        self.downscale_factor = downscale_factor
        
        # Initialize storage for results
        self.total_points = np.zeros((1, 3))
        self.total_colors = np.zeros((1, 3))
        
        # Load images and camera parameters
        self.image_paths, self.K = self._load_images_and_K()
        self.image_list, self.K = self._downscale_images_and_K()
        
    def _load_images_and_K(self):
        """
        Loads images from a directory and camera intrinsic parameters from a file.
        
        Returns:
            image_paths (list[str]): List of paths to the images.
            K (ndarray (3, 3)): The K matrix.
        """
        image_paths = [os.path.join(self.image_directory, i) for i in os.listdir(self.image_directory)]
        print(f"Loaded {len(image_paths)} images: {image_paths}")
        
        with open(self.k_path, "r") as f:
            lines = f.readlines()
            K = np.float32([i.strip().split(" ") for i in lines])
            
        return image_paths, K
    
    def _downscale_images_and_K(self):
        """
        Downscale the images using Gaussian pyramid and adjust camera intrinsic parameters.
        
        Returns:
            downscaled_image_list (list[ndarray]): List of downscaled images.
            K_adjusted (ndarray (3, 3)): Downscaled K matrix.
        """
        downscaled_image_list = []
        
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            for _ in range(1, int(self.downscale_factor / 2) + 1):
                img = cv2.pyrDown(img)
            downscaled_image_list.append(img)
        
        # Adjust camera matrix for downscaling
        K_adjusted = self.K.copy()
        K_adjusted[0, 0] /= self.downscale_factor  # fx
        K_adjusted[1, 1] /= self.downscale_factor  # fy
        K_adjusted[0, 2] /= self.downscale_factor  # cx
        K_adjusted[1, 2] /= self.downscale_factor  # cy
        
        return downscaled_image_list, K_adjusted
    
    @staticmethod
    def find_features(image_0, image_1):
        """
        Feature detection using the SIFT algorithm and KNN matching.
        
        Args:
            image_0 (ndarray): First image
            image_1 (ndarray): Second image
            
        Returns:
            features_0 (ndarray (N, 2)): Keypoints(features) of image 0.
            features_1 (ndarray (N, 2)): Keypoints(features) of image 1.
        """
        # Initialize SIFT
        sift = cv2.SIFT_create()
        
        # Detect and compute features
        key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
        key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)
        
        # Match features using BruteForce matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_0, desc_1, k=2)
        
        # Apply Lowe's ratio test
        features_0 = []
        features_1 = []
        for best_match, second_best_match in matches:
            if best_match.distance < 0.70 * second_best_match.distance:
                features_0.append(key_points_0[best_match.queryIdx].pt)
                features_1.append(key_points_1[best_match.trainIdx].pt)
        
        return np.float32(features_0), np.float32(features_1)
    
    @staticmethod
    def triangulation(projection_matrix_A, projection_matrix_B, features_A, features_B):
        """
        Compute 3D point cloud from two sets of 2D image points using triangulation.
        
        Args:
            projection_matrix_A (ndarray (3, 4)): Projection matrix for first camera
            projection_matrix_B (ndarray (3, 4)): Projection matrix for second camera
            features_A (ndarray (N, 2)): 2D points in first image
            features_B (ndarray (N, 2)): 2D points in second image
            
        Returns:
            pt_cloud (ndarray (N, 4)): 3D point cloud.
        """
        pt_cloud = cv2.triangulatePoints(projection_matrix_A, projection_matrix_B, 
                                       features_A.T, features_B.T)
        pt_cloud = pt_cloud / pt_cloud[3]
        return pt_cloud
    
    @staticmethod
    def solve_pnp(points_3d, features, K, dist_coeff, initial):
        """
        Solve the Perspective-n-Point (PnP) problem.
        
        Args:
            points_3d (ndarray (N, 1, 3)): Array of 3D points
            features (ndarray (2, N) or (N, 2)): Array of corresponding 2D features
            K (ndarray (3, 3)): Camera intrinsic matrix
            dist_coeff (ndarray (5, 1)): Distortion coefficients
            initial (int): Initial flag
            
        Returns:
            rotation_matrix (ndarray (3, 3)): Rotation matrix.
            translation_vector (ndarray (3, 1)): Translation vector.
            features (ndarray (N, 2)): Inlier features.
            points_3d (ndarray (N, 3)): Inlier 3D points.
        """
        if initial == 1:
            points_3d = points_3d[:, 0, :]
            features = features.T
        
        try:
            # Solve PnP with RANSAC
            _, rotation_vector_calc, translation_vector, inlier = cv2.solvePnPRansac(
                points_3d, features, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector_calc)
            
            if inlier is not None:
                features = features[inlier[:, 0]]
                points_3d = points_3d[inlier[:, 0]]
                
        except cv2.error:
            print("Warning: PnP failed, returning default pose")
            rotation_matrix = np.eye(3)
            translation_vector = np.zeros((3, 1))
        
        return rotation_matrix, translation_vector, features, points_3d
    
    @staticmethod
    def calculate_reprojection_error(points_3d, features, transform_matrix, K, homogenity):
        """
        Calculate reprojection error between 3D points and their corresponding 2D features.
        
        Args:
            points_3d (ndarray (4, N) or (N, 3)): Array of 3D points
            features (ndarray (2, N) or (N, 2)): Array of corresponding 2D features
            transform_matrix (ndarray (3, 4)): Transformation matrix [R|t]
            K (ndarray (3, 3)): Camera intrinsic matrix
            homogenity (int): Indicator for homogeneous coordinates
            
        Returns:
            error (float): The mean reprojection error.
            points_3d (ndarray (N, 1, 3)): The (possibly transformed) 3D points.
        """
        if points_3d.size == 0:
            return float('inf'), points_3d
        
        # Extract rotation and translation
        rot_matrix = transform_matrix[:3, :3]
        tran_vector = transform_matrix[:3, 3]
        rot_vector, _ = cv2.Rodrigues(rot_matrix)
        
        # Convert to homogeneous coordinates if needed
        if homogenity == 1:
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
        
        # Reshape for cv2.projectPoints
        points_3d = np.asarray(points_3d).reshape(-1, 1, 3)
        
        # Project 3D points to 2D
        features_calc, _ = cv2.projectPoints(points_3d, rot_vector, tran_vector, K, None)
        
        if features_calc is None:
            return float('inf'), points_3d
        
        # Calculate error
        features_calc = np.float32(features_calc[:, 0, :])
        features = np.float32(features.T if homogenity == 1 else features)
        
        error = cv2.norm(features_calc, features, cv2.NORM_L2)
        return error / len(features_calc), points_3d
    
    @staticmethod
    def find_correspondences(img_points_1, img_points_2, img_points_3):
        """
        Find matching points between three images and return their indices and unmatched points.
        
        Args:
            img_points_1 (ndarray (N,2)): 2D points in first image
            img_points_2 (ndarray (N,2)): 2D points in second image
            img_points_3 (ndarray (N,2)): 2D points in third image
            
        Returns:
            matched_idx_1 (ndarray (N)): Array of indices of the points in the first image.
            matched_idx_2 (ndarray (N)): Array of indices of the points in the second image.
            unmatched_points_image2 (ndarray (N,2)): Array of 2D points in the second image that are not in the first image.
            unmatched_points_image3 (ndarray (N,2)): Array of 2D points in the third image that are not in the first image.
        """
        matched_idx_1 = []
        matched_idx_2 = []
        
        # Find matching points between first and second image
        for i in range(img_points_1.shape[0]):
            a = np.where(img_points_2 == img_points_1[i, :])
            if a[0].size != 0:
                matched_idx_1.append(i)
                matched_idx_2.append(a[0][0])
        
        # Find unmatched points
        unmatched_points_image2 = np.ma.array(img_points_2, mask=False)
        unmatched_points_image2.mask[matched_idx_2] = True
        unmatched_points_image2 = unmatched_points_image2.compressed()
        unmatched_points_image2 = unmatched_points_image2.reshape(
            int(unmatched_points_image2.shape[0] / 2), 2)
        
        unmatched_points_image3 = np.ma.array(img_points_3, mask=False)
        unmatched_points_image3.mask[matched_idx_2] = True
        unmatched_points_image3 = unmatched_points_image3.compressed()
        unmatched_points_image3 = unmatched_points_image3.reshape(
            int(unmatched_points_image3.shape[0] / 2), 2)
        
        return (np.array(matched_idx_1, dtype=np.int32),
                np.array(matched_idx_2, dtype=np.int32),
                unmatched_points_image2,
                unmatched_points_image3)
    
    def _initialize_first_two_cameras(self):
        """
        Initialize the first two camera poses using the first two images.
        
        Returns:
            tuple: Initial camera matrices, features, and 3D points
        """
        # Initialize transform matrices
        # Transform matrix represents rotation and translation [R|t]
        transform_matrix_0 = np.array([[1, 0, 0, 0], 
                                     [0, 1, 0, 0], 
                                     [0, 0, 1, 0]])
        transform_matrix_1 = np.empty((3, 4))
        
        # Create projection matrices
        # Projection matrix represents K.[R|t]
        projection_matrix_0 = np.matmul(self.K, transform_matrix_0)
        projection_matrix_1 = np.empty((3, 4))
        
        # Get first two images
        image_0 = self.image_list[0]
        image_1 = self.image_list[1]
        
        # Find features between first two images using SIFT and KNN
        features_0, features_1 = self.find_features(image_0, image_1)
        
        # Find essential matrix and filter features
        essential_matrix, inlier_mask = cv2.findEssentialMat(
            features_0, features_1, self.K, method=cv2.RANSAC, 
            prob=0.999, threshold=0.4, mask=None)
        features_0 = features_0[inlier_mask.ravel() == 1]
        features_1 = features_1[inlier_mask.ravel() == 1]
        
        # Recover pose and filter features
        _, rotation_matrix, translation_matrix, inlier_mask = cv2.recoverPose(
            essential_matrix, features_0, features_1, self.K)
        features_0 = features_0[inlier_mask.ravel() > 0]
        features_1 = features_1[inlier_mask.ravel() > 0]
        
        # Update transform matrix
        # R1 = R_relative * R0
        transform_matrix_1[:3, :3] = np.matmul(rotation_matrix, transform_matrix_0[:3, :3])
        # t1 = t_0 + (R0 * t_relative)
        transform_matrix_1[:3, 3] = (transform_matrix_0[:3, 3] + 
                                   np.matmul(transform_matrix_0[:3, :3], translation_matrix.ravel()))
        
        # Update projection matrix
        projection_matrix_1 = np.matmul(self.K, transform_matrix_1)
        
        # Triangulate points: find 3D points from corresponding 2D points
        points_3d = self.triangulation(projection_matrix_0, projection_matrix_1, features_0, features_1)
        
        # Calculate and refine using PnP
        error, points_3d = self.calculate_reprojection_error(
            points_3d, features_1.T, transform_matrix_1, self.K, homogenity=1)
        _, _, features_1, points_3d = self.solve_pnp(
            points_3d, features_1.T, self.K, np.zeros((5, 1), dtype=np.float32), initial=1)
        
        return (transform_matrix_0, transform_matrix_1, projection_matrix_0, projection_matrix_1,
                image_0, image_1, features_0, features_1, points_3d)
    
    def reconstruct(self, show_progress=True):
        """
        Perform the complete Structure from Motion reconstruction.
        
        Args:
            show_progress (bool): Whether to show progress and intermediate results
        """
        # Initialize first two cameras
        (transform_matrix_0, transform_matrix_1, projection_matrix_0, projection_matrix_1,
         image_0, image_1, features_0, features_1, points_3d) = self._initialize_first_two_cameras()
        
        # Initialize previous and current camera matrices
        prev_transform_matrix = transform_matrix_0
        current_transform_matrix = transform_matrix_1
        prev_projection_matrix = projection_matrix_0
        current_projection_matrix = projection_matrix_1
        prev_image = image_0
        current_image = image_1
        prev_features = features_0
        current_features = features_1
        
        # Process remaining images
        remaining_images_count = len(self.image_list) - 2
        for image_idx in tqdm(range(remaining_images_count), desc="Processing images"):
            next_image = self.image_list[image_idx + 2]
            current_to_next_features, next_features = self.find_features(current_image, next_image)
            
            if image_idx != 0:
                points_3d = self.triangulation(prev_projection_matrix, current_projection_matrix, prev_features, current_features)
                points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
                points_3d = points_3d[:, 0, :]
            
            # Find correspondences
            matched_prev_idx, matched_current_idx, unmatched_current_points, unmatched_next_points = \
                self.find_correspondences(current_features, current_to_next_features, next_features)
            corresponding_next_points = next_features[matched_current_idx]
            
            # Solve PnP for next camera
            rot_matrix, tran_matrix, corresponding_next_points, points_3d = self.solve_pnp(
                points_3d[matched_prev_idx], corresponding_next_points, self.K, 
                np.zeros((5, 1), dtype=np.float32), initial=0)
            
            current_transform_matrix = np.hstack((rot_matrix, tran_matrix))
            next_projection_matrix = np.matmul(self.K, current_transform_matrix)
            
            # Calculate reprojection error
            error, points_3d = self.calculate_reprojection_error(
                points_3d, corresponding_next_points, current_transform_matrix, self.K, homogenity=0)
            
            # Triangulate new points
            points_3d = self.triangulation(current_projection_matrix, next_projection_matrix, 
                                         unmatched_current_points, unmatched_next_points)
            error, points_3d = self.calculate_reprojection_error(
                points_3d, unmatched_next_points.T, current_transform_matrix, self.K, homogenity=1)
            
            if show_progress:
                print(f"Image {image_idx+3}/{len(self.image_list)} - Reprojection Error: {error:.4f}")
            
            # Store results
            self.total_points = np.vstack((self.total_points, points_3d[:, 0, :]))
            next_image_points = np.array(unmatched_next_points.T, dtype=np.int32)
            color_vector = np.array([next_image[point[1], point[0]] for point in next_image_points.T])
            self.total_colors = np.vstack((self.total_colors, color_vector))
            
            # Update for next iteration
            prev_transform_matrix = np.copy(current_transform_matrix)
            prev_projection_matrix = np.copy(current_projection_matrix)
            prev_image = np.copy(current_image)
            current_image = np.copy(next_image)
            prev_features = np.copy(current_to_next_features)
            current_features = np.copy(next_features)
            current_projection_matrix = np.copy(next_projection_matrix)
            
            # Show current image if requested
            if show_progress:
                cv2.imshow('Current Image', next_image)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
        
        if show_progress:
            cv2.destroyAllWindows()
    
    def export_to_ply(self, filename='result.ply'):
        """
        Export reconstructed 3D points to PLY format.
        
        Args:
            filename (str): Output filename for PLY file
        """
        out_points = self.total_points.reshape(-1, 3) * 200
        out_colors = self.total_colors.reshape(-1, 3)
        print(f"Exporting {len(out_points)} points to {filename}")
        
        verts = np.hstack([out_points, out_colors])
        
        # Filter outliers
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''
        
        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    
    def export_to_obj(self, filename='result.obj'):
        """
        Export reconstructed 3D points to OBJ format.
        
        Args:
            filename (str): Output filename for OBJ file
        """
        out_points = self.total_points.reshape(-1, 3) * 200
        out_colors = self.total_colors.reshape(-1, 3)
        print(f"Exporting {len(out_points)} points to {filename}")
        
        verts = np.hstack([out_points, out_colors])
        
        # Filter outliers
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]
        
        with open(filename, 'w') as f:
            # Write vertices with colors
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(v[3])} {int(v[4])} {int(v[5])}\n")
    
    def run(self, result_format, output_filename=None):
        """
        Run the complete Structure from Motion pipeline.
        
        Args:
            result_format (str): Output format ('ply' or 'obj')
            output_filename (str): Custom output filename (optional)
        """
        print("Starting Structure from Motion reconstruction...")
        print(f"Processing {len(self.image_list)} images")
        
        # Perform reconstruction
        self.reconstruct()
        
        # Export results
        if output_filename is None:
            output_filename = f"result.{result_format}"
        
        if result_format.lower() == "ply":
            self.export_to_ply(output_filename)
        elif result_format.lower() == "obj":
            self.export_to_obj(output_filename)
        else:
            raise ValueError("result_format must be 'ply' or 'obj'")
        
        print(f"Reconstruction complete! Results saved to {output_filename}")


# Example usage
if __name__ == "__main__":
    # Create SfM instance
    sfm = StructureFromMotion("example/monument", "example/K.txt")
    
    # Run reconstruction and export to PLY
    sfm.run("ply")
    
    # Alternative usage:
    # sfm.run("obj", "my_reconstruction.obj")