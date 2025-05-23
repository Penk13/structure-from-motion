import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from tomlkit import boolean
from tqdm import tqdm


def load_images_and_K(dir_path, k_path):
    """
    Loads images from a directory and camera intrinsic parameters from a file.

    Args:
        dir_path (str): Path to the directory containing the images.
        k_path (str): Path to the file containing the camera intrinsic parameters.

    Returns:
        image_paths (list[str]): List of paths to the images.
        K (ndarray (3, 3)): The K matrix.
    """

    image_paths = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    print(image_paths)

    with open(k_path, "r") as f:
        lines = f.readlines()
        K = np.float32([i.strip().split(" ") for i in lines])

    return image_paths, K


def downscale_images_and_K(image_list, K, scale=2):
    """
    Downscale the images using Gaussian pyramid and downscale camera intrinsic parameters.

    Args:
        image_list (list[str]): List of paths to the images.
        K (ndarray): Camera intrinsic parameters (K matrix).
        scale (int, optional): Downscale factor. Defaults to 2.

    Returns:
        downscaled_image_list (list[ndarray]): List of downscaled images.
        K (ndarray (3, 3)): Downscaled K matrix.
    """

    downscaled_image_list = []
    for img in image_list:
        img = cv2.imread(img)
        for _ in range(1, int(scale / 2) + 1):
            img = cv2.pyrDown(img)
        downscaled_image_list.append(img)

    K[0, 0] /= scale #fx
    K[1, 1] /= scale #fy
    K[0, 2] /= scale #cx
    K[1, 2] /= scale #cy
    
    return downscaled_image_list, K


def triangulation(projection_matrix_A, projection_matrix_B, features_A, features_B):
    """
    Compute the 3D point cloud from two sets of 2D image points using triangulation.

    Args:
        projection_matrix_A (ndarray (3, 4)): Projection matrix for the first camera.
        projection_matrix_B (ndarray (3, 4)): Projection matrix for the second camera.
        features_A (ndarray (N, 2)): 2D image points in the first image.
        features_B (ndarray (N, 2)): 2D image points in the second image.

    Returns:
        pt_cloud (ndarray (N, 4)): 3D point cloud.
    """

    pt_cloud = cv2.triangulatePoints(projection_matrix_A, projection_matrix_B, features_A.T, features_B.T)
    pt_cloud = pt_cloud / pt_cloud[3]

    return pt_cloud


def pnp(obj_point, image_point, K, dist_coeff, initial):
    if initial == 1:
        obj_point = obj_point[:, 0 ,:]
        image_point = image_point.T

    try:
        # Try to solve PnP with RANSAC
        _, rot_vector_calc, tran_vector, inlier = cv2.solvePnPRansac(obj_point, image_point, K, dist_coeff, cv2.SOLVEPNP_ITERATIVE)
        rot_matrix, _ = cv2.Rodrigues(rot_vector_calc)
        
        if inlier is not None:
            image_point = image_point[inlier[:, 0]]
            obj_point = obj_point[inlier[:, 0]]
            
    except cv2.error:
        # If PnP fails, return identity rotation and zero translation
        print("Warning: PnP failed, returning default pose")
        rot_matrix = np.eye(3)
        tran_vector = np.zeros((3, 1))
        inlier = None

    return rot_matrix, tran_vector, image_point, obj_point


def reprojection_error(points_3d, features, transform_matrix, K, homogenity):
    """
    Calculate the reprojection error between 3D points and their corresponding 2D features.

    Args:
        points_3d (ndarray (4, N) or (N, 3)): Array of 3D points.
        features (ndarray (2, N) or (N, 2)): Array of corresponding 2D features.
        transform_matrix (ndarray (3,4)): Transformation matrix [R|t].
        K (ndarray (3,3)): Camera intrinsic matrix.
        homogenity (int): Indicator for homogeneous coordinates.

    Returns:
        error (float): The mean reprojection error.
        points_3d (ndarray (N, 1, 3)): The (possibly transformed) 3D points.
    """
    # Handle empty object points
    if points_3d.size == 0:
        return float('inf'), points_3d  # Return high error if no points

    # Extract rotation matrix and translation vector from transformation matrix
    rot_matrix = transform_matrix[:3, :3]
    tran_vector = transform_matrix[:3, 3]
    rot_vector, _ = cv2.Rodrigues(rot_matrix)
    
    # Convert to homogeneous coordinates if needed
    if homogenity == 1:
        points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
    
    # Ensure points_3d is reshaped to (N, 1, 3) for cv2.projectPoints
    points_3d = np.asarray(points_3d).reshape(-1, 1, 3)
    
    # Reproject 3D points to 2D image space using the camera matrix
    features_calc, _ = cv2.projectPoints(points_3d, rot_vector, tran_vector, K, None)
    
    # Check if projection failed
    if features_calc is None:
        return float('inf'), points_3d
    
    # Reshape calculated features for error computation
    features_calc = np.float32(features_calc[:, 0, :])
    features = np.float32(features.T if homogenity == 1 else features)
    
    # Calculate and return the reprojection error
    error = cv2.norm(features_calc, features, cv2.NORM_L2)
    return error / len(features_calc), points_3d


def to_ply(point_clouds, colors):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
    verts = np.hstack([out_points, out_colors])

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
    
    with open('result.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


def to_obj(point_clouds, colors):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
    verts = np.hstack([out_points, out_colors])

    mean = np.mean(verts[:, :3], axis=0)
    scaled_verts = verts[:, :3] - mean
    dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)

    verts = verts[indx]
    with open('result.obj', 'w') as f:
        # Write vertices with colors
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {int(v[3])} {int(v[4])} {int(v[5])}\n")
 

def correspondences(img_points_1, img_points_2, img_points_3):
    cr_points_1 = []
    cr_points_2 = []

    for i in range(img_points_1.shape[0]):
        a = np.where(img_points_2 == img_points_1[i, :])
        if a[0].size != 0:
            cr_points_1.append(i)
            cr_points_2.append(a[0][0])

    mask_array_1 = np.ma.array(img_points_2, mask=False)
    mask_array_1.mask[cr_points_2] = True
    mask_array_1 = mask_array_1.compressed()
    mask_array_1 = mask_array_1.reshape(int(mask_array_1.shape[0] / 2), 2)

    mask_array_2 = np.ma.array(img_points_3, mask=False)
    mask_array_2.mask[cr_points_2] = True
    mask_array_2 = mask_array_2.compressed()
    mask_array_2 = mask_array_2.reshape(int(mask_array_2.shape[0] / 2), 2)

    return (
        np.array(cr_points_1, dtype=np.int32),
        np.array(cr_points_2, dtype=np.int32),
        mask_array_1,
        mask_array_2
    )


def find_features(image_0, image_1):
    '''
    Feature detection using the SIFT algorithm and KNN.

    Args:
        image_0 (ndarray): First image.
        image_1 (ndarray): Second image.

    Returns:
        features_0 (ndarray (N, 2)): Keypoints(features) of image 0.
        features_1 (ndarray (N, 2)): Keypoints(features) of image 1.
    '''

    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Convert to grayscale (SIFT requires grayscale images)
    # detectAndCompute() detects keypoints and computes descriptors
    # key_points is a list of keypoints in the image
    # descr is a descriptor of the image (128 floats for each keypoint)
    key_points_0, desc_0 = sift.detectAndCompute(cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY), None)
    key_points_1, desc_1 = sift.detectAndCompute(cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY), None)

    # BruteForce Matcher to compare descriptors
    bf = cv2.BFMatcher()
    # Find its 2 nearest neighbors for each descriptor in desc_0 and desc_1 based on Euclidean distance
    # Filter matches based on Lowe's ratio
    matches = bf.knnMatch(desc_0, desc_1, k=2)
    features_0 = []
    features_1 = []
    for best_match, second_best_match in matches:
        if best_match.distance < 0.70 * second_best_match.distance:
            features_0.append(key_points_0[best_match.queryIdx].pt)
            features_1.append(key_points_1[best_match.trainIdx].pt)

    return np.float32(features_0), np.float32(features_1)


def run(image_directory: str, k_path: str, result_format: str):
    # Loads images from a directory and camera intrinsic parameters from a file.
    image_paths, K = load_images_and_K(image_directory, k_path)

    # Downscale the images using Gaussian pyramid and downscale camera intrinsic parameters.
    image_list, K = downscale_images_and_K(image_paths, K)

    # This is transform matrix, represent [R|t] 
    # R is 3x3 matrix, t is 3x1 vector
    transform_matrix_0 = np.array(
        [[1, 0, 0, 0], 
         [0, 1, 0, 0], 
         [0, 0, 1, 0]])
    transform_matrix_1 = np.empty((3, 4))

    # P = K.[R|t] ---> multiplication of K and [R|t] results in a projection matrix
    projection_matrix_0 = np.matmul(K, transform_matrix_0)
    projection_matrix_1 = np.empty((3, 4)) 

    # numpy array to hold 3D points and colors
    total_points = np.zeros((1, 3))
    total_colors = np.zeros((1, 3))

    # Read the first two images
    image_0 = image_list[0]
    image_1 = image_list[1]

    # Find features in the first two images using SIFT and KNN
    features_0, features_1 = find_features(image_0, image_1)

    # Find the essential matrix and filter the features (remove outliers)
    essential_matrix, inlier_mask = cv2.findEssentialMat(features_0, features_1, K, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    features_0 = features_0[inlier_mask.ravel() == 1]
    features_1 = features_1[inlier_mask.ravel() == 1]

    # Recover the pose and filter the features (remove outliers)
    _, rotation_matrix, translation_matrix, inlier_mask = cv2.recoverPose(essential_matrix, features_0, features_1, K)
    features_0 = features_0[inlier_mask.ravel() > 0]
    features_1 = features_1[inlier_mask.ravel() > 0]

    # Update the transform matrix [R|t] using the recovered pose
    # R1 = R_relative * R0
    transform_matrix_1[:3, :3] = np.matmul(rotation_matrix, transform_matrix_0[:3, :3])
    # t1 = t_0 + (R0 * t_relative)
    transform_matrix_1[:3, 3] = transform_matrix_0[:3, 3] + np.matmul(transform_matrix_0[:3, :3], translation_matrix.ravel())

    # P = K.[R|t] ---> multiplication of K and [R|t] results in a projection matrix
    projection_matrix_1 = np.matmul(K, transform_matrix_1)

    # Triangulate points: find 3D points from corresponding 2D points
    points_3d = triangulation(projection_matrix_0, projection_matrix_1, features_0, features_1)
    
    # Calculate reprojection error
    error, points_3d = reprojection_error(points_3d, features_1.T, transform_matrix_1, K, homogenity = 1)

    _, _, features_1, points_3d = pnp(points_3d, features_1.T, K, np.zeros((5, 1), dtype=np.float32), initial=1)
    total_images = len(image_list) - 2 
    threshold = 0.5

    for i in tqdm(range(total_images)):
        image_2 = image_list[i + 2]
        features_cur, features_2 = find_features(image_1, image_2)

        if i != 0:
            points_3d = triangulation(projection_matrix_0, projection_matrix_1, features_0, features_1)
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]

        cm_points_0, cm_points_1, cm_mask_0, cm_mask_1 = correspondences(features_1, features_cur, features_2)
        cm_points_2 = features_2[cm_points_1]

        rot_matrix, tran_matrix, cm_points_2, points_3d = pnp(points_3d[cm_points_0], cm_points_2, K, np.zeros((5, 1), dtype=np.float32), initial = 0)
        # print(rot_matrix.shape)
        # print(tran_matrix.shape)
        transform_matrix_1 = np.hstack((rot_matrix, tran_matrix))
        pose_2 = np.matmul(K, transform_matrix_1)

        error, points_3d = reprojection_error(points_3d, cm_points_2, transform_matrix_1, K, homogenity = 0)

        points_3d = triangulation(projection_matrix_1, pose_2, cm_mask_0, cm_mask_1)
        error, points_3d = reprojection_error(points_3d, cm_mask_1.T, transform_matrix_1, K, homogenity = 1)
        print("Reprojection Error: ", error)

        total_points = np.vstack((total_points, points_3d[:, 0, :]))
        points_left = np.array(cm_mask_1.T, dtype=np.int32)
        color_vector = np.array([image_2[l[1], l[0]] for l in points_left.T])
        total_colors = np.vstack((total_colors, color_vector)) 

        transform_matrix_0 = np.copy(transform_matrix_1)
        projection_matrix_0 = np.copy(projection_matrix_1)
        # plt.scatter(i, error)
        # plt.pause(0.05)

        image_0 = np.copy(image_1)
        image_1 = np.copy(image_2)
        features_0 = np.copy(features_cur)
        features_1 = np.copy(features_2)
        projection_matrix_1 = np.copy(pose_2)
        cv2.imshow(image_paths[0].split('\\')[-2], image_2)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()
    if result_format == "ply":
        to_ply(total_points, total_colors)
    elif result_format == "obj":
        to_obj(total_points, total_colors)

run("example/monument", "example/K.txt", "ply")
