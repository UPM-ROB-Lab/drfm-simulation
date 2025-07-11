import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import pyvista as pv
import pickle

# =============================================================================
# Function Definitions (保持不变)
# =============================================================================

def area_triangle(x1, x2, x3):
    """Calculate the area of a triangle."""
    a = np.linalg.norm(x2 - x1)
    b = np.linalg.norm(x3 - x2)
    c = np.linalg.norm(x1 - x3)
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

def dividetriangle(x1, x2, x3, a0):
    """Recursively subdivide a triangle."""
    area_tri = area_triangle(x1, x2, x3)
    if area_tri > a0:
        len_side12 = np.linalg.norm(x1 - x2)
        len_side23 = np.linalg.norm(x2 - x3)
        len_side31 = np.linalg.norm(x3 - x1)

        if len_side12 >= len_side23 and len_side12 >= len_side31:
            midpoint = (x1 + x2) / 2
            vertx1_modify = x1
            vertx2_modify = midpoint
            vertx3_modify = x3
            vertx1_new = midpoint
            vertx2_new = x2
            vertx3_new = x3
        elif len_side23 >= len_side12 and len_side23 >= len_side31:
            midpoint = (x2 + x3) / 2
            vertx1_modify = x1
            vertx2_modify = x2
            vertx3_modify = midpoint
            vertx1_new = x1
            vertx2_new = midpoint
            vertx3_new = x3
        else:
            midpoint = (x3 + x1) / 2
            vertx1_modify = x1
            vertx2_modify = x2
            vertx3_modify = midpoint
            vertx1_new = midpoint
            vertx2_new = x2
            vertx3_new = x3

        y1 = dividetriangle(vertx1_modify, vertx2_modify, vertx3_modify, a0)
        y2 = dividetriangle(vertx1_new, vertx2_new, vertx3_new, a0)
        return y1 + y2
    else:
        return [[x1, x2, x3]]

def read_stl_file(filename):
    """Read an STL file."""
    try:
        # Try reading as ASCII first
        your_mesh = mesh.Mesh.from_file(filename)
    except ValueError:
        # If ASCII fails, try reading as binary
        print(f"Warning: Could not read {filename} as ASCII, attempting binary mode.")
        your_mesh = mesh.Mesh.from_file(filename, mode=mesh.Mode.BINARY)
    vertices = your_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return vertices, faces

def compute_mesh_properties(vertices, faces):
    """Compute mesh properties."""
    shape_matrix = []
    for face in faces:
        vert1 = vertices[face[0], :]
        vert2 = vertices[face[1], :]
        vert3 = vertices[face[2], :]
        pos_tri = (vert1 + vert2 + vert3) / 3
        vec1 = vert2 - vert1
        vec2 = vert3 - vert2
        normal_tri = np.cross(vec1, vec2)
        normal_tri = normal_tri / np.linalg.norm(normal_tri)
        area_tri_loc = area_triangle(vert1, vert2, vert3)
        shape_matrix.append(np.concatenate((pos_tri, normal_tri, [area_tri_loc])))
    return np.array(shape_matrix)

def remesh(vertices, faces, minimum_area):
    """Remesh the STL file."""
    new_shape_matrix = []
    for3dshape_new = []
    numTriangles = faces.shape[0]

    print("Starting remeshing process...")
    for i in range(numTriangles):
        if i % 100 == 0:
            print(f"Processing triangle {i+1}/{numTriangles}")
        vert1 = vertices[faces[i, 0], :]
        vert2 = vertices[faces[i, 1], :]
        vert3 = vertices[faces[i, 2], :]
        split_tri = dividetriangle(vert1, vert2, vert3, minimum_area)
        for tri in split_tri:
            vert1, vert2, vert3 = tri
            for3dshape_new.append(np.concatenate((vert1, vert2, vert3)))
            pos_tri = (vert1 + vert2 + vert3) / 3
            vec1 = vert2 - vert1
            vec2 = vert3 - vert2
            normal_tri = np.cross(vec1, vec2)
            normal_tri = normal_tri / np.linalg.norm(normal_tri)
            area_tri_loc = area_triangle(vert1, vert2, vert3)
            new_shape_matrix.append(np.concatenate((pos_tri, normal_tri, [area_tri_loc])))

    print("Remeshing completed.")
    return np.array(new_shape_matrix), np.array(for3dshape_new)

# ... (其他函数定义可以根据需要保留或删除，这里省略以简化代码)

if __name__ == '__main__':
    # Parameters
    stl_filenames = ['models/small_wheel_left_meters_rotatedY_simple.STL', 'models/small_wheel_right_meters_rotatedY_simple.STL']  # Scaled wheel files
    minimum_area = 5e-3 # Adjusted for meters scale (original 5e-3 / 1e6)
    flip_normals = True
    latest_coeff_path = 'latest_coeff.mat'
    output_filenames = ['small_wheel_left_metersY_simple.pkl', 'small_wheel_right_metersY_simple.pkl'] # Corresponding output PKL names

    # Load latest_coeff
    if not os.path.exists(latest_coeff_path):
        print(f"Error: The file '{latest_coeff_path}' does not exist.")
        exit()  # 退出程序，因为缺少必要文件
    else:
        latest_coeff = loadmat(latest_coeff_path)

    # Process both wheels
    for stl_filename, pkl_filename in zip(stl_filenames, output_filenames):
        print(f"\nProcessing {stl_filename}...")
        
        # Read and remesh
        print("Reading STL file...")
        vertices, faces = read_stl_file(stl_filename)
        print("Computing original mesh properties...")
        shape_matrix = compute_mesh_properties(vertices, faces)
        print("Remeshing...")
        new_shape_matrix, for3dshape_new = remesh(vertices, faces, minimum_area)

        if flip_normals:
            new_shape_matrix[:, 3:6] = -new_shape_matrix[:, 3:6]

        new_shape_matrix = new_shape_matrix[~np.isnan(new_shape_matrix).any(axis=1)]

        print(f"Total Area original mesh: {np.sum(shape_matrix[:, 6]):.5f}")
        print(f"Total area refined mesh: {np.sum(new_shape_matrix[:, 6]):.5f}")
        print(f"Total elements in original mesh: {shape_matrix.shape[0]}")
        print(f"Total elements in refined mesh: {new_shape_matrix.shape[0]}")
        print(f"Area Cutoff: {minimum_area}")
        print(f"Max element area in original mesh: {np.max(shape_matrix[:, 6])}")
        print(f"Max element area in refined mesh: {np.max(new_shape_matrix[:, 6])}")

        # Save new_shape_matrix to .pkl
        try:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(new_shape_matrix, f)
            print(f"new_shape_matrix saved to {pkl_filename}")
        except Exception as e:
            print(f"Error saving new_shape_matrix: {e}")

        # Load and verify (Optional, for testing)
        try:
            with open(pkl_filename, 'rb') as f:
                loaded_shape_matrix = pickle.load(f)
            print(f"new_shape_matrix loaded from {pkl_filename}")
            if np.array_equal(new_shape_matrix, loaded_shape_matrix):
                print("Loaded data is identical to the original data.")
            else:
                print("Loaded data is different from the original data.")
        except Exception as e:
            print(f"Error loading new_shape_matrix: {e}")


    # ... (删除或注释掉其他部分，例如 visualize_remesh 和 run_simulation)
