import os
import cv2
import json
import numpy as np
from scipy.spatial.distance import cdist


def load_json(filepath):
    """Try loading a JSON file and return its data, or None if failed."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"❌ Error loading {os.path.basename(filepath)}: {e}")
        return None

def save_json(data, filepath):
    """Save data as a JSON file, ensuring directories exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_polygon_mask_for_viz_multiclass(json_data, image_shape=(3000, 3000), mode='mask', classes_to_include=None):
    """
    Convert polygon annotations to a mask where each class has a different label (value).

    Parameters:
        json_data (dict): Annotation data loaded from JSON.
        image_shape (tuple): Shape of the output mask.
        mode (str): 'mask' for filled mask, 'outline' for outlines only.
        classes_to_include (list or None): List of class keys to include, or None to include all.

    Returns:
        np.ndarray: Mask image with class labels.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)

    for class_key, objects in json_data.items():
        if (classes_to_include is not None) and (class_key not in classes_to_include):
            continue  # Skip this class

        class_value = int(class_key) + 1  # Class 0 → mask value 1, Class 1 → mask value 2, etc.

        for obj in objects:
            if obj['type'] != 'POLYGON':
                continue  # skip non-polygon annotations

            points = obj['points']
            pts = np.array(points, dtype=np.int32).reshape(-1, 2)

            if mode == 'mask':
                cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], color=class_value)

            elif mode == 'outline':
                cv2.polylines(mask, [pts.reshape((-1, 1, 2))], isClosed=True, color=class_value, thickness=2)

            else:
                raise ValueError("Mode must be 'mask' or 'outline'.")

    return mask

def convert_json_to_polygon_format(conversion_file, class_name, polygon_json):

    new_version = {"points": conversion_file['contours'], "bounding_boxes": conversion_file["bounding_boxes"]}

    ids = []
    width = []
    height = []

    for index, contour in enumerate(new_version["points"]):
        length = 5 - len(str(index))
        ids.append("FFF-" + ("0"*length) + str(index))

    for i in conversion_file['bounding_boxes']:
        start, finish = i
        x1, y1 = start
        x2, y2 = finish
        width.append(x2 - x1)
        height.append(y2 - y1)

    extras = len(ids) - len(conversion_file['bounding_boxes'])

    new_version["id"] = ids
    new_version["width"] = width + ([0] * extras)
    new_version["height"] = height + ([0] * extras)
    new_version["confidence"] = [1.0] * len(ids)
    new_version["connected_components_centroids"] = conversion_file["connected_components_centroids"] + ([0] * extras)


    # Step 1: Compute actual centroids from contours (if needed)
    def compute_centroid(contour):
        contour = np.array(contour)  # Convert to NumPy array
        return tuple(np.mean(contour, axis=0))  # Compute mean x, y


    contour_centroids = [compute_centroid(c) for c in conversion_file["contours"]]

    # Step 2: Compute distances between all centroids and contour centroids
    dist_matrix = cdist(conversion_file["connected_components_centroids"], contour_centroids)

    # Step 3: Match centroids to contours (find the closest contour)
    matches = np.argmin(dist_matrix, axis=1)  # Get index of nearest contour

    del new_version["bounding_boxes"]

    objects = []

    for i in new_version["id"]:
        id_position = new_version["id"].index(i)
        points = np.array(new_version["points"][id_position]).flatten().tolist()

        object_dict = {
            "id": i,
            "type": "POLYGON",
            "x": int(np.round(contour_centroids[id_position][0])),
            "y": int(np.round(contour_centroids[id_position][1])),
            "width": new_version["width"][id_position],
            "height": new_version["height"][id_position],
            "radiusX": 0,
            "radiusY": 0,
            "points": points,
            "floatPoints": [float(point) for point in points],
            "active": True,
            "confidence": new_version["confidence"][id_position],
            "associatedModelPrefix": None,
            "classID": None
        }

        objects.append(object_dict)

    polygon_json[class_name] = objects

    # for i, contour in enumerate(conversion_file["contours"][:len(new_version["id"])][:2]):
    #     points = np.array(new_version["points"][i]).flatten().tolist()
    #     x = points[0::2]
    #     y = points[1::2]
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.plot(x + [x[0]], y + [y[0]], marker='o', linestyle='-')
    #
    #     center = contour_centroids[i]
    #
    #     print(center)
    #
    #     # Given circle parameters
    #     circle_x, circle_y = center
    #     circle_radius = 10 / 2  # Diameter 10 means radius is 5
    #
    #     plt.scatter(circle_x, circle_y, s=(circle_radius * 2) ** 2 * 3.14, color='red')
    #
    #     plt.xlabel("X-axis")
    #     plt.ylabel("Y-axis")
    #     plt.title("Polygon from Given Points")
    #     plt.gca().invert_yaxis()  # Invert Y-axis for correct orientation
    #     plt.grid(True)
    #     plt.show()

    return polygon_json

def convert_cellpose_mask_to_json(instance_mask, polygon_class='0'):
    """
    Convert a Cellpose instance mask to the custom JSON polygon format.
    Computes width, height, and sets confidence automatically.

    Parameters:
        instance_mask (np.ndarray): Instance mask where each object has a unique ID.
        polygon_class (str): The class key to assign to all polygons.

    Returns:
        dict: JSON-ready dictionary containing the annotations.
    """
    annotations_by_class = {}

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0]  # Exclude background

    for i, instance_id in enumerate(instance_ids):
        # Create binary mask for this instance
        instance_binary = (instance_mask == instance_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(instance_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue  # skip empty objects

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Get polygon points
        points = contour.squeeze().tolist()
        if len(points) < 3:
            continue  # skip degenerate polygons
        if not isinstance(points[0], list):
            points = [points]  # ensure it's a list of points

        # Calculate centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Compute bounding box (width, height)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y

        # Dummy confidence — set to 1.0
        confidence = 1.0

        object_dict = {
            "id": int(instance_id),
            "type": "POLYGON",
            "x": cx,
            "y": cy,
            "width": width,
            "height": height,
            "radiusX": 0,
            "radiusY": 0,
            "points": points,
            "floatPoints": [float(p) for pt in points for p in pt],
            "active": True,
            "confidence": confidence,
            "associatedModelPrefix": None,
            "classID": None
        }

        # Add to annotations under the specified class
        annotations_by_class.setdefault(polygon_class, []).append(object_dict)

    return annotations_by_class

def load_polygon_mask_for_cellpose(json_path, image_shape=(3000, 3000), classes_to_include=None, verbose=False):
    """
    Convert polygon annotations to an instance mask where each object has a unique label (value),
    compatible with Cellpose training.

    Parameters:
        json_path (str): Path to JSON annotation file.
        image_shape (tuple): Shape of the output mask.
        classes_to_include (list or None): List of class keys to include, or None to include all.
        verbose (bool): If True, print processing information.

    Returns:
        np.ndarray: Instance mask image.
    """
    with open(json_path, 'r') as f:
        annotations_by_class = json.load(f)

    instance_mask = np.zeros(image_shape, dtype=np.uint16)
    instance_id = 1

    if verbose:
        print(f"Processing JSON: {json_path}")

    for class_key, objects in annotations_by_class.items():
        if (classes_to_include is not None) and (class_key not in classes_to_include):
            if verbose:
                print(f"  Skipping class {class_key} (not in classes_to_include)")
            continue

        for obj in objects:
            if obj['type'] != 'POLYGON':
                if verbose:
                    print(f"    Skipping object ID {obj.get('id', 'unknown')} (type {obj['type']})")
                continue

            points = obj['points']
            pts = np.array(points, dtype=np.int32).reshape(-1, 2)

            if pts.shape[0] < 3:
                if verbose:
                    print(f"    Skipping object ID {obj.get('id', 'unknown')} (invalid polygon with {pts.shape[0]} points)")
                continue

            cv2.fillPoly(instance_mask, [pts.reshape((-1, 1, 2))], color=instance_id)

            instance_id += 1

    if verbose:
        print(f"Finished processing {json_path}. Total instances: {instance_id - 1}\n")

    return instance_mask

def should_skip(existing_files, name):
    """Check if a file should be skipped based on existing conversions."""
    return name in existing_files

def convert_araceli_json_to_rewire_format(params, group, index_start):
    """Convert Araceli mask info JSONs to Rewire polygon JSON format."""
    group_dir = os.path.join(params["directory"], group)
    existing_files = set(os.listdir(os.path.join(group_dir, params["polygon_dir"]))) | set(params["broken_files"])

    count = index_start

    for filename in os.listdir(os.path.join(group_dir, params["nuc_dir"])):
        name, _ = filename.split(".")
        base_name = name[:-3]
        new_filename = base_name + ".json"
        cell_filename = base_name + "_w3.json"

        if should_skip(existing_files, new_filename):
            print(f"🔄 Skipping {new_filename}, already exists.")
            continue

        print(f"\n▶ Processing nucleus: {filename}")

        nuc_path = os.path.join(group_dir, params["nuc_dir"], filename)
        nuc_data = load_json(nuc_path)
        if nuc_data is None:
            continue

        polygon_json = {}
        nuclei_polygon_json = convert_json_to_polygon_format(nuc_data, 0, polygon_json)

        cell_path = os.path.join(group_dir, params["cells_dir"], cell_filename)
        cell_data = load_json(cell_path)

        if cell_data:
            print(f"✅ Cell mask loaded: {cell_filename}")
            final_polygon_json = convert_json_to_polygon_format(cell_data, 1, nuclei_polygon_json)
        else:
            final_polygon_json = nuclei_polygon_json

        output_path = os.path.join(group_dir, params["polygon_dir"], new_filename)
        save_json(final_polygon_json, output_path)

        print(f"💾 Saved new file ({count}): {new_filename}")
        count += 1


if __name__ == "__main__":

    params_for_araceli_conversion = {
        "directory": r"W:\training sets\INTERNAL\Broad",
        "groups": [
            "BR00135824\masks",
            "BR00136506\masks",
            "TC00660048\masks"
        ],
        "cells_dir": "cells\mask_info",
        "nuc_dir": "nuclei\mask_info",
        "polygon_dir": "annotations",
        "broken_files": ["F4_s4.json"]
    }

    index = 1
    for group in params_for_araceli_conversion["groups"]:
        index = convert_araceli_json_to_rewire_format(params_for_araceli_conversion, group, index)
