import os
import pickle

INPUT_PKL = "Test_annotation.pkl"
OUTPUT_PKL = "Test_annotation_modified.pkl"

def transform_paths(data_dict):
    """
    For each key in data_dict, modify the image_path of each entry by:
      - Replacing the prefix '/home3/lk/surveillance_images/Anomaly_Test/' 
        with '../ucf_crime/frames/'
      - Renaming 'image_XXXX.jpg' to a 6-digit zero-padded filename.
    """
    new_data = {}

    for key, bboxes in data_dict.items():
        updated_list = []
        for item in bboxes:
            old_path, x1, y1, x2, y2 = item
            
            # 1) Remove old prefix
            rel_path = old_path.replace("/home3/lk/surveillance_images/Anomaly_Test/", "")
            
            # 2) Split directory from filename
            dir_part, filename = os.path.split(rel_path)
            
            # 3) Extract the integer from "image_XXXX.jpg"
            #    e.g. "image_0861" -> "0861" -> int(861) -> "861" -> zero-pad to "000861"
            stem, ext = os.path.splitext(filename)  # e.g. stem="image_0861", ext=".jpg"
            number_str = stem.replace("image_", "")  # e.g. "0861"
            number_int = int(number_str)             # e.g. 861
            new_filename = f"{number_int:06d}{ext}"  # e.g. "000861.jpg"
            
            # 4) Construct new path
            new_path = os.path.join("../ucf_crime/frames", dir_part, new_filename)
            
            # 5) Create the updated annotation entry
            updated_list.append([new_path, x1, y1, x2, y2])
        
        new_data[key] = updated_list
    
    return new_data

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load original data
    with open(INPUT_PKL, "rb") as f:
        data = pickle.load(f)
    
    # Transform paths
    new_data = transform_paths(data)
    
    # Save the updated structure
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(new_data, f)

    print(f"Modified data saved to '{OUTPUT_PKL}'.")
