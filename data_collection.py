# data_collection.py

import os
import pandas as pd
import cv2


def collect_data(
    data_file=r"C:\\Users\\Durgesh\\Desktop\\TIH FINAL (IMP)\\synthetic_crop_13crops_500each(npk with image).csv",
    image_dir=r"C:\\Users\\Durgesh\\Desktop\\TIH FINAL (IMP)\\IMAGE\\RESIZED_IMAGES",
):
    """
    Loads soil image data and associated features from a CSV file and image directory.

    Parameters:
        data_file (str): Path to the CSV file containing metadata.
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple containing lists of images, soil types, N values, P values,
        K values, pH values, and crop recommendations.
    """
    # Load the CSV file
    soil_data = pd.read_csv(data_file)

    # Print columns to verify
    print("Columns in CSV:", soil_data.columns.tolist())

    # Initialize lists to store data
    images_list = []
    soil_types = []
    N_values = []
    P_values = []
    K_values = []
    pH_values = []
    crop_lists = []

    # Loop through the CSV file and load images and data
    for index, row in soil_data.iterrows():
        try:
            image_filename = str(row["IMAGE"]).strip()
            image_path = os.path.join(image_dir, image_filename)
            if os.path.isfile(image_path):
                # Load image
                img = cv2.imread(image_path)
                if img is None:
                    print(
                        f"Warning: Unable to read image {image_path}. Skipping this entry."
                    )
                    continue
                images_list.append(img)

                # Append other data
                soil_types.append(str(row["Soilname"]).strip())
                N_values.append(float(row["Nitrogen (kg/ha)"]))
                P_values.append(float(row["Phosphorus (kg/ha)"]))
                K_values.append(float(row["Potassium (kg/ha)"]))
                pH_values.append(float(row["pH"]))
                # Split the crops into a list
                crops = [crop.strip() for crop in str(row["Crop"]).split(",")]
                crop_lists.append(crops)
            else:
                print(f"Warning: Image {image_path} not found. Skipping this entry.")
        except KeyError as e:
            print(f"KeyError: {e}. Skipping this row.")
        except Exception as e:
            print(f"Error processing row {index}: {e}. Skipping this row.")

    print(f"Data collection complete. Total samples collected: {len(images_list)}")
    return images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists


if __name__ == "__main__":
    # Collect the data
    images_list, soil_types, N_values, P_values, K_values, pH_values, crop_lists = (
        collect_data()
    )
    # Print first entry to verify
    if images_list:
        print("First image shape:", images_list[0].shape)
        print("First soil type:", soil_types[0])
        print("First NPK values:", N_values[0], P_values[0], K_values[0])
        print("First pH value:", pH_values[0])
        print("First crop list:", crop_lists[0])
    else:
        print("No images were loaded.")
