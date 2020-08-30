from main import noisy_circle
import pandas as pd
import numpy as np
import random

img_dir = "/media/HDD_2TB.1/circles_train/"


# Generating Random Circles
def generate_train_data():
    num_images = 100000
    max_noise = 4

    list_of_circles = []
    for i in range(num_images):
        # Fetching dimensions of Noisy Circle
        row_col_rad, img = noisy_circle(200, 100, random.uniform(0.01, 1) * max_noise)
        row, col, rad = row_col_rad

        # Saving to numpy format
        img_loc = img_dir + str(i + 1) + ".npy"
        np.save(img_loc, img)

        circle_dict = {}
        circle_dict.update({'Img_path': img_loc, 'Row': row, 'Col': col, 'Radius': rad})

        # Adding Circle to the List of Circles
        list_of_circles.append(circle_dict)

    data_df = pd.DataFrame.from_dict(list_of_circles)
    return data_df


if __name__ == '__main__':
    circle_data_df = generate_train_data()
    # Saving Data Frame to CSV
    circle_data_df.to_csv(img_dir + 'train_data.csv', index=False)
