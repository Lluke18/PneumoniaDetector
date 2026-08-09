import os
import pandas as pd

df = pd.read_csv("stage2_train_metadata.csv")

image_files = os.listdir("./New_DS/")

patient_ids = [
    os.path.splitext(filename)[0]
    for filename in image_files
    if filename.endswith(".png")
]


selected_df = df[df["patientId"].isin(patient_ids)]

selected_df = selected_df.drop_duplicates(
    subset="patientId",
    keep="first"
)

print("Images:", len(patient_ids))
print("CSV rows:", len(selected_df))

selected_df.to_csv(
    "./train_cropped_no_duplicates.csv",
    index=False
)