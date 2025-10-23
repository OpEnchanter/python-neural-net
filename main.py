import json, time
from tqdm import tqdm
from keras.datasets import mnist
data = mnist.load_data()

with open("imdata.csv", "w") as f:
    imdata = data[0][0].tolist()
    imdata_normalized = []
    with tqdm(total=len(imdata), desc="Normalizing pixel data", unit="img") as pbar:
        for img in imdata:
            single_imdata = []
            for px in img:
                for py in px:
                    single_imdata.append(py/255)
            imdata_normalized.append(single_imdata)
            pbar.update(1)

    with tqdm(total=len(imdata), desc="Writing pixel data", unit="img") as pbar:
        for img in imdata_normalized:
            for px in img:
                f.write(f"{str(px)},")
            f.write("\n")
            pbar.update(1)

with open("imlabels.csv", "w") as f:
    with tqdm(total=len(data[0][1]), desc="Writing labels", unit="label") as pbar:
        for label in data[0][1]:
            f.write(f"{label}\n")
            pbar.update(1)
    