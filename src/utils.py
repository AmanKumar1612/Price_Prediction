import os
import urllib.request
from tqdm import tqdm
from functools import partial
from pathlib import Path
import multiprocessing

def download_image(image_link, savefolder):
    if isinstance(image_link, str):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if not os.path.exists(image_save_path):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
            except Exception as ex:
                print(f'Warning: Not able to download - {image_link}\n{ex}')
    return

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # ✅ Use fewer processes (safe range: 4–8)
    num_processes = min(8, multiprocessing.cpu_count())

    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(num_processes) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
