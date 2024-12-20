
import requests 
from bs4 import BeautifulSoup
import pandas as pd
import subprocess
import tempfile
import os



base_url = 'https://www.spicemarketcolour.com.au/'
seasons_links_dict = {'spring':['bright-spring-celebrities','true-spring-celebrities','light-spring-celebrities'], 'summer':['light-summer-celebrities','true-summer-celebrities','soft-summer-celebrities'], 'autumn':['soft-autumn-celebrities','true-autumn-celebrities','dark-autumn-celebrities'], 'winter':['dark-winter-celebrities','true-winter-celebrities','bright-winter-celebrities']}


headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
      'Accept-Encoding': 'none',
      'Accept-Language': 'en-US,en;q=0.8',
      'Connection': 'keep-alive'}

request = requests.get(base_url, headers=headers)


seasonal_images = {}
seasons = ['spring','winter','summer','autumn']

#  extract all img tags with class img, download the images, and save onto dictionary
for s in seasons:
    i = 0
    s_links = []
    for url in seasons_links_dict[s]:
        try: 
            request = requests.get(base_url+url, headers=headers)
            soup = BeautifulSoup(request.content, 'html.parser')
            images = soup.find_all('img', attrs={'class':'thumb-image'})
            links = [listing['data-src'] for listing in images]
            for l in links: 
                local_path = filename.format(s, i)
                with open(local_path, "wb") as f:
                    f.write(l.content)
                i += 1
                s_links.append(local_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {seasonal_images[s][i]}: {e}")
        except Exception as e:
            print(f"An error occurred with image {seasonal_images[s][i]}: {e}")
    seasonal_images[s] = s_links

processed_images = {}
script_path = "./preprocess.sh"


#Run preprocess script on each image
for s in seasons: 
    input_folder = f"../../combined-demo/pics/{s}"
    output_folder = f"../../combined-demo/processed_images/{s}"
    for image_name in os.listdir(input_folder):
        print(image_name)
        input_image_path = os.path.join(input_folder, image_name)
        output_image_path = os.path.join(output_folder, image_name)
        try:
            subprocess.run(["bash", script_path, input_image_path,output_image_path], check=True)
            print(f"Processed {image_name} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to process {image_name}: {e}")    
    

        


