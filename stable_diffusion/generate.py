# Make sure you're logged in with `huggingface-cli login`
import requests
from io import BytesIO
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Initialize model for simple diffusion
model_id_or_path = "runwayml/stable_diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
	model_id_or_path,
  safety_checker = None
)

pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# Generate the outfit on a mannequin
prompt = "mannequin with a crop top and short pleated skirt"
url = "https://www.nahanco.com/media/catalog/product/m/i/michelle1w.jpg?optimize=medium&bg-color=255,255,255&fit=bounds&height=390&width=560&canvas=560:390"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
improperly_colored_outfit = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

# improperly_colored_outfit[0]
improperly_colored_outfit[0].save("generated_outfit.png")