# 1) Installa le librerie necessarie (diffusers, transformers, accelerate, safetensors).
!pip install diffusers transformers accelerate safetensors

# 2) Importa le librerie.
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display

# 3) Carica il modello "runwayml/stable-diffusion-v1-5" (o un altro se preferisci).
#    NOTA: richiede una GPU per funzionare in tempi ragionevoli (menu Runtime -> Change runtime type -> GPU).
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# 4) Sposta il modello sulla GPU (selezionata in Colab).
pipe.to("cuda")

# 5) Definisci il prompt testuale.
prompt = "roman empire"

# 6) Genera l'immagine.
#    - num_inference_steps = numero di passi (più alto = immagine più dettagliata, ma più lenta)
#    - guidance_scale = quanto l'immagine deve seguire fedelmente il prompt
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# 7) Visualizza e/o salva l’immagine generata
display(image)
image.save("image.png")
