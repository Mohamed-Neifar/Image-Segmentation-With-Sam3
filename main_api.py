import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gradio_client import Client, handle_file

# Imports locaux SAM 3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# --- 1. FONCTIONS DE VISUALISATION (Votre code) ---
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.rand(3, 1), [[0.6]]], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))


# --- 2. INITIALISATION DU MODÈLE LOCAL ---
print("Chargement du modèle local SAM 3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Note : build_sam3_image_model chargera les poids par défaut ou spécifiés
model = build_sam3_image_model().to(device)
processor = Sam3Processor(model)


def process_hybrid(image_path, prompt_text):
    if not os.path.exists(image_path):
        print(f"❌ Erreur : {image_path} introuvable.")
        return

    # --- ÉTAPE A : APPEL API (Résultats écrits) ---
    print(f"\n--- 🛰️ CONSULTATION API ---")
    try:
        client = Client("akhaliq/sam3")
        api_result = client.predict(
            image=handle_file(image_path),
            text=prompt_text,
            threshold=0.5,
            mask_threshold=0.5,
            api_name="/segment"
        )
        print("Résultats de l'API :")
        print(api_result[1])  # Affiche les scores et infos texte du serveur
    except Exception as e:
        print(f"⚠️ Erreur API : {e}")

    # --- ÉTAPE B : CALCUL LOCAL (Image segmentée) ---
    print(f"\n--- 💻 CALCUL LOCAL ---")
    raw_image = Image.open(image_path).convert("RGB")

    # Inférence locale
    inference_state = processor.set_image(raw_image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    # --- ÉTAPE C : VISUALISATION ---
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)

    for mask, box, score in zip(masks, boxes, scores):
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        show_box(box.cpu().numpy(), plt.gca())

    plt.title(f"SAM 3 Local - Prompt: {prompt_text}")
    plt.axis('off')
    plt.show()


# --- BOUCLE PRINCIPALE ---
if __name__ == "__main__":
    while True:
        img_input = input("\nEntrez le nom de l'image (ou 'q' pour quitter) : ")
        if img_input.lower() == 'q': break

        prompt_input = input("Entrez l'objet à segmenter : ")

        process_hybrid(img_input, prompt_input)