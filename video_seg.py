import os
import cv2
import torch
import numpy as np
import sys

# Ajout du path pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sam3.model_builder import build_sam3_video_predictor

# --- CONFIGURATION ---
VIDEO_PATH = "penguins.mp4"
OUTPUT_VIDEO_PATH = "output_penguin_seg.mp4"
TEXT_PROMPT = "penguin"
DEVICE_ID = 0


def show_mask_on_image(image, mask, color=np.array([30 / 255, 144 / 255, 255 / 255])):
    """Superpose un masque coloré sur une image BGR"""
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = np.zeros_like(image)
    mask_image[:] = color * 255

    overlay = image.copy()
    mask_indices = mask > 0
    overlay[mask_indices] = cv2.addWeighted(image[mask_indices], 0.4, mask_image[mask_indices], 0.6, 0)
    return overlay


def main():
    print("Initialisation du prédicteur SAM 3...")

    if torch.cuda.is_available():
        gpus = [DEVICE_ID]
    else:
        gpus = []

    predictor = build_sam3_video_predictor(gpus_to_use=gpus)

    print(f"Chargement de la vidéo : {VIDEO_PATH}")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=VIDEO_PATH,
        )
    )
    session_id = response["session_id"]
    print(f"Session démarrée (ID: {session_id})")

    # 4. Ajout du prompt textuel (VERSION CORRIGÉE)
    print(f"Envoi du prompt : '{TEXT_PROMPT}'")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=TEXT_PROMPT,
        )
    )

    # --- CORRECTION DU BUG KEYERROR ---
    outputs = response.get("outputs", {})
    if not outputs:
        print("⚠️ Attention : Le modèle n'a rien renvoyé pour ce prompt.")
    elif "masks" in outputs:
        print(f"✅ Objet détecté sur la frame 0 (Masques présents).")
    else:
        # C'est ici que ça plantait avant. Maintenant on gère le cas.
        print(f"ℹ️ Prompt enregistré. Clés disponibles : {list(outputs.keys())}")
        print("Continuons vers la propagation pour voir si le masque apparaît...")

    # 5. Préparation de la sortie vidéo
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print("Propagation du masque et sauvegarde de la vidéo...")

    # 6. Propagation
    frame_idx = 0
    stream = predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    )

    for response_packet in stream:
        ret, frame = cap.read()
        if not ret: break

        outputs = response_packet.get("outputs")

        # Vérification robuste des masques
        if outputs and "masks" in outputs:
            masks_tensor = outputs["masks"]

            if isinstance(masks_tensor, torch.Tensor):
                masks_np = masks_tensor.cpu().numpy()
            else:
                masks_np = masks_tensor

            # On vérifie qu'il y a bien des données dans le masque
            if masks_np.shape[0] > 0:
                # [Objet, Frame_interne, H, W] -> souvent [0, 0, :, :]
                mask_2d = masks_np[0, 0]
                frame = show_mask_on_image(frame, mask_2d)

        out.write(frame)
        print(f"Traitement frame {frame_idx}", end='\r')
        frame_idx += 1

    cap.release()
    out.release()

    try:
        predictor.handle_request(dict(type="close_session", session_id=session_id))
    except:
        pass

    print(f"\n✅ Terminée ! Vidéo sauvegardée sous : {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()