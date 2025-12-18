# -*- coding: utf-8 -*-
"""
TP1 - Manipulation d'images sous Python
Affichage des images originales, puis des composantes RVB, HSV et YCrCb (sans OpenCV)
+ Domaine fréquentiel (DCT, DWT sans PyWavelets)
+ Parcours de base d'images
"""

# =============================================================================
# IMPORTATION DES MODULES
# =============================================================================
print("=" * 60)
print("TP1 - MANIPULATION D'IMAGES SOUS PYTHON")
print("=" * 60)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

print("✓ Modules importés avec succès")

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def verifier_fichier(chemin):
    return os.path.exists(chemin)

def afficher_proprietes(image, nom):
    if isinstance(image, np.ndarray):
        print(f"  {nom}: Shape {image.shape}, Type {image.dtype}")
    elif hasattr(image, 'size'):
        print(f"  {nom}: Size {image.size}, Mode {getattr(image, 'mode', 'N/A')}")

def separer_composantes_rgb(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        return image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    return None, None, None

def enregistrer_composantes(r, g, b, dossier="Composantes_RGB"):
    os.makedirs(dossier, exist_ok=True)
    Image.fromarray(r).save(os.path.join(dossier, "Canal_Rouge.png"))
    Image.fromarray(g).save(os.path.join(dossier, "Canal_Vert.png"))
    Image.fromarray(b).save(os.path.join(dossier, "Canal_Bleu.png"))
    print(f"💾 Composantes enregistrées dans '{dossier}'")

# =============================================================================
# FONCTIONS POUR LA BASE D'IMAGES
# =============================================================================

def choisir_base_images():
    """Permet à l'utilisateur de choisir une base d'images"""
    print("\n" + "="*50)
    print("SELECTION DE LA BASE D'IMAGES")
    print("="*50)

    # ⭐⭐ REMPLACEZ CE CHEMIN PAR VOTRE VRAI CHEMIN ⭐⭐
    votre_chemin_principal = "/home/belahouel/Documents/Master\ 2/IRDM/TP/base\ de\ fleurs/"

    # Vérifier d'abord votre chemin principal
    if os.path.isdir(votre_chemin_principal):
        print(f"✅ VOTRE BASE PRINCIPALE TROUVÉE: {votre_chemin_principal}")
        utiliser_principal = input("Voulez-vous utiliser cette base? (O/n): ").strip().lower()
        if utiliser_principal in ['', 'o', 'oui', 'y', 'yes']:
            return votre_chemin_principal

    # Si non, continuer avec la sélection normale
    dossiers_images = []
    for item in os.listdir('.'):
        if os.path.isdir(item):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            images_trouvees = []
            for ext in extensions:
                images_trouvees.extend(glob.glob(os.path.join(item, ext)))
            if images_trouvees:
                dossiers_images.append(item)

    print("\nDossiers disponibles contenant des images :")
    for i, dossier in enumerate(dossiers_images, 1):
        nb_images = len([f for f in os.listdir(dossier) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        print(f"  {i}. {dossier} ({nb_images} images)")

    print(f"  {len(dossiers_images) + 1}. Saisir un chemin personnalisé")
    print(f"  {len(dossiers_images) + 2}. Utiliser le dossier courant")

    try:
        choix = int(input(f"\nChoisissez une option (1-{len(dossiers_images) + 2}): "))

        if 1 <= choix <= len(dossiers_images):
            return dossiers_images[choix - 1]
        elif choix == len(dossiers_images) + 1:
            chemin_perso = input("Entrez le chemin du dossier: ")
            if os.path.isdir(chemin_perso):
                return chemin_perso
            else:
                print("❌ Dossier introuvable. Utilisation du dossier courant.")
                return "."
        else:
            return "."
    except:
        print("❌ Choix invalide. Utilisation du dossier courant.")
        return "."

def parcourir_base_images(dossier_base):
    """Parcourt et affiche toutes les images d'une base par groupes de 10"""
    print(f"\n📁 Parcours de la base: {dossier_base}")

    # Recherche des images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images_paths = []
    for ext in extensions:
        images_paths.extend(glob.glob(os.path.join(dossier_base, ext)))
        images_paths.extend(glob.glob(os.path.join(dossier_base, ext.upper())))

    if not images_paths:
        print("❌ Aucune image trouvée dans la base sélectionnée.")
        return

    print(f"📸 {len(images_paths)} image(s) trouvée(s)")

    # Affichage des images par groupes de 10
    for i in range(0, len(images_paths), 10):
        groupe_images = images_paths[i:i+10]

        # Créer une figure avec 2 lignes et 5 colonnes pour 10 images
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Base d'images: {dossier_base} - Images {i+1} à {i+len(groupe_images)}",
                    fontsize=16, fontweight='bold')

        # Aplatir le tableau d'axes pour faciliter le parcours
        axes = axes.ravel()

        for j, (ax, chemin_image) in enumerate(zip(axes, groupe_images)):
            try:
                img = Image.open(chemin_image)
                # Conversion en RGB pour l'affichage
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                ax.imshow(np.array(img))
                nom_fichier = os.path.basename(chemin_image)
                ax.set_title(f"{i+j+1}. {nom_fichier}", fontsize=10)
                ax.axis('off')

                # Affichage des propriétés dans la console
                print(f"  {i+j+1}. {nom_fichier} - {img.size} - {img.mode}")

            except Exception as e:
                ax.text(0.5, 0.5, f"Erreur\n{os.path.basename(chemin_image)}",
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f"{i+j+1}. ERREUR", fontsize=10)
                ax.axis('off')
                print(f"  {i+j+1}. {os.path.basename(chemin_image)} - ❌ Erreur: {e}")

        # Cacher les axes vides s'il y a moins de 10 images dans le dernier groupe
        for j in range(len(groupe_images), 10):
            axes[j].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

        # Pause entre les groupes d'images
        if i + 10 < len(images_paths):
            continuer = input("\nAppuyez sur Entrée pour voir le groupe suivant (ou 'q' pour quitter): ")
            if continuer.lower() == 'q':
                print("Arrêt demandé par l'utilisateur.")
                break

    print(f"\n✅ Parcours terminé. {len(images_paths)} images affichées.")

# =============================================================================
# CONVERSIONS MANUELLES
# =============================================================================

def rgb_to_hsv_manual(image_rgb):
    image = image_rgb.astype('float') / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    cmax, cmin = np.max(image, axis=2), np.min(image, axis=2)
    diff = cmax - cmin

    h = np.zeros_like(cmax)
    mask = diff != 0
    h[mask & (cmax == r)] = (60 * ((g - b) / diff) % 360)[mask & (cmax == r)]
    h[mask & (cmax == g)] = (60 * ((b - r) / diff + 2))[mask & (cmax == g)]
    h[mask & (cmax == b)] = (60 * ((r - g) / diff + 4))[mask & (cmax == b)]

    s = np.zeros_like(cmax)
    s[cmax != 0] = (diff / cmax)[cmax != 0]
    v = cmax

    hsv = np.stack([h / 360, s, v], axis=-1)
    return (hsv * 255).astype(np.uint8)

def rgb_to_ycrcb_manual(image_rgb):
    image = image_rgb.astype('float')
    R, G, B = image[..., 0], image[..., 1], image[..., 2]
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128
    ycrcb = np.stack([Y, Cr, Cb], axis=-1)
    return np.clip(ycrcb, 0, 255).astype(np.uint8)

# =============================================================================
# FONCTIONS DE TRANSFORMATION FRÉQUENTIELLE OPTIMISÉES
# =============================================================================

def dct2_manual(image):
    """Calcul DCT 2D optimisé avec vectorisation NumPy"""
    M, N = image.shape
    image = image.astype(float)

    # Pré-calcul des coefficients
    x = np.arange(M)[:, np.newaxis]
    u = np.arange(M)
    cos_x = np.cos((2*x + 1) * u * np.pi / (2*M))

    y = np.arange(N)[:, np.newaxis]
    v = np.arange(N)
    cos_y = np.cos((2*y + 1) * v * np.pi / (2*N))

    # Facteurs de normalisation
    alpha_u = np.ones(M) * np.sqrt(2/M)
    alpha_u[0] = np.sqrt(1/M)
    alpha_v = np.ones(N) * np.sqrt(2/N)
    alpha_v[0] = np.sqrt(1/N)

    # DCT 2D vectorisée
    dct = alpha_u[:, np.newaxis] * (cos_x.T @ image @ cos_y) * alpha_v

    return dct

def dwt_manual(image):
    """DWT 2D optimisée (Haar) avec redimensionnement pour éviter les problèmes de taille"""
    image = image.astype(float)

    # S'assurer que les dimensions sont paires
    rows, cols = image.shape
    if rows % 2 != 0:
        image = image[:rows-1, :]
        rows -= 1
    if cols % 2 != 0:
        image = image[:, :cols-1]
        cols -= 1

    # Transformation horizontale
    low_horizontal = (image[0::2, :] + image[1::2, :]) / 2
    high_horizontal = (image[0::2, :] - image[1::2, :]) / 2

    # Transformation verticale sur les résultats horizontaux
    LL = (low_horizontal[:, 0::2] + low_horizontal[:, 1::2]) / 2
    LH = (low_horizontal[:, 0::2] - low_horizontal[:, 1::2]) / 2
    HL = (high_horizontal[:, 0::2] + high_horizontal[:, 1::2]) / 2
    HH = (high_horizontal[:, 0::2] - high_horizontal[:, 1::2]) / 2

    return LL, LH, HL, HH
# =============================================================================
# FONCTIONS DE COMPARAISON PAR HISTOGRAMME
# =============================================================================

def calculer_histogramme(image):
    """Calcule l'histogramme normalisé d'une image en niveaux de gris."""
    if len(image.shape) == 3:
        image = np.array(Image.fromarray(image).convert('L'))  # conversion en niveaux de gris
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float) / np.sum(hist)  # normalisation
    return hist

def comparer_histogrammes(hist1, hist2, methode="correlation"):
    """Compare deux histogrammes selon une méthode donnée."""
    if methode == "correlation":
        # corrélation de Pearson entre deux histogrammes
        num = np.sum((hist1 - hist1.mean()) * (hist2 - hist2.mean()))
        den = np.sqrt(np.sum((hist1 - hist1.mean())**2) * np.sum((hist2 - hist2.mean())**2))
        return num / den if den != 0 else 0
    elif methode == "chi2":
        return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))
    elif methode == "intersection":
        return np.sum(np.minimum(hist1, hist2))
    else:
        raise ValueError("Méthode de comparaison inconnue")

def comparaison_base_histogrammes(dossier_base, methode="correlation"):
    """Compare les histogrammes des images d'une base entre elles."""
    print("\n🎯 PHASE 3: COMPARAISON DES IMAGES PAR HISTOGRAMME")
    print("="*60)

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images_paths = []
    for ext in extensions:
        images_paths.extend(glob.glob(os.path.join(dossier_base, ext)))
        images_paths.extend(glob.glob(os.path.join(dossier_base, ext.upper())))

    if len(images_paths) < 2:
        print("❌ Pas assez d'images pour comparer.")
        return

    print(f"📸 {len(images_paths)} images trouvées dans la base.")

    # Sélection de la première image comme référence
    ref_path = images_paths[0]
    ref_img = np.array(Image.open(ref_path).convert('L'))
    ref_hist = calculer_histogramme(ref_img)

    scores = []
    noms = []

    for path in images_paths:
        img = np.array(Image.open(path).convert('L'))
        hist = calculer_histogramme(img)
        score = comparer_histogrammes(ref_hist, hist, methode)
        scores.append(score)
        noms.append(os.path.basename(path))

    # Tri des scores (descendant si corrélation/intersection)
    reverse = methode in ["correlation", "intersection"]
    indices_tries = np.argsort(scores)[::-1] if reverse else np.argsort(scores)

    # Affichage graphique
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(scores)), [scores[i] for i in indices_tries], color='gray')
    plt.xticks(range(len(scores)), [noms[i] for i in indices_tries], rotation=45, ha='right')
    plt.title(f"Comparaison des histogrammes ({methode}) — image de référence: {os.path.basename(ref_path)}")
    plt.ylabel("Similarité")
    plt.tight_layout()
    plt.show()

    # Afficher la référence et les 3 plus similaires
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("Image de référence et les 3 plus similaires", fontsize=14, fontweight='bold')

    axes[0].imshow(ref_img, cmap='gray')
    axes[0].set_title("Référence")
    axes[0].axis('off')

    for i, idx in enumerate(indices_tries[1:4], start=1):
        img = np.array(Image.open(images_paths[idx]).convert('L'))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{noms[idx]}\nScore: {scores[idx]:.3f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # PARCOURS DE LA BASE D'IMAGES
    # -------------------------------------------------------------------------
    print("\n🎯 PHASE 1: PARCOURS DE LA BASE D'IMAGES")
    dossier_base = choisir_base_images()
    parcourir_base_images(dossier_base)

    # -------------------------------------------------------------------------
    # TRAITEMENT DES IMAGES SPÉCIFIQUES (partie originale)
    # -------------------------------------------------------------------------
    print("\n🎯 PHASE 2: ANALYSE DES IMAGES SPÉCIFIQUES")

    image_couleur_path = "Image couleurs.bmp"
    image_gris_path = "Image niveaux de gris.bmp"

    if not verifier_fichier(image_couleur_path) or not verifier_fichier(image_gris_path):
        print(f"❌ Les fichiers '{image_couleur_path}' ou '{image_gris_path}' sont introuvables.")
        print("   Poursuite du programme avec la base d'images uniquement.")
        return

    # OUVERTURE DES IMAGES ORIGINALES
    image_couleur = np.array(Image.open(image_couleur_path))
    image_gris = np.array(Image.open(image_gris_path).convert("L"))

    afficher_proprietes(image_couleur, "Image Couleur Originale")
    afficher_proprietes(image_gris, "Image Niveaux de Gris Originale")

    # 🔹 Fenêtre d'affichage des images originales
    fig0, axes0 = plt.subplots(1, 2, figsize=(10, 5))
    fig0.suptitle("Images Originales Importées", fontsize=16, fontweight='bold')
    axes0[0].imshow(image_couleur)
    axes0[0].set_title("Image Couleur Originale")
    axes0[0].axis("off")
    axes0[1].imshow(image_gris, cmap="gray")
    axes0[1].set_title("Image Niveaux de Gris")
    axes0[1].axis("off")
    plt.show(block=False)

    # CONVERSIONS ET COMPOSANTES
    r, g, b = separer_composantes_rgb(image_couleur)
    enregistrer_composantes(r, g, b)

    hsv = rgb_to_hsv_manual(image_couleur)
    ycrcb = rgb_to_ycrcb_manual(image_couleur)

    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    # AFFICHAGE DES TROIS FENÊTRES (RVB, HSV, YCrCb)
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle("Composantes RVB", fontsize=16, fontweight='bold')
    for ax, data, title in zip(axes1, [r, g, b], ['Rouge (R)', 'Vert (G)', 'Bleu (B)']):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("Composantes HSV", fontsize=16, fontweight='bold')
    for ax, data, title in zip(axes2, [h, s, v], ['Teinte (H)', 'Saturation (S)', 'Valeur (V)']):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle("Composantes YCrCb", fontsize=16, fontweight='bold')
    for ax, data, title in zip(axes3, [Y, Cr, Cb], ['Y (Luminance)', 'Cr (Rouge)', 'Cb (Bleu)']):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()

    # 🔹 DOMAINE FRÉQUENTIEL : DCT + DWT
    print("\n⏳ Calcul de la DCT et DWT manuelles...")

    # === DCT ===
    dct_result = dct2_manual(image_gris)
    dct_display = np.log1p(np.abs(dct_result))   # compression dynamique
    dct_display = dct_display / dct_display.max() * 255
    dct_display = dct_display.astype(np.uint8)

    fig_dct, ax_dct = plt.subplots(figsize=(6, 6))
    ax_dct.imshow(dct_display, cmap='gray', vmin=0, vmax=255)
    ax_dct.set_title("Domaine fréquentiel (DCT manuelle)")
    ax_dct.axis('off')

    # === DWT ===
    LL, LH, HL, HH = dwt_manual(image_gris)

    fig_dwt, axes_dwt = plt.subplots(2, 2, figsize=(8, 8))
    fig_dwt.suptitle("Domaine fréquentiel (DWT manuelle - Haar)", fontsize=14, fontweight='bold')
    titles = ['Approximation (LL)', 'Détails horizontaux (LH)', 'Détails verticaux (HL)', 'Détails diagonaux (HH)']
    for ax, data, title in zip(axes_dwt.ravel(), [LL, LH, HL, HH], titles):
        ax.imshow(data, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.show()
    print("✅ Affichage DCT et DWT terminé avec succès.")
        # -------------------------------------------------------------------------
    # COMPARAISON DES IMAGES PAR HISTOGRAMME
    # -------------------------------------------------------------------------
    comparaison_base_histogrammes(dossier_base, methode="correlation")


# =============================================================================
# LANCEMENT
# =============================================================================
if __name__ == "__main__":
    main()
