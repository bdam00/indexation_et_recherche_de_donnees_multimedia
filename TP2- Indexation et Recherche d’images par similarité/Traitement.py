# =============================================================================
# IMPORTATION DES MODULES
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance
import pywt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider

# =============================================================================
# FONCTIONS DE BASE
# =============================================================================

def load_image(path):
    """Charge une image couleur (BGR -> RGB)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l’image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_images_from_folder(folder):
    """Charge toutes les images d’un dossier et les retourne dans un dictionnaire."""
    images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            path = os.path.join(folder, filename)
            images[filename] = load_image(path)
    return images

# =============================================================================
# 1) DESCRIPTEURS DE COULEUR SIMPLES
# =============================================================================

def color_descriptors(image):
    """Retourne un vecteur [R_mean, G_mean, B_mean, R_var, G_var, B_var, R_m3, G_m3, B_m3]."""
    feats = []
    for i in range(3):
        channel = image[:, :, i].astype(np.float64)
        mean = np.mean(channel)
        var = np.var(channel)
        moment3 = np.mean((channel - mean) ** 3)
        feats.extend([mean, var, moment3])
    return np.array(feats)

def compare_by_color_vector(query_img, dataset_images, method="euclidean"):
    """
    Compare les images en utilisant les vecteurs de descripteurs simples.
    method : 'euclidean', 'manhattan', ou 'cosine'
    """
    query_vec = color_descriptors(query_img)
    results = []

    for name, img in dataset_images.items():
        vec = color_descriptors(img)

        if method == "euclidean":
            dist = np.linalg.norm(query_vec - vec)
        elif method == "manhattan":
            dist = np.sum(np.abs(query_vec - vec))
        elif method == "cosine":
            num = np.dot(query_vec, vec)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(vec)
            dist = 1 - (num / denom) if denom != 0 else 1.0
        else:
            raise ValueError("Méthode inconnue")

        results.append((name, dist))

    results.sort(key=lambda x: x[1])  # petite distance = plus proche
    return results


# =============================================================================
# 2) HISTOGRAMMES RGB / LAB / YCrCb
# =============================================================================

def show_histograms(image, title="Histogrammes"):
    """Affiche les histogrammes (courbes) dans différents espaces colorimétriques."""
    color_spaces = {
        "RGB": ("RGB", None),
        "CIE Lab": ("Lab", cv2.COLOR_RGB2LAB),
        "YCrCb": ("YCrCb", cv2.COLOR_RGB2YCrCb)
    }

    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=14)

    for i, (name, conv) in enumerate(color_spaces.items(), 1):
        if conv[1] is not None:
            converted = cv2.cvtColor(image, conv[1])
        else:
            converted = image.copy()

        plt.subplot(1, 3, i)
        for j, col in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([converted], [j], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            plt.plot(hist, color=col)
        plt.title(name)
        plt.xlabel("Intensité")
        plt.ylabel("Fréquence normalisée")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# =============================================================================
# COMPARAISON DES HISTOGRAMMES LAB / YCrCb POUR LES TOP 10 IMAGES
# =============================================================================
# =============================================================================
# FENÊTRE INTERACTIVE POUR LES HISTOGRAMMES LAB / YCrCb
# =============================================================================

def interactive_lab_ycrcb_histograms(query_img, dataset_images, results, top_n=10):
    """
    Affiche une interface interactive pour comparer les histogrammes LAB et YCrCb
    entre l'image requête et les top images similaires.
    """

    color_spaces = {
        "CIE Lab": cv2.COLOR_RGB2LAB,
        "YCrCb": cv2.COLOR_RGB2YCrCb
    }

    for space_name, conv in color_spaces.items():
        # Initialisation
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.subplots_adjust(bottom=0.25)
        fig.suptitle(f"Comparaison interactive ({space_name})", fontsize=14)

        # Axe pour le slider
        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Image n°', 1, top_n, valinit=1, valstep=1)

        # Fonction de calcul d'histogramme normalisé
        def calc_hist(img):
            converted = cv2.cvtColor(img, conv)
            hist_data = []
            for j, col in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([converted], [j], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_data.append(hist)
            return hist_data

        # Histogrammes de l’image requête
        query_hists = calc_hist(query_img)
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title("Image requête")
        axes[0, 0].axis('off')

        for j, col in enumerate(['r', 'g', 'b']):
            axes[0, 1].plot(query_hists[j], color=col)
        axes[0, 1].set_title(f"Histogramme requête ({space_name})")
        axes[0, 1].grid(alpha=0.3)

        # Initialisation de la première image similaire
        current_name = results[0][0]
        similar_img = dataset_images[current_name]
        similar_hists = calc_hist(similar_img)

        img_display = axes[1, 0].imshow(similar_img)
        axes[1, 0].set_title(f"Top 1 : {current_name}")
        axes[1, 0].axis('off')

        lines = []
        for j, col in enumerate(['r', 'g', 'b']):
            (line,) = axes[1, 1].plot(similar_hists[j], color=col)
            lines.append(line)
        axes[1, 1].set_title(f"Histogramme Top 1 ({space_name})")
        axes[1, 1].grid(alpha=0.3)

        # Fonction de mise à jour
        def update(val):
            idx = int(slider.val) - 1
            name = results[idx][0]
            img = dataset_images[name]
            hists = calc_hist(img)

            # Mettre à jour image et histogrammes
            img_display.set_data(img)
            axes[1, 0].set_title(f"Top {idx + 1} : {name}")
            for j in range(3):
                lines[j].set_ydata(hists[j])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()



# =============================================================================
# 3) HISTOGRAMMES 3D ET COMPARAISON
# =============================================================================

def compute_histogram(image):
    """Calcule un histogramme 3D RGB réduit et normalisé."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_by_histogram(query_img, dataset_images, metric=cv2.HISTCMP_INTERSECT):
    """Compare une image requête avec toutes les autres images du dataset (histogrammes)."""
    query_hist = compute_histogram(query_img)
    results = []
    for name, img in dataset_images.items():
        hist = compute_histogram(img)
        score = cv2.compareHist(query_hist, hist, metric)
        results.append((name, score))

    reverse = metric in [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT]
    results.sort(key=lambda x: x[1], reverse=reverse)
    return results

# =============================================================================
# 4) HISTOGRAMMES DANS LE DOMAINE FRÉQUENTIEL (DCT)
# =============================================================================

def compute_dct_histogram(image):
    """
    Calcule un histogramme combiné des coefficients DCT pour les trois canaux RGB.
    On conserve davantage de fréquences et on effectue une normalisation robuste.
    """
    hist_total = []

    for i in range(3):  # R, G, B
        channel = image[:, :, i].astype(np.float32)
        h, w = channel.shape

        # Redimensionner pour cohérence (puissance de 2)
        h2, w2 = (h // 8) * 8, (w // 8) * 8
        channel = cv2.resize(channel, (w2, h2))

        # Appliquer DCT 2D
        dct = cv2.dct(channel)
        dct_mag = np.log1p(np.abs(dct))  # log pour réduire les grandes amplitudes

        # Normalisation par canal
        dct_mag = cv2.normalize(dct_mag, None, 0, 1, cv2.NORM_MINMAX)

        # Histogramme sur une échelle plus fine
        hist = cv2.calcHist([dct_mag.astype(np.float32)], [0], None, [128], [0, 1])
        cv2.normalize(hist, hist)
        hist_total.append(hist.flatten())

    return np.concatenate(hist_total)


def compare_by_dct_histogram(query_img, dataset_images, metric=cv2.HISTCMP_CORREL):
    """Compare les histogrammes DCT entre l’image requête et la base."""
    query_hist = compute_dct_histogram(query_img)
    results = []
    for name, img in dataset_images.items():
        hist = compute_dct_histogram(img)
        score = cv2.compareHist(query_hist.astype('float32'), hist.astype('float32'), metric)
        results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# =============================================================================
# 5) HISTOGRAMMES DCT & DWT (AFFICHAGE COMPARATIF)
# =============================================================================

def show_top_dct_histograms(query_img, dataset_images, results, top_n=10, title="Top images DCT"):
    """Affiche l'image requête + top_n images trouvées avec DCT."""
    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=14)

    # Image requête
    plt.subplot(2, 6, 1)
    plt.imshow(query_img)
    plt.title("Image requête")
    plt.axis("off")

    # Top images
    for i, (name, score) in enumerate(results[:top_n], start=2):
        plt.subplot(2, 6, i)
        plt.imshow(dataset_images[name])
        plt.title(f"{name}\nScore={score:.3f}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =============================================================================
# 6) HISTOGRAMMES DWT (WAVELETS)
# =============================================================================

def compute_dwt_histogram(image, wavelet='haar'):
    """
    Calcule l'histogramme combiné des 4 sous-bandes DWT (cA, cH, cV, cD).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    cA, (cH, cV, cD) = pywt.dwt2(gray, wavelet)
    coeffs = [cA, cH, cV, cD]

    hist_total = []
    for c in coeffs:
        mag = np.abs(c)
        if np.max(mag) != 0:
            mag = mag / np.max(mag)
        hist = cv2.calcHist([mag.astype(np.float32)], [0], None, [256], [0, 1])
        cv2.normalize(hist, hist)
        hist_total.append(hist.flatten())
    return np.concatenate(hist_total)

def compare_by_dwt_histogram(query_img, dataset_images, metric=cv2.HISTCMP_CORREL, wavelet='haar'):
    """
    Compare les histogrammes DWT entre l’image requête et les images de la base.
    metric : métrique de comparaison OpenCV (ex : cv2.HISTCMP_CORREL)
    wavelet : type d’ondelette ('haar', 'db1', 'coif1', etc.)
    """
    query_hist = compute_dwt_histogram(query_img, wavelet=wavelet)
    results = []

    for name, img in dataset_images.items():
        hist = compute_dwt_histogram(img, wavelet=wavelet)
        score = cv2.compareHist(query_hist.astype('float32'),
                                hist.astype('float32'),
                                metric)
        results.append((name, score))

    # Pour les métriques où un score plus grand = plus similaire
    reverse = metric in [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT]
    results.sort(key=lambda x: x[1], reverse=reverse)
    return results
#
# def compute_dwt_histogram(image, wavelet='haar'):
#     """
#     Calcule l'histogramme combiné des sous-bandes DWT pour les 3 canaux RGB.
#     Chaque canal contribue à la signature finale.
#     """
#     hist_total = []
#
#     for i in range(3):  # R, G, B
#         channel = image[:, :, i].astype(np.float32)
#         cA, (cH, cV, cD) = pywt.dwt2(channel, wavelet)
#         coeffs = [cA, cH, cV, cD]
#
#         for c in coeffs:
#             mag = np.abs(c)
#             mag = mag / np.max(mag) if np.max(mag) != 0 else mag
#             hist = cv2.calcHist([mag.astype(np.float32)], [0], None, [128], [0, 1])
#             cv2.normalize(hist, hist)
#             hist_total.append(hist.flatten())
#
#     return np.concatenate(hist_total)


def show_dwt_subbands_histograms(image, wavelet='haar'):
    """
    Affiche les 4 sous-bandes (cA, cH, cV, cD) et leurs histogrammes pour l'image requête.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    cA, (cH, cV, cD) = pywt.dwt2(gray, wavelet)

    subbands = {'Approximation (cA)': cA, 'Horizontal (cH)': cH, 'Vertical (cV)': cV, 'Diagonale (cD)': cD}

    plt.figure(figsize=(12, 8))
    plt.suptitle("Sous-bandes DWT et leurs histogrammes", fontsize=14)

    for i, (name, coeff) in enumerate(subbands.items(), 1):
        mag = np.abs(coeff)
        if np.max(mag) != 0:
            mag = mag / np.max(mag)

        hist = cv2.calcHist([mag.astype(np.float32)], [0], None, [256], [0, 1])
        cv2.normalize(hist, hist)

        plt.subplot(4, 2, 2 * i - 1)
        plt.imshow(mag, cmap='gray')
        plt.title(name)
        plt.axis('off')

        plt.subplot(4, 2, 2 * i)
        plt.plot(hist, color='blue')
        plt.title(f"Histogramme {name}")
        plt.xlabel("Intensité")
        plt.ylabel("Fréquence normalisée")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def show_top_dwt_results(query_img, dataset_images, results, top_n=10, title="Top 10 images similaires (DWT)"):
    """Affiche l'image requête + top images DWT."""
    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=14)

    plt.subplot(2, 6, 1)
    plt.imshow(query_img)
    plt.title("Image requête")
    plt.axis("off")

    for i, (name, score) in enumerate(results[:top_n], start=2):
        plt.subplot(2, 6, i)
        plt.imshow(dataset_images[name])
        plt.title(f"{name}\nScore={score:.3f}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()



# =============================================================================
# 7) CORRÉLOGRAMMES DE COULEURS
# =============================================================================

# -----------------------------
# Corrélogramme couleur (corrigé)
# -----------------------------
def compute_color_correlogram(image, bins=4, distances=[1,3,5], resize_to=128):
    """
    Calcule un corrélogramme des couleurs (auto-correlogramme simplifié).
    - image : image RGB (numpy array)
    - bins : nombre de niveaux par canal (ex: 4 -> 64 couleurs)
    - distances : liste de distances (en pixels) à considérer (ex: [1,3,5])
    - resize_to : redimensionne l'image pour accélérer le calcul
    Retour : vecteur normalisé de longueur (bins**3) * len(distances)
    """
    # Petite sécurité : si image trop petite, on ne resize pas trop petit
    h0, w0 = image.shape[:2]
    target = resize_to
    if min(h0, w0) < resize_to:
        target = min(h0, w0)

    img = cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # Quantification correcte : valeurs 0..bins-1
    q = (img.astype(np.int32) * bins) // 256
    q = np.clip(q, 0, bins - 1)

    r = q[:, :, 0]
    g = q[:, :, 1]
    b = q[:, :, 2]
    color_indices = (r * bins * bins) + (g * bins) + b  # valeurs 0 .. bins^3 -1
    K = bins ** 3

    # Préparer vecteur de sortie : pour chaque distance on a K valeurs (auto-correlogramme)
    correlogram = np.zeros((len(distances), K), dtype=np.float64)
    # Comptages pour normalisation : nombre de voisins valides considérés pour chaque couleur & distance
    counts = np.zeros((len(distances), K), dtype=np.int64)

    # Directions (8) pour la distance d : on prend 8 points à distance exacte d (approx. manhattan/diagonals)
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    # Parcours des pixels (éviter bords en fonction du plus grand distance)
    max_d = max(distances)
    ys = range(max_d, h - max_d)
    xs = range(max_d, w - max_d)

    for (dy, dx) in dirs:
        # pour chaque direction on testera tous distances
        for di, d in enumerate(distances):
            # décalage
            shift_y = dy * d
            shift_x = dx * d

            # on compare la zone centrale et la zone décalée
            center = color_indices[max_d: h - max_d, max_d: w - max_d]  # shape (h-2max, w-2max)
            neighbor = color_indices[max_d + shift_y: h - max_d + shift_y,
                                     max_d + shift_x: w - max_d + shift_x]

            # Flatten
            center_f = center.ravel()
            neighbor_f = neighbor.ravel()

            # Pour performance : compter par couleur via bincount
            # On veut pour chaque couleur c : nombre de fois où neighbor == c while center == c
            # Donc on filter positions where center==c and neighbor==c.
            # Méthode : pour all possible colors, we can accumulate with boolean masks,
            # but faster is to loop unique colors present in center to reduce ops.

            # Compute mask of matches (same color)
            match_mask = (center_f == neighbor_f)
            matched_centers = center_f[match_mask]  # colors where neighbor equals center

            if matched_centers.size > 0:
                bc = np.bincount(matched_centers, minlength=K)
                correlogram[di] += bc  # accumulate matches for this direction

            # For normalization we need how many neighbor checks were done per color (i.e., occurrences of each center color)
            bc_center = np.bincount(center_f, minlength=K)
            counts[di] += bc_center  # each center pixel produced 1 neighbor-check for this direction

    # Maintenant normaliser : pour chaque distance d and color c => matches/ (counts)
    # évitez division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = np.zeros_like(correlogram)
        mask_nonzero = counts > 0
        normalized[mask_nonzero] = correlogram[mask_nonzero] / counts[mask_nonzero]

    # Flatten (distances concatenés) et normaliser globalement (optionnel)
    vec = normalized.flatten()
    # Normalisation L1 pour stabilité (somme 1) si somme>0
    s = np.sum(vec)
    if s > 0:
        vec = vec / s

    return vec


def compare_by_correlogram(query_img, dataset_images, bins=4, distances=[1,3,5]):
    """
    Calcule et compare les corrélogrammes (auto-correlogrammes) entre la requête et la base.
    Retourne une liste triée par similarité (corrélation de Pearson).
    """
    query_corr = compute_color_correlogram(query_img, bins=bins, distances=distances)
    results = []
    for name, img in dataset_images.items():
        corr = compute_color_correlogram(img, bins=bins, distances=distances)
        # Similarity : corr (Pearson) via numpy.corrcoef; fallback to small value if constant
        if np.all(query_corr == query_corr[0]) or np.all(corr == corr[0]):
            score = -1.0  # pas d'information discriminante
        else:
            # corrcoef returns 2x2 matrix, [0,1] is the correlation
            score = np.corrcoef(query_corr, corr)[0, 1]
            if np.isnan(score):
                score = -1.0
        results.append((name, float(score)))
    # plus grand score = plus similaire
    results.sort(key=lambda x: x[1], reverse=True)
    return results





# =============================================================================
# AFFICHAGE DES RÉSULTATS (GRILLE D'IMAGES)
# =============================================================================

def show_top_similar(query_img, dataset_images, results, top_n=10, title="Résultats de similarité"):
    """Affiche les top N images les plus similaires à la requête."""
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 6, 1)
    plt.imshow(query_img)
    plt.title("Image requête")
    plt.axis('off')

    for i, (name, score) in enumerate(results[:top_n], start=2):
        plt.subplot(2, 6, i)
        plt.imshow(dataset_images[name])
        plt.title(f"{name}\nScore={score:.3f}")
        plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    base_path = "/home/belahouel/Documents/Master 2/IRDM/TP/TP2/base2/"
    query_path = "/home/belahouel/Documents/Master 2/IRDM/TP/TP2/req2.jpg"

    print("=== Chargement de la base d'images... ===")
    images = load_images_from_folder(base_path)
    print(f"{len(images)} images chargées.")

    print("=== Chargement de l'image requête... ===")
    query_img = load_image(query_path)

    # -------------------------------------------------------------------------
    # ÉTAPE 1 : RECHERCHE PAR VECTEUR DE COULEUR
    # -------------------------------------------------------------------------
    print("\n=== Méthode 1 : Vecteurs de couleur (moyenne, variance, moment3) ===")
    results_vec = compare_by_color_vector(query_img, images, method="euclidean")

    print("\nTop 10 (méthode vectorielle) :")
    for rank, (name, dist) in enumerate(results_vec[:10], start=1):
        print(f"{rank:2d}. {name:25s} -> Distance = {dist:.4f}")

    show_top_similar(query_img, images, results_vec, top_n=10,
                     title="Méthode 1 - Similarité (Vecteurs de couleur)")

    # -------------------------------------------------------------------------
    # ÉTAPE 2 : AFFICHAGE DES HISTOGRAMMES DE L'IMAGE REQUÊTE
    # -------------------------------------------------------------------------
    print("\n=== Calcul et affichage des histogrammes de l'image requête ===")
    show_histograms(query_img, title="Histogrammes (RGB, CIE Lab, YCrCb)")

    # -------------------------------------------------------------------------
    # ÉTAPE 3 : RECHERCHE PAR HISTOGRAMMES
    # -------------------------------------------------------------------------
    print("\n=== Méthode 2 : Histogrammes RGB ===")
    results_hist = compare_by_histogram(query_img, images, metric=cv2.HISTCMP_INTERSECT)

    print("\nTop 10 (méthode histogramme) :")
    for rank, (name, score) in enumerate(results_hist[:10], start=1):
        print(f"{rank:2d}. {name:25s} -> Score = {score:.4f}")

    show_top_similar(query_img, images, results_hist, top_n=10,
                     title="Méthode 2 - Similarité (Histogrammes RGB)")
    # -------------------------------------------------------------------------
    # ÉTAPE 2.1 : COMPARAISON DES HISTOGRAMMES LAB / YCrCb
    # -------------------------------------------------------------------------
    print("\n=== Comparaison des histogrammes LAB et YCrCb pour les 10 meilleurs résultats ===")
    interactive_lab_ycrcb_histograms(query_img, images, results_hist, top_n=10)


    # -------------------------------------------------------------------------
    # ÉTAPE 4 : COMPARAISON DES DEUX MÉTHODES
    # -------------------------------------------------------------------------
    print("\n=== Comparaison des deux méthodes ===")
    top_vec = [name for name, _ in results_vec[:10]]
    top_hist = [name for name, _ in results_hist[:10]]

    common = set(top_vec).intersection(set(top_hist))
    precision_similarity = len(common) / 10 * 100
    print(f"Images communes dans les deux top 10 : {len(common)}/10")
    print(f"Précision de recouvrement : {precision_similarity:.1f}%")

    # -------------------------------------------------------------------------
    # ÉTAPE 5 : AFFICHAGE CÔTE À CÔTE DES HISTOGRAMMES + IMAGES
    # -------------------------------------------------------------------------
    print("\n=== Comparaison visuelle des histogrammes et images ===")
    best_vec_name = results_vec[0][0]
    best_hist_name = results_hist[0][0]
    best_vec_img = images[best_vec_name]
    best_hist_img = images[best_hist_name]

    imgs_to_show = [
        (query_img, "Image Requête"),
        (best_vec_img, f"Top 1 - Méthode Vecteur ({best_vec_name})"),
        (best_hist_img, f"Top 1 - Méthode Histogramme ({best_hist_name})")
    ]

    # --- Affichage des histogrammes ---
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(imgs_to_show, 1):
        plt.subplot(2, 3, i)
        for j, col in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img], [j], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            plt.plot(hist, color=col)
        plt.title(title)
        plt.xlabel("Intensité")
        plt.ylabel("Fréquence normalisée")
        plt.grid(alpha=0.3)

    # --- Affichage des images correspondantes en dessous ---
    for i, (img, title) in enumerate(imgs_to_show, 1):
        plt.subplot(2, 3, i + 3)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # ÉTAPE 4 : RECHERCHE PAR HISTOGRAMMES DCT (DOMAINE FRÉQUENTIEL)
    # -------------------------------------------------------------------------
    print("=== Méthode 3 : Histogrammes dans le domaine fréquentiel (DCT RGB amélioré) ===")
    results_dct = compare_by_dct_histogram(query_img, images, metric=cv2.HISTCMP_CORREL)
    show_top_dct_histograms(query_img, images, results_dct, top_n=10, title="Top 10 images similaires (DCT RGB amélioré)")


    # -------------------------------------------------------------------------
    # ÉTAPE 6 : RECHERCHE PAR HISTOGRAMMES DWT (ONDELETTES)
    # -------------------------------------------------------------------------
    print("\n=== Méthode 4 : Histogrammes DWT (ondelettes RGB amélioré) ===")
    show_dwt_subbands_histograms(query_img, wavelet='haar')
    results_dwt = compare_by_dwt_histogram(query_img, images, metric=cv2.HISTCMP_CORREL)
    show_top_dwt_results(query_img, images, results_dwt, top_n=10, title="Top 10 images similaires (DWT RGB amélioré)")


    # =============================================================================
    # UTILISATION DES CORRÉLOGRAMMES POUR L'INDEXATION
    # =============================================================================
    print("\n=== Méthode 5 : Corrélogrammes de couleurs ===")
    results_corr = compare_by_correlogram(query_img, images)

    print("\nTop 10 (méthode Corrélogramme) :")
    for rank, (name, score) in enumerate(results_corr[:10], start=1):
        print(f"{rank:2d}. {name:25s} -> Corrélation = {score:.4f}")

    show_top_similar(query_img, images, results_corr, top_n=10,
                     title="Méthode 5 - Similarité (Corrélogrammes de couleurs)")



