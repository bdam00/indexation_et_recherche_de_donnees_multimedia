#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.filters import gabor  # pour Gabor (skimage)
from scipy.spatial.distance import euclidean, cityblock, cosine, chebyshev
import pandas as pd
from scipy.ndimage import uniform_filter
from skimage.feature import structure_tensor
import warnings

# -------------------------------------------------------------------
# Tentative d'import de pywt (pour ondelettes). Si absent, on désactive.
# -------------------------------------------------------------------
try:
    import pywt
    PYWT_AVAILABLE = True
except Exception:
    PYWT_AVAILABLE = False

# ----------------------------
# ---------- HARALICK ----------
# ----------------------------
# (Conserve exactement l'implémentation que tu avais)
def haralick_features(image_path_or_array, mode="RGB"):
    """
    Haralick features. Prend en entrée soit un chemin (str) soit un image-array numpy.
    Si image_path_or_array est un array, il doit être une image (H,W) ou (H,W,3).
    Retour : vecteur numpy
    """
    # Si on reçoit un chemin, on lit l'image
    if isinstance(image_path_or_array, str):
        img = imread(image_path_or_array)
    else:
        img = image_path_or_array

    if mode == "GRAY":
        img_gray = rgb2gray(img)
        img_gray = (img_gray * 255 / np.max(img_gray + 1e-10)).astype(np.uint8)
        glcm = graycomatrix(img_gray, distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        P = glcm / np.sum(glcm)
        entropy = -np.sum(P * np.log(P + 1e-10))
        return np.array([energy, contrast, correlation, homogeneity, entropy])

    else:
        if mode == "RGB":
            conv_img = img
        elif mode == "HSV":
            conv_img = rgb2hsv(img)
        elif mode == "LAB":
            conv_img = rgb2lab(img)
        else:
            raise ValueError("Mode inconnu. Utilisez GRAY, RGB, HSV ou LAB.")

        conv_img = (conv_img * 255 / np.max(conv_img + 1e-10)).astype(np.uint8)
        features_total = []

        for channel in range(3):
            ch = conv_img[:, :, channel]
            glcm = graycomatrix(ch, distances=[1],
                                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            P = glcm / np.sum(glcm)
            entropy = -np.sum(P * np.log(P + 1e-10))
            features_total.extend([energy, contrast, correlation, homogeneity, entropy])

        return np.array(features_total)


# ------------------------------------
# ---------- TAMURA (single-channel) ----------
# ------------------------------------
def tamura_features_single_channel(img_chan):
    """
    Tamura features computed on a 2D array (single channel).
    Returns 6-descriptors for that channel.
    Implementation choices:
      - coarseness : chosen_sizes average across scales [3,5,9,17]
      - contrast    : sigma / (mu4)^(1/4)
      - directionality: 1 - normalized entropy of gradient angle histogram
      - line-likeness : mean of (l1-l2)/(l1+l2) from structure tensor
      - regularity : inverse of std of block means (block size 16)
      - roughness : coarseness + contrast (approx)
    """
    img_float = img_chan.astype(np.float64)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
    h, w = img_float.shape

    # 1) coarseness (multi-scale local variance)
    window_sizes = [3, 5, 9, 17]
    local_vars = []
    for s in window_sizes:
        mean = uniform_filter(img_float, size=s, mode='reflect')
        mean_sq = uniform_filter(img_float**2, size=s, mode='reflect')
        var = mean_sq - mean**2
        local_vars.append(var)
    stacked = np.stack(local_vars, axis=-1)
    argmax = np.argmax(stacked, axis=-1)
    chosen_sizes = np.array([window_sizes[k] for k in argmax.flatten()]).reshape(h, w)
    coarseness = np.mean(chosen_sizes)

    # 2) contrast (Tamura)
    mu = np.mean(img_float)
    sigma = np.std(img_float)
    mu4 = np.mean((img_float - mu) ** 4)
    contrast_t = sigma / ((mu4 ** 0.25) + 1e-10)

    # 3) directionality (histogram of gradient orientations weighted by magnitude)
    gy, gx = np.gradient(img_float)
    magnitude = np.hypot(gx, gy)
    angles = np.arctan2(gy, gx)
    angles = np.mod(angles, np.pi)
    nbins = 16
    hist, _ = np.histogram(angles.flatten(), bins=nbins, range=(0, np.pi), weights=magnitude.flatten())
    p = hist / (np.sum(hist) + 1e-10)
    entropy_ang = -np.sum([pi * np.log(pi + 1e-10) for pi in p]) / np.log(nbins + 1e-10)
    directionality = 1.0 - entropy_ang

    # 4) line-likeness (structure tensor eigenvalue ratio)
    Axx, Axy, Ayy = structure_tensor(img_float, sigma=1.0)
    tmp = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy ** 2)
    l1 = (Axx + Ayy + tmp) / 2
    l2 = (Axx + Ayy - tmp) / 2
    eps = 1e-10
    ratio = (l1 - l2) / (l1 + l2 + eps)
    line_likeness = np.nanmean(np.abs(ratio))

    # 5) regularity (inverse of std of block means)
    blk_size = 16
    ph = int(np.ceil(h / blk_size) * blk_size)
    pw = int(np.ceil(w / blk_size) * blk_size)
    padded = np.zeros((ph, pw))
    padded[:h, :w] = img_float
    block_means = []
    for i in range(0, ph, blk_size):
        for j in range(0, pw, blk_size):
            blk = padded[i:i+blk_size, j:j+blk_size]
            block_means.append(np.mean(blk))
    block_means = np.array(block_means)
    reg = 1.0 / (1.0 + np.std(block_means))

    # 6) roughness
    roughness = coarseness + contrast_t

    return np.array([coarseness, contrast_t, directionality, line_likeness, reg, roughness])


def tamura_features(image_or_array):
    """
    Wrapper : accepts a path or an image array.
    If given a color image, user should pass channels individually via the caller.
    This function will accept either a str (path) or 2D numpy array.
    """
    if isinstance(image_or_array, str):
        img = imread(image_or_array)
        if img.ndim == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.astype(np.float64)
        return tamura_features_single_channel(img_gray)
    else:
        # assume 2D array
        return tamura_features_single_channel(image_or_array)


# ------------------------------------
# ---------- GABOR (single-channel) - OPTIMIZED ----------
# ------------------------------------
def gabor_features_single_channel(img_chan, n_frequencies=2, n_orientations=4, freq_min=0.15, freq_max=0.35):
    """
    Apply skimage.filters.gabor on a single 2D channel array.
    Optimized defaults:
      - n_frequencies=2  (freqs: linspace(freq_min, freq_max))
      - n_orientations=4 (0,45,90,135 degrees)
    Returns concatenated [mean, variance] for each (freq,theta).

    We use variance (np.var) rather than std to match requested mean+variance.
    """
    img_float = img_chan.astype(np.float64)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)

    freqs = np.linspace(freq_min, freq_max, n_frequencies)
    thetas = np.linspace(0, np.pi, n_orientations, endpoint=False)

    feats = []
    for f in freqs:
        for theta in thetas:
            real, imag = gabor(img_float, frequency=f, theta=theta)
            mag = np.hypot(real, imag)
            feats.append(np.mean(mag))
            feats.append(np.var(mag))  # variance as requested
    return np.array(feats)


def gabor_features(image_or_array, n_frequencies=2, n_orientations=4, freq_min=0.15, freq_max=0.35):
    """
    Wrapper accepting a path or 2D array.
    If given path to color image, caller should split channels and pass one channel at a time.
    """
    if isinstance(image_or_array, str):
        img = imread(image_or_array)
        if img.ndim == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.astype(np.float64)
        return gabor_features_single_channel(img_gray, n_frequencies, n_orientations, freq_min, freq_max)
    else:
        return gabor_features_single_channel(image_or_array, n_frequencies, n_orientations, freq_min, freq_max)


# ------------------------------------
# ---------- WAVELET (single-channel) ----------
# ------------------------------------
def wavelet_features_single_channel(img_chan, wavelet_name='db1', level=1):
    """
    Compute wavelet features on a single 2D channel array using pywt.
    Returns stats on approximation + details: mean, std, energy for each band.
    """
    if not PYWT_AVAILABLE:
        raise ImportError("pywt required for wavelet_features_single_channel.")
    img_float = img_chan.astype(np.float64)
    img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet_name, level=level)
    feats = []
    cA = coeffs[0]
    feats.extend([np.mean(cA), np.std(cA), np.sum(cA**2)])
    for detail_level in coeffs[1:]:
        cH, cV, cD = detail_level
        for band in (cH, cV, cD):
            feats.append(np.mean(band))
            feats.append(np.std(band))
            feats.append(np.sum(band**2))
    return np.array(feats)


def wavelet_features(image_or_array, wavelet_name='db1', level=1):
    """
    Wrapper accepting path or 2D array.
    """
    if not PYWT_AVAILABLE:
        raise ImportError("pywt required for wavelet_features.")
    if isinstance(image_or_array, str):
        img = imread(image_or_array)
        if img.ndim == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img.astype(np.float64)
        return wavelet_features_single_channel(img_gray, wavelet_name, level)
    else:
        return wavelet_features_single_channel(image_or_array, wavelet_name, level)


# --------------------------------------------------------
# Helper : compute features for a given method and color-space
# --------------------------------------------------------
def compute_features_for_image_in_space(image_path, method_name, space, method_kwargs):
    """
    Compute features for a single image according to method and requested color space.
    method_name: 'HARALICK', 'TAMURA', 'GABOR', 'WAVELET'
    space: 'GRAY','RGB','HSV','LAB'
    method_kwargs: kwargs passed to the underlying method (n_frequencies, etc.)

    For HARALICK we delegate to haralick_features(..., mode=space).
    For other methods we compute per-channel features and concatenate:
      - RGB: split to R,G,B channels
      - HSV: convert rgb2hsv and split H,S,V
    Returns a 1D numpy array (concatenated features).
    """
    # HARALICK handles spaces internally (including GRAY/LAB)
    if method_name == 'HARALICK':
        return haralick_features(image_path, mode=space)

    # For other methods we only accept RGB and HSV (as requested)
    if space not in ('RGB', 'HSV'):
        raise ValueError(f"Method {method_name} is only applied on RGB/HSV in this pipeline (got {space})")

    img = imread(image_path)
    if img.ndim != 3:
        # If grayscale image provided, replicate channel for processing
        img = np.stack([img] * 3, axis=-1)

    if space == 'RGB':
        conv = img  # (H,W,3)
    else:
        conv = rgb2hsv(img)

    # For each channel compute method-specific features (function expects 2D array)
    channel_feats = []
    for c in range(3):
        chan = conv[:, :, c]
        if method_name == 'TAMURA':
            feats = tamura_features(chan)  # single-channel tamura
        elif method_name == 'GABOR':
            feats = gabor_features(chan,
                                   n_frequencies=method_kwargs.get('n_frequencies', 2),
                                   n_orientations=method_kwargs.get('n_orientations', 4),
                                   freq_min=method_kwargs.get('freq_min', 0.15),
                                   freq_max=method_kwargs.get('freq_max', 0.35))
        elif method_name == 'WAVELET':
            feats = wavelet_features(chan,
                                     wavelet_name=method_kwargs.get('wavelet_name', 'db1'),
                                     level=method_kwargs.get('level', 1))
        else:
            raise ValueError("Méthode inconnue.")
        channel_feats.append(feats)
    # Concatenate channel-wise
    concat = np.concatenate([f.flatten() for f in channel_feats])
    return concat


# ============================================================
# Extraction over folder (uses compute_features_for_image_in_space)
# ============================================================
def extract_features_from_folder_by_space(folder_path, method_name, space, **method_kwargs):
    """
    Parcourt le dossier et calcule le vecteur de descripteurs selon method_name et espace.
    Retour : features_db (list of np arrays), image_names
    """
    features_db = []
    image_names = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, file)
            feats = compute_features_for_image_in_space(img_path, method_name, space, method_kwargs)
            features_db.append(feats)
            image_names.append(file)
            print(f"[INFO] Extraction ({method_name} | {space}) terminée pour {file}")
    return np.array(features_db), image_names


# ============================================================
# Distance chooser
# ============================================================
def choose_distance_function():
    print("\n===== Sélection de la méthode de calcul de distance =====")
    print("1️⃣  Euclidienne  (L2)")
    print("2️⃣  Manhattan    (L1)")
    print("3️⃣  Cosine       (orientation)")
    print("4️⃣  Chebyshev    (distance max dimensionnelle)")
    choice = input("\n👉 Entrez le numéro correspondant à la distance souhaitée [1-4] : ")

    if choice == "1":
        return euclidean, "Euclidienne"
    elif choice == "2":
        return cityblock, "Manhattan"
    elif choice == "3":
        return cosine, "Cosine"
    elif choice == "4":
        return chebyshev, "Chebyshev"
    else:
        print("[⚠️] Choix invalide — utilisation de la distance Euclidienne par défaut.")
        return euclidean, "Euclidienne"


# ============================================================
# Comparison function (handles differing vector sizes by padding)
# ============================================================
def find_similar_images_from_features(query_features, db_features, image_names, distance_func, top_k=10):
    distances = []
    for feat in db_features:
        if feat.shape != query_features.shape:
            la = max(feat.size, query_features.size)
            a = np.zeros(la); b = np.zeros(la)
            a[:feat.size] = feat.flatten()
            b[:query_features.size] = query_features.flatten()
            d = distance_func(a, b)
        else:
            d = distance_func(query_features, feat)
        distances.append(d)
    results = list(zip(image_names, distances))
    results.sort(key=lambda x: x[1])
    df = pd.DataFrame(results, columns=["Image", "Distance"])
    return df, results[:top_k]


# ============================================================
# Create result figure (unchanged)
# ============================================================
def create_result_figure(query_path, dataset_path, top_results, mode_label, distance_name):
    fig = plt.figure(figsize=(16, 9))
    query_img = imread(query_path)
    ax_query = plt.subplot2grid((3, 5), (0, 2), colspan=1)
    ax_query.imshow(query_img)
    ax_query.set_title(f"🟦 Image requête\nEspace : {mode_label}", fontsize=12, color="blue")
    ax_query.axis("off")

    for idx, (img_name, dist) in enumerate(top_results[:10]):
        img_path = os.path.join(dataset_path, img_name)
        img = imread(img_path)
        row = 1 + (idx // 5)
        col = idx % 5
        ax = plt.subplot2grid((3, 5), (row, col))
        ax.imshow(img)
        ax.set_title(f"{img_name}\nDist={dist:.2f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ============================================================
# ====================== MAIN ==============================
# ============================================================
if __name__ == "__main__":
    # ---------- chemins (à adapter si besoin) ----------
    query_image = "/home/belahouel/Documents/Master 2/IRDM/TP/TP3/req2.jpg"
    dataset_folder = "/home/belahouel/Documents/Master 2/IRDM/TP/TP3/texture2"

    # ---------- 1) Choix de la méthode d'extraction ----------
    print("===== SÉLECTION DU DESCRIPTEUR DE TEXTURE =====")
    print("1 - Haralick (GLCM)  (conserve la version existante)")
    print("2 - Tamura (6 descripteurs)")
    print("3 - Gabor (bank of filters)  - paramétrable (mais par défaut simple)")
    print("4 - Ondelette (DWT) - nécessite pywt (pywavelets)")

    method_choice = input("👉 Entrez votre choix [1-4] : ").strip()

    if method_choice == "1":
        method_name = 'HARALICK'
        # Haralick: we will run all four spaces automatically
        method_kwargs = {}
    elif method_choice == "2":
        method_name = 'TAMURA'
        method_kwargs = {}
    elif method_choice == "3":
        method_name = 'GABOR'
        # paramètres demandés à l'utilisateur (defaults optimized)
        n_freq = input("Nombre de fréquences (ex 2) [default 2]: ").strip()
        n_orient = input("Nombre d'orientations (ex 4) [default 4]: ").strip()
        try:
            n_freq = int(n_freq) if n_freq else 2
            n_orient = int(n_orient) if n_orient else 4
        except:
            n_freq, n_orient = 2, 4
        method_kwargs = {'n_frequencies': n_freq, 'n_orientations': n_orient, 'freq_min': 0.15, 'freq_max': 0.35}
    elif method_choice == "4":
        if not PYWT_AVAILABLE:
            print("\n[⚠️] PyWavelets (pywt) n'est pas installé — l'option Ondelette est désactivée.")
            print("Pour l'activer, installez pywavelets : pip install pywavelets")
            raise SystemExit("Terminer l'exécution (pywt manquant).")
        method_name = 'WAVELET'
        level_input = input("Niveau de décomposition (1-3 recommandé) [default 1]: ").strip()
        wave_name = input("Nom de l'ondelette (ex: db1, db2, sym4) [default db1]: ").strip()
        try:
            level_input = int(level_input) if level_input else 1
        except:
            level_input = 1
        if not wave_name:
            wave_name = 'db1'
        method_kwargs = {'level': level_input, 'wavelet_name': wave_name}
    else:
        print("[⚠️] Choix invalide. On utilisera HARALICK par défaut.")
        method_name = 'HARALICK'
        method_kwargs = {}

    # ---------- 2) Choix de la distance ----------
    distance_func, distance_name = choose_distance_function()

    # ---------- 3) Choix du nombre top_k ----------
    top_k_input = input("Combien de résultats top_k afficher ? [default 10]: ").strip()
    try:
        top_k = int(top_k_input) if top_k_input else 10
    except:
        top_k = 10

    print(f"\n==> Méthode : {method_name} | Distance : {distance_name} | top_k = {top_k}")

    # ---------- Define spaces per method (automatic) ----------
    # Haralick: run on GRAY, RGB, HSV, LAB
    # Other methods: run on RGB and HSV only (as requested)
    if method_name == 'HARALICK':
        spaces = ['GRAY', 'RGB', 'HSV', 'LAB']
    else:
        spaces = ['RGB', 'HSV']

    # ---------- Data structures to store results ----------
    summary = {}            # mean distance per space
    all_figs = []           # store matplotlib figure objects to show at the end
    summary_rows = []       # rows for final summary table

    # ---------- 4) For each space: extract features for database, extract query features, compute distances ----------
    for space in spaces:
        print(f"\n--- Traitement de la méthode {method_name} sur l'espace {space} ---")
        # extraction for dataset
        db_features, image_names = extract_features_from_folder_by_space(dataset_folder, method_name, space, **method_kwargs)

        # extraction for query image
        if method_name == 'HARALICK':
            query_feats = haralick_features(query_image, mode=space)
        else:
            query_feats = compute_features_for_image_in_space(query_image, method_name, space, method_kwargs)

        # compute distances and top results
        df_results, top_results = find_similar_images_from_features(query_feats, db_features, image_names,
                                                                    distance_func, top_k=top_k)

        # store summary stats
        mean_dist = df_results['Distance'].mean()
        summary[space] = mean_dist
        summary_rows.append({
            "Méthode": method_name,
            "Espace": space,
            "Dimension vecteur (ex)": query_feats.size,
            "Distance utilisée": distance_name,
            "Distance moyenne": mean_dist
        })

        # print dataset distances table for this space
        print(f"\n===== Résultats ({method_name} | {space}) =====")
        print(df_results)

        # create and store the figure for this space (but do not show yet)
        fig = create_result_figure(query_image, dataset_folder, top_results, f"{method_name} - {space}", distance_name)
        all_figs.append(fig)

    # ---------- 5) At the end: show all figures at once (they remain open) ----------
    print("\nAffichage de toutes les figures (fenêtres ouvertes jusqu'à fermeture manuelle)...")
    plt.show()

    # ---------- 6) Final comparative table (only spaces used) ----------
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.sort_values(by="Distance moyenne", ascending=True)

    print("\n==================== TABLEAU COMPARATIF FINAL ====================")
    print(df_summary.to_string(index=False))

    # Interpretation: best (minimal mean distance)
    best_row = df_summary.iloc[0]
    print("\n✅ Interprétation :")
    print(f"➡️ Méthode {method_name} — Espace {best_row['Espace']} présente la plus petite distance moyenne "
          f"({best_row['Distance moyenne']:.4f}) avec la distance {distance_name}.")
