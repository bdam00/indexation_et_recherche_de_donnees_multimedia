import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
from matplotlib.image import imread

class ShapeBasedImageRetrieval:
    def __init__(self, dataset_path, query_image_path):
        self.dataset_path = dataset_path
        self.query_image_path = query_image_path
        self.methods = {
            'ORB': cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=15),
            'SIFT': cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02, edgeThreshold=10),
            'SURF': self._init_surf()  # Initialisation spéciale pour SURF
        }
        self.dataset_descriptors = {}

    def _init_surf(self):
        """Initialise SURF de manière robuste"""
        try:
            # Essayer d'abord avec opencv-contrib
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=300, nOctaves=4, nOctaveLayers=3)
            print("✅ SURF initialisé avec succès (opencv-contrib)")
            return surf
        except AttributeError:
            try:
                # Essayer avec l'interface standard (versions récentes)
                surf = cv2.SURF_create(hessianThreshold=300, nOctaves=4, nOctaveLayers=3)
                print("✅ SURF initialisé avec succès (interface standard)")
                return surf
            except:
                print("❌ SURF non disponible - installation requise")
                print("💡 Exécutez: pip install opencv-contrib-python")
                return None

    def extract_descriptors(self, image_path, method_name):
        """Extrait les descripteurs selon la méthode choisie"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return None, None

            method = self.methods[method_name]
            if method is None:
                return None, None

            # Redimensionner les images trop grandes pour améliorer la cohérence
            h, w = image.shape
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

            keypoints, descriptors = method.detectAndCompute(image, None)

            # Vérification spéciale pour SURF
            if method_name == 'SURF' and descriptors is not None:
                print(f"🔍 SURF: {len(keypoints)} keypoints, {descriptors.shape} descriptors")

            return keypoints, descriptors
        except Exception as e:
            print(f"Erreur lors de l'extraction pour {image_path} avec {method_name}: {e}")
            return None, None

    def index_dataset(self, method_name='ORB'):
        """Indexe toutes les images du dataset"""
        print("Indexation du dataset en cours...")
        self.dataset_descriptors.clear()

        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        if not os.path.exists(self.dataset_path):
            print(f"Le dossier {self.dataset_path} n'existe pas!")
            return

        image_files = [f for f in Path(self.dataset_path).iterdir()
                      if f.is_file() and f.suffix.lower() in supported_formats]

        if not image_files:
            print("Aucune image trouvée dans le dossier!")
            return

        print(f"Trouvé {len(image_files)} images à indexer...")

        successful = 0
        for i, image_file in enumerate(image_files):
            if i % 10 == 0:
                print(f"Indexation {i}/{len(image_files)}...")

            keypoints, descriptors = self.extract_descriptors(str(image_file), method_name)
            if descriptors is not None and len(descriptors) > 10:
                self.dataset_descriptors[str(image_file)] = {
                    'descriptors': descriptors,
                    'keypoints': keypoints
                }
                successful += 1

        print(f"Indexation terminée: {successful}/{len(image_files)} images indexées avec {method_name}")

    def calculate_similarity_score(self, query_descriptors, db_data, method_name):
        """Calcule le score de similarité avec différentes méthodes"""
        db_descriptors = db_data['descriptors']

        # Vérification de base
        if query_descriptors is None or db_descriptors is None:
            return 0

        if len(query_descriptors) < 2 or len(db_descriptors) < 2:
            return 0

        try:
            if method_name == 'ORB':
                # Pour ORB, utiliser BFMatcher avec Hamming distance
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = matcher.knnMatch(query_descriptors, db_descriptors, k=2)

                # Appliquer le ratio test de Lowe
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                score = len(good_matches)

            else:
                # Pour SIFT/SURF, utiliser FLANN based matcher
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(query_descriptors, db_descriptors, k=2)

                # Ratio test de Lowe
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Score basé sur le nombre de bons matches et leur qualité
                if good_matches:
                    distances = [m.distance for m in good_matches]
                    avg_distance = np.mean(distances)
                    score = len(good_matches) * (1.0 / (avg_distance + 1e-6))
                else:
                    score = 0

            # Normaliser par le nombre de keypoints (éviter le biais des images avec beaucoup de points)
            normalization_factor = min(len(query_descriptors), len(db_descriptors))
            if normalization_factor > 0:
                normalized_score = score / normalization_factor
            else:
                normalized_score = score

            return normalized_score

        except Exception as e:
            print(f"Erreur dans le calcul de similarité avec {method_name}: {e}")
            return 0

    def compare_images(self, query_descriptors, method_name):
        """Compare les descripteurs de la requête avec ceux du dataset"""
        similarities = {}

        if query_descriptors is None:
            print("❌ Aucun descripteur de requête disponible")
            return similarities

        for img_path, db_data in self.dataset_descriptors.items():
            try:
                score = self.calculate_similarity_score(query_descriptors, db_data, method_name)
                if score > 0:  # Ne considérer que les images avec au moins une correspondance
                    similarities[img_path] = score
            except Exception as e:
                continue

        return similarities

    def find_similar_images(self, method_name='ORB', top_k=10):
        """Trouve les images les plus similaires"""
        print(f"🔍 Recherche d'images similaires avec {method_name}...")

        if not os.path.exists(self.query_image_path):
            print(f"❌ ERREUR: Image requête non trouvée: {self.query_image_path}")
            return []

        if not os.path.exists(self.dataset_path):
            print(f"❌ ERREUR: Dataset non trouvé: {self.dataset_path}")
            return []

        # Vérification spéciale pour SURF
        if method_name == 'SURF' and self.methods['SURF'] is None:
            print("❌ SURF non disponible - installation requise")
            print("💡 Exécutez: pip install opencv-contrib-python")
            return []

        self.index_dataset(method_name)

        if not self.dataset_descriptors:
            print("❌ Aucun descripteur extrait du dataset!")
            return []

        query_kp, query_desc = self.extract_descriptors(self.query_image_path, method_name)

        if query_desc is None:
            print("❌ Impossible d'extraire les descripteurs de l'image requête")
            return []

        print(f"✅ {len(query_desc)} descripteurs extraits de l'image requête")

        similarities = self.compare_images(query_desc, method_name)

        if not similarities:
            print("❌ Aucune similarité trouvée!")
            return []

        print(f"✅ {len(similarities)} images similaires trouvées")
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities[:top_k]

    def create_result_figure(self, top_results, method_name):
        """Crée la figure de résultats avec le format demandé"""
        fig = plt.figure(figsize=(16, 9))

        # Image requête en haut au centre
        try:
            query_img = imread(self.query_image_path)
            ax_query = plt.subplot2grid((3, 5), (0, 2), colspan=1)
            ax_query.imshow(query_img)

            # Indiquer si SURF est disponible
            surf_status = " (SURF disponible)" if method_name == 'SURF' and self.methods['SURF'] is not None else ""
            ax_query.set_title(f"🟦 Image requête\nMéthode : {method_name}{surf_status}", fontsize=12, color="blue")
            ax_query.axis("off")
        except Exception as e:
            ax_query = plt.subplot2grid((3, 5), (0, 2), colspan=1)
            ax_query.text(0.5, 0.5, f"Erreur\n{Path(self.query_image_path).name}",
                         ha='center', va='center', fontsize=10)
            ax_query.axis("off")

        # Affichage des 10 images similaires
        for idx, (img_path, score) in enumerate(top_results[:10]):
            try:
                img = imread(img_path)
                row = 1 + (idx // 5)
                col = idx % 5

                ax = plt.subplot2grid((3, 5), (row, col))
                ax.imshow(img)
                ax.set_title(f"{Path(img_path).name}\nScore={score:.4f}", fontsize=8)
                ax.axis("off")

                # Encadrement vert pour le meilleur résultat
                if idx == 0:
                    for spine in ax.spines.values():
                        spine.set_color('green')
                        spine.set_linewidth(3)
            except Exception as e:
                ax = plt.subplot2grid((3, 5), (row, col))
                ax.text(0.5, 0.5, f"Erreur\n{Path(img_path).name}",
                       ha='center', va='center', fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        return fig

    def save_and_display_results(self, results, method_name):
        """Sauvegarde les résultats dans un fichier et les affiche"""
        if not results:
            print("❌ Aucun résultat à afficher")
            return

        n_results = len(results)

        # Création de la figure
        fig = self.create_result_figure(results, method_name)

        # Sauvegarde du résultat
        timestamp = int(time.time())
        output_filename = f"resultats_forme_{method_name}_{timestamp}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"💾 Résultats sauvegardés dans: {output_filename}")

        # Affichage console détaillé
        print(f"\n{'='*80}")
        print(f"🎯 TOP {n_results} IMAGES SIMILAIRES - {method_name}")
        print(f"{'='*80}")
        print(f"📁 Image requête: {Path(self.query_image_path).name}")
        print(f"📂 Dataset: {Path(self.dataset_path).name}")
        print(f"💾 Fichier résultats: {output_filename}")
        print(f"{'='*80}")
        for i, (path, score) in enumerate(results):
            rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            print(f"{rank_icon} Score: {score:.6f} | {Path(path).name}")
        print(f"{'='*80}")

        plt.close(fig)

        # Essayer d'ouvrir l'image avec le visualiseur par défaut
        try:
            if os.name == 'posix':
                os.system(f"xdg-open {output_filename}")
            print(f"👀 Ouverture des résultats...")
        except:
            print(f"📁 Le fichier {output_filename} a été créé. Ouvrez-le manuellement.")

def check_opencv_modules():
    """Vérifie quels modules OpenCV sont disponibles"""
    print("🔍 Vérification des modules OpenCV...")

    # Vérifier SIFT
    try:
        sift = cv2.SIFT_create()
        print("✅ SIFT: Disponible")
    except:
        print("❌ SIFT: Non disponible")

    # Vérifier SURF
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        print("✅ SURF (xfeatures2d): Disponible")
    except:
        try:
            surf = cv2.SURF_create(400)
            print("✅ SURF (standard): Disponible")
        except:
            print("❌ SURF: Non disponible - installez opencv-contrib-python")

    # Vérifier ORB
    try:
        orb = cv2.ORB_create()
        print("✅ ORB: Disponible")
    except:
        print("❌ ORB: Non disponible")

    print()

def main():
    # ⚠️⚠️⚠️ MODIFIEZ CES CHEMINS ⚠️⚠️⚠️
    dataset_path = "/home/belahouel/Documents/Master 2/IRDM/TP/TP4/baseforme3"
    query_image_path = "/home/belahouel/Documents/Master 2/IRDM/TP/TP4/req3.jpg"

    print("🚀 SYSTÈME DE RECHERCHE D'IMAGES PAR FORME")
    print("=" * 60)

    # Vérifier les modules disponibles
    check_opencv_modules()

    # Vérification des chemins
    if not os.path.exists(dataset_path):
        print(f"❌ ERREUR: Dataset non trouvé: {dataset_path}")
        return

    if not os.path.exists(query_image_path):
        print(f"❌ ERREUR: Image requête non trouvée: {query_image_path}")
        return

    # Méthodes à tester (selon disponibilité)
    methodes_disponibles = ['ORB', 'SIFT']

    # Vérifier si SURF est disponible
    retrieval_system = ShapeBasedImageRetrieval(dataset_path, query_image_path)
    if retrieval_system.methods['SURF'] is not None:
        methodes_disponibles.append('SURF')
        print("🎯 SURF sera testé")
    else:
        print("💡 Pour activer SURF: pip install opencv-contrib-python")

    print(f"🔧 Méthodes à tester: {', '.join(methodes_disponibles)}")
    print()

    for methode_choisie in methodes_disponibles:
        print(f"🎯 TEST AVEC LA MÉTHODE: {methode_choisie}")
        print("-" * 40)

        # Recherche des images similaires
        start_time = time.time()
        results = retrieval_system.find_similar_images(method_name=methode_choisie, top_k=10)
        end_time = time.time()

        if results:
            print(f"✅ Recherche terminée en {end_time - start_time:.2f} secondes")
            retrieval_system.save_and_display_results(results, methode_choisie)
        else:
            print("❌ Aucun résultat trouvé avec cette méthode.")

        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
