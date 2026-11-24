"""
Generates sample disaster test images (simulated rubble, collapsed debris, low-light)
for testing ResQ_sense offline without a physical camera or disaster site.
"""

from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_disaster_scene(filename: str = "sample_rubble_scene.jpg"):
    """
    Creates an image simulating a low-light disaster site with concrete rubble textures
    and human-like silhouettes.
    """
    h, w = 720, 1280
    # Background: dark concrete gray with noise
    scene = np.ones((h, w, 3), dtype=np.uint8) * 35
    noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add jagged rubble polygons (collapsed concrete slabs)
    pts1 = np.array([[100, 720], [350, 420], [600, 500], [450, 720]], np.int32)
    pts2 = np.array([[700, 720], [850, 380], [1150, 480], [1200, 720]], np.int32)
    pts3 = np.array([[300, 300], [550, 220], [800, 280], [650, 380]], np.int32)

    cv2.fillPoly(scene, [pts1], (55, 60, 65))
    cv2.fillPoly(scene, [pts2], (65, 70, 75))
    cv2.fillPoly(scene, [pts3], (45, 50, 55))

    # Add simulated cracks / structural debris lines
    for _ in range(25):
        pt_start = (np.random.randint(0, w), np.random.randint(200, h))
        pt_end = (pt_start[0] + np.random.randint(-150, 150), pt_start[1] + np.random.randint(-100, 100))
        cv2.line(scene, pt_start, pt_end, (20, 20, 20), thickness=np.random.randint(1, 3))

    # Draw simulated human silhouette 1: Standing individual (Ambulatory)
    # Head
    cv2.circle(scene, (420, 260), 22, (180, 195, 210), -1)
    # Torso
    cv2.rectangle(scene, (400, 282), (440, 370), (70, 85, 120), -1)
    # Legs
    cv2.rectangle(scene, (405, 370), (418, 450), (40, 45, 60), -1)
    cv2.rectangle(scene, (422, 370), (435, 450), (40, 45, 60), -1)

    # Draw simulated human silhouette 2: Lying/Trapped survivor under debris (Prone)
    # Horizontal body
    cv2.ellipse(scene, (850, 580), (75, 22), 5, 0, 360, (60, 70, 110), -1)
    # Head at left end
    cv2.circle(scene, (775, 578), 18, (175, 190, 205), -1)
    # Partially overlapping debris slab over legs
    pts_trap = np.array([[880, 550], [1020, 520], [1040, 620], [870, 630]], np.int32)
    cv2.fillPoly(scene, [pts_trap], (50, 55, 60))

    # Apply a slight blur and lighting vignette
    scene = cv2.GaussianBlur(scene, (3, 3), 0)

    out_path = OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), scene)
    print(f"[+] Sample disaster scene generated at: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    generate_synthetic_disaster_scene()
