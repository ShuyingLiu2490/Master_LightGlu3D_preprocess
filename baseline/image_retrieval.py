import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from hloc import extract_features, pairs_from_retrieval


def most_similar_image(reference_dir, query_dir, output_dir):

    feature_dir = output_dir / "features.h5"
    pair_file = output_dir / "pairs.txt"

    output_dir.mkdir(exist_ok=True)

    ref_list_file = output_dir / "reference_list.txt"
    query_list_file = output_dir / "query_list.txt"

    with open(ref_list_file, "w") as f:
        for img in sorted(reference_dir.iterdir()):
            if img.suffix.lower() in [".jpg", ".png", ".jpeg"] and img.stat().st_size != 0:
                f.write(str(img.name) + "\n")

    with open(query_list_file, "w") as f:
        ## TODO: change to query iterator
        f.write(str(sorted(query_dir.iterdir())[0].name) + "\n")
        # for query in sorted(query_dir.iterdir()):
        #     if query.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        #         f.write(str(query.name) + "\n")

    feature_conf = extract_features.confs["netvlad"]  # 'netvlad' 'dir', 'openibl', 'megaloc'
    print("Extracting global features...")

    extract_features.main(
        conf=feature_conf,
        image_list=ref_list_file,
        image_dir= reference_dir,
        feature_path=feature_dir,
    )

    extract_features.main(
        conf=feature_conf,
        image_list=query_list_file,
        image_dir=query_dir,
        feature_path=feature_dir,
    )

    print("Performing image retrieval...")

    pairs_from_retrieval.main(
        descriptors=feature_dir,     
        output=pair_file,               
        num_matched=1,                    
        query_list=query_list_file,        
        db_list=ref_list_file,            
    )

    print("\nThe most similar reference image:")
    with open(pair_file) as f:
        for line in f.readlines():
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_image, matched_image = parts[0], parts[1]
                    print(matched_image)
                else:
                    print(f"Invalid line format: {line.strip()}")

    ## Display the query and matched reference image
    query_img = mpimg.imread(query_dir / query_image)
    match_img = mpimg.imread(reference_dir / matched_image)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(query_img)
    axes[0].set_title(f'Query Image\n{query_image}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
        
    axes[1].imshow(match_img)
    axes[1].set_title(f'Best Match\n{matched_image}', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
        
    result_plot_path = output_dir / "retrieval_result.png"
    plt.savefig(result_plot_path, dpi=150, bbox_inches='tight')
    print(f"Matched image pair saved: {result_plot_path}")
        
    plt.show()

    return matched_image

