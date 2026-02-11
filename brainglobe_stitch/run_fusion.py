


from pathlib import Path
from image_mosaic import ImageMosaic
from lr_fuse import   l_r_fuse


right_path = Path(r"D:\James\2026-02-06\N001\stitched\right")
left_path = Path(r"D:\James\2026-02-06\N001\stitched\left")

output_path = right_path.parent / "fused_paths.h5"

fused = l_r_fuse(
    image_mosaic_left = ImageMosaic(left_path),
    image_mosaic_right = ImageMosaic(right_path),
    pyramid_level = 2,
    order = 0,
    save_result=True,
    output_path=output_path,
)
