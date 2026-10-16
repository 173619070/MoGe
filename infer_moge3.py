import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path

# Add search paths
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, r"E:\ComfyUI")

import argparse
import time
import cv2
import torch
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image
import safetensors.torch as st

import flex_gemm
flex_gemm.config.AUTOTUNE_MODE = 'never'

import utils3d_moge as utils3d
from moge.utils.vis import colorize_depth, colorize_normal
from comfy.ldm.moge.model import MoGeModel

def infer_image(
    model: MoGeModel,
    image_path: str,
    output_dir: str,
    max_size: int = 1024,
    resolution_level: int = 9,
    refine_steps: int = 3,
    apply_mask: bool = True,
    remove_edges: bool = True,
    export_glb: bool = True,
    export_ply: bool = True,
    device: str = "cuda"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img_rgb.shape[:2]

    img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    print(f"Running inference for {image_path} (size {w}x{h}, refine_steps={refine_steps})...")
    t0 = time.time()
    with torch.no_grad():
        output = model.infer(
            img_tensor,
            resolution_level=resolution_level,
            refine_steps=refine_steps,
            apply_mask=apply_mask
        )
    print(f"Done in {time.time() - t0:.2f}s!")

    points = output['points'][0].cpu().numpy()
    depth = output['depth'][0].cpu().numpy()
    mask = output['mask'][0].cpu().numpy()
    intrinsics = output['intrinsics'][0].cpu().numpy()
    normal = output.get('normal')
    if normal is not None:
        normal = normal[0].cpu().numpy()

    if remove_edges:
        edge = utils3d.np.depth_map_edge(depth, ltol=0.01)
        mask_cleaned = mask & ~edge
    else:
        mask_cleaned = mask

    # 1. Visualizations
    depth_vis = colorize_depth(depth)
    cv2.imwrite(str(out_dir / "depth.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    if normal is not None:
        normal_vis = colorize_normal(normal)
        cv2.imwrite(str(out_dir / "normal.png"), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "mask.png"), mask_cleaned.astype(np.uint8) * 255)

    # 2. 3D Mesh
    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        img_rgb.astype(np.float32) / 255.0,
        utils3d.np.uv_map((h, w)),
        mask=mask_cleaned,
        tri=True
    )
    vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
    vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)

    stem = Path(image_path).stem
    if export_glb:
        glb_path = out_dir / f"{stem}_mesh.glb"
        trimesh.Trimesh(
            vertices=vertices * np.array([-1, 1, -1], dtype=np.float32),
            faces=faces, 
            visual=trimesh.visual.texture.TextureVisuals(
                uv=vertex_uvs, 
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray(img_rgb),
                    metallicFactor=0.2,
                    roughnessFactor=0.8
                )
            ),
            process=False
        ).export(glb_path)
        print(f"Exported: {glb_path}")

    if export_ply:
        ply_mesh_path = out_dir / f"{stem}_mesh.ply"
        trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            vertex_colors=vertex_colors,
            process=False
        ).export(ply_mesh_path)
        print(f"Exported: {ply_mesh_path}")

        pcd_path = out_dir / f"{stem}_pointcloud.ply"
        trimesh.PointCloud(
            vertices=vertices, 
            colors=vertex_colors,
        ).export(pcd_path)
        print(f"Exported: {pcd_path}")

def main():
    parser = argparse.ArgumentParser(description="MoGe-3 Native Python Inference")
    parser.add_argument("-i", "--input", type=str, default=r"E:\MoGe\example_images\01_HouseIndoor.jpg", help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default=r"E:\MoGe\output", help="Output directory")
    parser.add_argument("-w", "--weights", type=str, default=r"E:\ComfyUI\models\geometry_estimation\moge_3_vitl_fp16.safetensors", help="Path to weights")
    parser.add_argument("--refine_steps", type=int, default=3, help="Sparse 3D refinement steps (default: 3)")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum image dimension (default: 1024)")
    parser.add_argument("--resolution_level", type=int, default=9, help="Inference resolution level (0-9)")
    parser.add_argument("--no_mask", action="store_true", help="Do not apply mask")
    parser.add_argument("--no_remove_edges", action="store_true", help="Do not remove edge discontinuities")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading weights from {args.weights}...")
    sd = st.load_file(args.weights)
    model = MoGeModel(sd)
    print("Model ready!")

    infer_image(
        model=model,
        image_path=args.input,
        output_dir=args.output,
        max_size=args.max_size,
        resolution_level=args.resolution_level,
        refine_steps=args.refine_steps,
        apply_mask=not args.no_mask,
        remove_edges=not args.no_remove_edges,
        export_glb=True,
        export_ply=True,
        device=device
    )

if __name__ == '__main__':
    main()
