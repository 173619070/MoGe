import sys
import os
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import safetensors.torch as st
import trimesh
from PIL import Image

# Ensure path to ComfyUI for model definition and moge for utilities
sys.path.insert(0, r"E:\ComfyUI")
sys.path.insert(0, r"E:\MoGe")

from comfy.ldm.moge.model import MoGeModel
import utils3d_moge as utils3d
from moge.utils.vis import colorize_depth, colorize_normal

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    weights_path = r"E:\ComfyUI\models\geometry_estimation\moge_3_vitl_fp16.safetensors"
    print(f"Loading MoGe-3 weights from {weights_path}...")
    t0 = time.time()
    sd = st.load_file(weights_path)
    model = MoGeModel(sd)
    print(f"Model loaded in {time.time() - t0:.2f}s, version: {model.version}")
    
    img_path = r"E:\MoGe\example_images\01_HouseIndoor.jpg"
    print(f"Reading image: {img_path}")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize if too large
    max_size = 1024
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img_rgb.shape[:2]
    print(f"Inference image size: {w}x{h}")

    img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
    
    print("Running MoGe-3 inference with 3-step sparse 3D refinement...")
    t_start = time.time()
    with torch.no_grad():
        output = model.infer(img_tensor, resolution_level=9, refine_steps=3, apply_mask=True)
    infer_time = time.time() - t_start
    print(f"Inference completed in {infer_time:.2f}s!")
    
    points = output['points'][0].cpu().numpy()     # (H, W, 3)
    depth = output['depth'][0].cpu().numpy()       # (H, W)
    mask = output['mask'][0].cpu().numpy()         # (H, W)
    intrinsics = output['intrinsics'][0].cpu().numpy() # (3, 3)
    normal = output.get('normal')
    if normal is not None:
        normal = normal[0].cpu().numpy()
        
    out_dir = Path(r"E:\MoGe\test_output")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Clean boundary edges to avoid stretching/pulling
    print("Cleaning depth edges...")
    edge = utils3d.np.depth_map_edge(depth, ltol=0.01)
    mask_cleaned = mask & ~edge
    
    # 2. Build 3D mesh
    print("Building 3D textured mesh...")
    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        img_rgb.astype(np.float32) / 255.0,
        utils3d.np.uv_map((h, w)),
        mask=mask_cleaned,
        tri=True
    )
    vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
    vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)
    
    glb_path = out_dir / "mesh.glb"
    ply_path = out_dir / "mesh.ply"
    
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
    print(f"Exported textured GLB: {glb_path} (Size: {glb_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(ply_path)
    print(f"Exported PLY point/mesh: {ply_path} (Size: {ply_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Visualizations
    depth_vis = colorize_depth(depth)
    cv2.imwrite(str(out_dir / "depth.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    if normal is not None:
        normal_vis = colorize_normal(normal)
        cv2.imwrite(str(out_dir / "normal.png"), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
    
    print("All outputs generated successfully in E:\\MoGe\\test_output!")

if __name__ == '__main__':
    main()
