import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path

# Add search paths
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, r"E:\ComfyUI")

import time
import tempfile
import itertools
from typing import *
import atexit
from concurrent.futures import ThreadPoolExecutor
import shutil

import cv2
import torch
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image
import gradio as gr
import safetensors.torch as st
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware

import flex_gemm
flex_gemm.config.AUTOTUNE_MODE = 'never'

import utils3d_moge as utils3d
from moge.utils.vis import colorize_depth, colorize_normal
from moge.utils.gradio_3d_viewer import DepthMap3DViewer
from comfy.ldm.moge.model import MoGeModel

TEMP_DIR = Path(tempfile.gettempdir(), 'moge3_ui')
TEMP_DIR.mkdir(exist_ok=True, parents=True)

MODEL_WEIGHTS_PATH = r"E:\ComfyUI\models\geometry_estimation\moge_3_vitl_fp16.safetensors"

print(f"Loading MoGe-3 from: {MODEL_WEIGHTS_PATH} ...")
t0 = time.time()
sd = st.load_file(MODEL_WEIGHTS_PATH)
model = MoGeModel(sd)
print(f"MoGe-3 loaded in {time.time() - t0:.2f}s! Version: {model.version}")

thread_pool_executor = ThreadPoolExecutor(max_workers=2)

def delete_later(path: Union[str, os.PathLike], delay: int = 600):
    def _delete():
        try:
            p = Path(path)
            if p.is_file():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    def _wait_and_delete():
        time.sleep(delay)
        _delete()
    thread_pool_executor.submit(_wait_and_delete)
    atexit.register(_delete)

def run_inference(image: np.ndarray, max_size: int = 1024, resolution_level: str = 'High', 
                  apply_mask: bool = True, remove_edge: bool = True, refine_steps: int = 3, 
                  request: gr.Request = None):
    if image is None:
        return None, None, None, None, "", None, "", "", ""

    larger_size = max(image.shape[:2])
    if larger_size > max_size:
        scale = max_size / larger_size
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]
    resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

    t0 = time.time()
    with torch.no_grad():
        output = model.infer(
            image_tensor,
            resolution_level=resolution_level_int,
            apply_mask=apply_mask,
            refine_steps=int(refine_steps)
        )
    infer_duration = time.time() - t0

    points = output['points'][0].cpu().numpy()
    depth = output['depth'][0].cpu().numpy()
    mask = output['mask'][0].cpu().numpy()
    intrinsics = output['intrinsics'][0].cpu().numpy()
    normal = output.get('normal')
    if normal is not None:
        normal = normal[0].cpu().numpy()

    if remove_edge:
        edge = utils3d.np.depth_map_edge(depth, ltol=0.01)
        mask_cleaned = mask & ~edge
    else:
        mask_cleaned = mask

    results = {
        'points': points,
        'depth': depth,
        'mask': mask,
        'intrinsics': intrinsics,
        'normal': normal,
        'mask_cleaned': mask_cleaned,
        'image': image
    }

    # Depth & Normal visualizations
    depth_vis = colorize_depth(depth)
    normal_vis = colorize_normal(normal) if normal is not None else None
    mask_vis = (mask_cleaned.astype(np.uint8) * 255)

    # 3D Mesh and Point Cloud generation
    faces, vertices, vertex_colors, vertex_uvs = utils3d.np.build_mesh_from_map(
        points,
        image.astype(np.float32) / 255.0,
        utils3d.np.uv_map((height, width)),
        mask=mask_cleaned,
        tri=True
    )
    vertices = vertices * np.array([1, -1, -1], dtype=np.float32) 
    vertex_uvs = vertex_uvs * np.array([1, -1], dtype=np.float32) + np.array([0, 1], dtype=np.float32)

    session_id = getattr(request, 'session_hash', 'session_default') if request else 'session_default'
    output_path = Path(TEMP_DIR, f"{session_id}_{int(time.time()*1000)}")
    output_path.mkdir(exist_ok=True, parents=True)

    # Export GLB (Textured mesh)
    glb_path = output_path / 'mesh.glb'
    trimesh.Trimesh(
        vertices=vertices * np.array([-1, 1, -1], dtype=np.float32),
        faces=faces, 
        visual=trimesh.visual.texture.TextureVisuals(
            uv=vertex_uvs, 
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(image),
                metallicFactor=0.2,
                roughnessFactor=0.8
            )
        ),
        process=False
    ).export(glb_path)

    # Export PLY (Colored mesh)
    ply_path = output_path / 'mesh.ply'
    trimesh.Trimesh(
        vertices=vertices, 
        faces=faces, 
        vertex_colors=vertex_colors,
        process=False
    ).export(ply_path)

    # Export Point Cloud PLY
    pcd_ply_path = output_path / 'pointcloud.ply'
    trimesh.PointCloud(
        vertices=vertices, 
        colors=vertex_colors,
    ).export(pcd_ply_path)

    # Images / Raw depth
    cv2.imwrite(str(output_path / 'depth.png'), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
    if normal_vis is not None:
        cv2.imwrite(str(output_path / 'normal.png'), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

    files = [str(glb_path), str(ply_path), str(pcd_ply_path), str(output_path / 'depth.png')]
    if normal_vis is not None:
        files.append(str(output_path / 'normal.png'))

    delete_later(output_path, delay=600)

    # FOV calculation
    fov_x, fov_y = utils3d.np.intrinsics_to_fov(intrinsics)
    fov_x, fov_y = np.rad2deg([fov_x, fov_y])

    # 3D Viewer HTML payload
    viewer_html_value = point_cloud_viewer.build(
        depth, image, intrinsics, output_path, mask=mask_cleaned,
    )

    viewer_message = f"**推理完成！耗时: {infer_duration:.2f}s** (稀疏 3D 细化步数: {refine_steps})"
    depth_message = f"图像分辨率: {width}x{height} | 水平视角 FOV: {fov_x:.1f}° | 垂直视角 FOV: {fov_y:.1f}°"

    return (
        results,
        depth_vis,
        normal_vis,
        mask_vis,
        viewer_html_value,
        files,
        f"**水平视场角 (H-FOV): {fov_x:.1f}° | 垂直视场角 (V-FOV): {fov_y:.1f}°**",
        viewer_message,
        depth_message
    )

def reset_measure(results: Optional[Dict[str, np.ndarray]]):
    if results is None:
        return [None, [], ""]
    return [results['image'], [], ""]

def measure(results: Optional[Dict[str, np.ndarray]], measure_points: List[Tuple[int, int]], event: gr.SelectData):
    if results is None:
        return [None, [], ""]
    point2d = event.index[0], event.index[1]
    measure_points.append(point2d)

    image = results['image'].copy()
    for p in measure_points:
        image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

    depth_text = ""
    for i, p in enumerate(measure_points):
        d = results['depth'][p[1], p[0]]
        depth_text += f"**点 P{i + 1} 深度: {d:.2f} 米** "

    if len(measure_points) == 2:
        point1, point2 = measure_points
        image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=2)
        p1_3d = results['points'][point1[1], point1[0]]
        p2_3d = results['points'][point2[1], point2[0]]
        distance = np.linalg.norm(p1_3d - p2_3d)
        measure_points = []
        distance_text = f" -> **两点空间真实距离: {distance:.2f} 米**"
        text = depth_text + distance_text
        return [image, measure_points, text]
    else:
        return [image, measure_points, depth_text]

print("Initializing Gradio UI...")
custom_css = """
.gradio-container { max-width: 95% !important; }
"""

with gr.Blocks(title="MoGe-3 原生 Python 3D 几何查看器") as demo:
    gr.Markdown(
        """
        # MoGe-3 交互式 3D 几何估计系统 (Native Python & Gradio UI)
        **模型架构**: MoGe-3 (ViT-L + 3D 稀疏体素自引导细化 UNet) | **加速引擎**: Triton FlexGEMM V2 (FP16)
        """
    )
    results = gr.State(value=None)
    measure_points = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=4):
            input_image = gr.Image(type="numpy", image_mode="RGB", label="输入图像 (Input Image)")
            with gr.Accordion(label="推理参数设置 (Settings)", open=True):
                refine_steps = gr.Slider(
                    minimum=0, maximum=7, value=3, step=1, 
                    label="稀疏 3D 细化步数 (Refine Steps)", 
                    info="0 = 关闭细化(基础速度); 3 = 推荐平衡值; 5~7 = 极致锐化栏杆/树叶/细薄结构"
                )
                resolution_level = gr.Dropdown(
                    ['Low', 'Medium', 'High', 'Ultra'], 
                    value='High', 
                    label="推理分辨率等级 (Resolution Level)"
                )
                max_size_input = gr.Slider(
                    minimum=384, maximum=2048, value=1024, step=64, 
                    label="图像最长边限制 (Max Image Size)"
                )
                apply_mask = gr.Checkbox(value=True, label="剔除无效区域/天空 (Apply Mask)")
                remove_edges = gr.Checkbox(value=True, label="过滤深度突变断层长边 (Remove Edge Discontinuities)")
            
            submit_btn = gr.Button("🚀 开始 3D 重建 (Generate 3D)", variant="primary", size="lg")

        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.Tab("🌐 3D 交互查看器 (3D View)"):
                    viewer_message = gr.Markdown("")
                    point_cloud_viewer = DepthMap3DViewer(height="620px")
                    fov_text = gr.Markdown("")
                with gr.Tab("🗺️ 深度图 (Depth)"):
                    depth_message = gr.Markdown("")
                    depth_map = gr.Image(type="numpy", label="彩色深度图 (Turbo Colormap)", format='png', interactive=False)
                with gr.Tab("🧭 表面法线 (Normal)"):
                    normal_map = gr.Image(type="numpy", label="表面法线图 (Surface Normals)", format='png', interactive=False)
                with gr.Tab("🎭 有效几何掩码 (Mask)"):
                    mask_map = gr.Image(type="numpy", label="有效掩码 (Valid Mask)", format='png', interactive=False)
                with gr.Tab("📏 3D 空间测距 (Measure)"):
                    gr.Markdown("### 在画面上连续点击任意两点，计算相机空间中的真实公制物理距离 (米)")
                    measure_image = gr.Image(type="numpy", show_label=False, format='webp', interactive=False, sources=[])
                    measure_text = gr.Markdown("请点击图像测量...")
                with gr.Tab("💾 3D 模型下载 (Download)"):
                    files = gr.File(type='filepath', label="导出的 3D 文件与图纸 (GLB / PLY / PNG)")

    example_dir = Path(r"E:\MoGe\example_images")
    if example_dir.exists():
        example_images = sorted(list(itertools.chain(*[example_dir.glob(f"*.{ext}") for ext in ['jpg', 'png', 'jpeg']])))
        if example_images:
            gr.Examples(
                examples=[str(p) for p in example_images],
                inputs=input_image,
                label="内置测试示例图片 (点击即可直接载入测试)"
            )

    submit_btn.click(
        fn=lambda: [None, None, None, None, "", None, "", "⏳ 正在进行 MoGe-3 推理与稀疏 3D 体素细化，请稍候...", ""],
        outputs=[results, depth_map, normal_map, mask_map, point_cloud_viewer, files, fov_text, viewer_message, depth_message]
    ).then(
        fn=run_inference,
        inputs=[input_image, max_size_input, resolution_level, apply_mask, remove_edges, refine_steps],
        outputs=[results, depth_map, normal_map, mask_map, point_cloud_viewer, files, fov_text, viewer_message, depth_message]
    ).then(
        fn=reset_measure,
        inputs=[results],
        outputs=[measure_image, measure_points, measure_text]
    )

    measure_image.select(
        fn=measure,
        inputs=[results, measure_points],
        outputs=[measure_image, measure_points, measure_text]
    )

def main():
    port = 7860
    print(f"Launching MoGe-3 Gradio UI on http://127.0.0.1:{port} ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        css=custom_css,
        share=False,
        allowed_paths=[str(TEMP_DIR)],
        app_kwargs={
            "middleware": [
                Middleware(
                    GZipMiddleware,
                    minimum_size=1024,
                    compresslevel=6,
                )
            ]
        },
    )

if __name__ == '__main__':
    main()
