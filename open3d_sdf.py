import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh

def load_mesh(dir):
    # Load the .obj modle exported form Blender
    # Compute the bbox of modle, you can adjust margin_value to control the margin.
    # Transfrom it to .ply modle for loading in open3d
    mesh = trimesh.load_mesh(dir)
    bbox_min, bbox_max = mesh.bounds
    margin_value = 0.1
    margin = margin_value * (bbox_max - bbox_min)
    bbox_min = bbox_min - margin
    bbox_max = bbox_max + margin
    x_min, x_max = bbox_min [0], bbox_max[0]
    y_min, y_max = bbox_min [1], bbox_max[1]
    z_min, z_max = bbox_min [2], bbox_max[2]
    mesh.export("001.ply")
    mesh= o3d.io.read_triangle_mesh("001.ply")
    return mesh, x_min, x_max, y_min, y_max, z_min, z_max

def set_sline(res):
    # Generate the sline for printing sdf value
    # z_index is the position of sline in z axis.
    # Return the pts_t (points of the sline)
    xs = np.linspace(x_min, x_max, res)
    ys = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    z_index = 0.5
    z_val = z_min + z_index * (z_max - z_min)
    Z = np.full_like(X, z_val)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pts_t = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    return xs, ys, z_val, pts_t

def compute_sdf(point):
    # compute the sdf value of points
    pttest = np.stack(point, axis=-1).reshape(-1, 3)
    pttest_t = o3d.core.Tensor(pttest, dtype=o3d.core.Dtype.Float32)
    print(scene.compute_signed_distance(pttest_t).numpy())
    
def plt_save(sdf_slice, xs, ys, z_val, save_path="open3d_sdf.png"):
    plt.figure(figsize=(6, 6))
    abs_max = np.max(np.abs(sdf_slice))
    # heatmap figure
    im = plt.imshow(
        sdf_slice.T,
        origin='lower',
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap='coolwarm',
        vmin=-abs_max,                  # 左右对称
        vmax=abs_max
    )

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Signed Distance")

    # zero-level line figure
    plt.contour(
        xs, ys, sdf_slice.T,
        levels=[0],
        colors='black',
        linewidths=1.5
    )

    plt.gca().set_aspect('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Open3D Signed Distance Slice (z={z_val})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ =='__main__':
    mesh, x_min, x_max, y_min, y_max, z_min, z_max = load_mesh(dir="001.obj", )

    voxel_size = 0.01
    sdf_volume = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size
    )
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    res = 256  # resolution of sline
    xs, ys, z_val, pts= set_sline(res)
    sdf = scene.compute_signed_distance(pts).numpy()
    sdf_slice = sdf.reshape(res, res)
    plt_save(sdf_slice, xs = xs, ys = ys, z_val=z_val)
    print("Print over")