import torch
import torch.nn.functional as F
from skimage.morphology import binary_dilation
from skimage.morphology import cube
from molgrid import MolDataset
import molgrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_edge(coordinateset,binary_labels, center):
    # Create ground truth tensor
    # 假设你有两个张量，xyz_coordinates 和 binary_labels
    # xyz_coordinates 是一个形状为 (8, N, 3) 的张量，其中 N 是每个集合的点的数量，3 表示 XYZ 坐标的维度
    # binary_labels 是一个形状为 (8, N) 的张量，包含每个集合中点的二进制标签
    gmaker_img = molgrid.GridMaker(dimension=32, gaussian_radius_multiple=3)
    grid_nps = []
    # 输出结果
    for i, points in enumerate(coordinateset):
        marked_as_1_indices = torch.nonzero(binary_labels[i] == 1, as_tuple=True)
        points_marked_as_1 = points[marked_as_1_indices[0]]
        #print(f"集合 {i + 1} 中坐标集合标记为1的点：", points_marked_as_1)
        c2grid = molgrid.Coords2Grid(gmaker_img, center=molgrid.float3(float(center[0][i]), float(center[1][i]), float(center[2][i])))
        origtypes = torch.ones(points_marked_as_1.numpy().shape[0], 1)
        radii = torch.ones((points_marked_as_1.numpy().shape[0])) * 3
        grid_gen = c2grid(points_marked_as_1, origtypes, radii)
        grid_np = grid_gen.numpy()
        #grid_np = binary_dilation(grid_np[0], cube(3))
        grid_np = grid_np.astype(float)
        if len(center)>1:
            grid_nps.append(grid_np)
        else:
            grid_nps.append(grid_np[0])
    grid_nps = torch.tensor(np.array(grid_nps))
    # print(torch.max(grid_nps))
    mask = grid_nps >=1.2
    #print(torch.max(grid_nps))
    grid_nps[mask] = 0
    mask2 = grid_nps > 0
    return grid_nps, mask2.float()





def get_mask(coordinateset, center, gmaker):
    # Create ground truth tensor
    c2grid = molgrid.Coords2Grid(gmaker, center=center)
    origtypes = torch.ones(coordinateset.coords.tonumpy().shape[0], 1)
    radii = torch.ones((coordinateset.coords.tonumpy().shape[0]))
    grid_gen = c2grid(torch.tensor(coordinateset.coords.tonumpy()), origtypes, radii)
    grid_np = grid_gen.numpy()
    grid_np = binary_dilation(grid_np[0], cube(3))
    grid_np = grid_np.astype(float)
    return torch.tensor(np.expand_dims(grid_np, axis=0))

def pc_normalize(pc, centroid,m=None):
    #centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    #m_ = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    #print(m_)
    if m is not None:
        pc = pc / m
    return pc

def visualize_voxels(voxels, voxles2=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax2 = fig.add_subplot(111, projection='3d')

    # 获取非零值的索引
    indices = np.nonzero(voxels)
    indices = np.transpose(indices)
    from matplotlib.colors import Normalize
    #norm = Normalize(vmin=np.min(voxels), vmax=np.max(voxels))
    #colors = norm(voxels[indices])

    # 画出体素
    ax.scatter(indices[1], indices[2], indices[3], c='r', marker='o')
    if voxles2 is not None:
        indices2 = np.nonzero(voxles2)
        #indices2 = np.transpose(indices2)
        # ax2.scatter(indices2[1], indices2[2], indices2[3], c='b', marker='o')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax2.set_xlabel('X')
    # ax2.set_ylabel('Y')
    # ax2.set_zlabel('Z')

    plt.savefig('/home/dbw/PythonProject/DeepPocket_pp/pictureoutput_plot.png')

class TrainScPDB(MolDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points_num = 400
        self.atomtype = 14
        self.max_dist = 32
        self.resolution = 1
        self.size = int(2 * self.max_dist/self.resolution) + 1
        self.gmaker_mask = molgrid.GridMaker(dimension=32, binary=True, gaussian_radius_multiple=-1, resolution=0.5)
        self.gmaker_img = molgrid.GridMaker(dimension=32, radius_scale=1.0)

    def __getitem__(self, item):
        center, coords, atomtypes, radii, labels, coord_sets = super(TrainScPDB, self).__getitem__(item)
        pocket = coord_sets[-1]
        protein = coord_sets[0]
        input_tensor = torch.zeros([14, self.size, self.size, self.size], dtype=torch.float32)
        mask_tensor = torch.zeros([1, self.size, self.size, self.size], dtype=torch.float32)
        protein_coords = torch.tensor(protein.coords.tonumpy())
        prontein_atomtypes = torch.tensor(protein.type_index.tonumpy())
        protein_radii = torch.tensor(protein.radii.tonumpy())
        protein_coords, prontein_atomtypes, protein_radii = self.knn(protein_coords, torch.tensor(labels[1:]), prontein_atomtypes, protein_radii)
        #protein_coords, prontein_atomtypes, protein_radii = self.filter_points(protein_coords, torch.tensor(labels[1:]), prontein_atomtypes, protein_radii)
        protein_ = molgrid.CoordinateSet(protein_coords.numpy(), prontein_atomtypes.numpy(), protein_radii.numpy(), self.atomtype)
        centers = molgrid.float3(float(labels[1]), float(labels[2]), float(labels[3]))
        self.gmaker_img.forward(centers, protein, input_tensor)
        pocket_edge = self.make_point(torch.tensor(pocket.coords.tonumpy()), protein_coords)
        pocket_edge = torch.unsqueeze(pocket_edge, dim=-1)
        #print(pocket.coords.tonumpy().shape)



        features = self.make_Feature(prontein_atomtypes)
        #rec_grid = data.make_grid(protein_coords, features,
        #                                max_dist=self.max_dist,
        #                                grid_resolution=self.resolution)
        #rec_grid = torch.from_numpy(rec_grid)

        mask_tensor = get_mask(pocket, centers, self.gmaker_mask)
        # mask_tensor = mask_tensor * 0.9 + (1 - mask_tensor) * (1 - 0.9)
        riconv_tensor = pc_normalize(protein_coords,torch.tensor([float(labels[1]), float(labels[2]), float(labels[3])]), 16.25)

        #visualize_voxels(mask_tensor,get_edge(protein_coords,pocket_edge,torch.tensor([float(labels[1]), float(labels[2]), float(labels[3])])))
        #visualize_voxels(input_tensor)
        # if torch.sum(pocket_edge) ==0 and torch.sum(mask_tensor) != 0:
        #     #print(torch.sum(mask_tensor))
        #     mask_tensor = torch.zeros([1, self.size, self.size, self.size], dtype=torch.float32)
        #print(rec_grid.shape)
       # print(mask_tensor.shape)
        #rec_grid, mask_tensor = torch.squeeze(rec_grid, 0), np.squeeze(mask_tensor, 0)
        #rec_grid = rec_grid.permute(3, 0, 1, 2)
        return input_tensor,riconv_tensor, mask_tensor, list([float(labels[1]), float(labels[2]), float(labels[3])]), protein.src,features,pocket_edge,protein_coords

    def set_points_num(self, num: int) -> None:
        self.points_num = num

    def make_point(self, pocket, prob_protein):
        differences = torch.abs(prob_protein[:, None, :] - pocket)

        differences = torch.norm(differences, dim=-1)
        #print(torch.min(differences))
        pocket_edge = torch.any(differences < 3, dim=-1)
        pocket_edge = pocket_edge.float()
        #print(torch.sum(pocket_edge))
        return pocket_edge

    def knn(self, Points, xyz, atomtypes, radii):
        num = self.points_num
        ref_c = torch.stack([xyz] * Points.shape[0], dim=0)
        query_c = Points
        delta = query_c - ref_c
        distances = torch.sqrt(torch.pow(delta, 2).sum(dim=1))
        sorted_dist, indices = torch.sort(distances)
        # if sorted_dist[num] > 16:
        #     print("error!! num out of edge")
        #num = torch.where(sorted_dist>16)[0][0]
        return query_c[indices[:num]], atomtypes[indices[:num]], radii[indices[:num]]

    def filter_points(self,Points, xyz, atomtypes, radii, distance_threshold= 16):
        distances = np.abs(Points - xyz)
        mask = torch.all(distances <= distance_threshold, dim=1)
        return Points[mask] ,atomtypes[mask], radii[mask]

    def make_Feature(self, atomtypes):
        #print(atomtypes.shape[0])
        #Feature = torch.zeros((self.points_num, self.atomtype))
       # print(Feature.shape)
        Feature = F.one_hot(atomtypes.to(torch.int64), num_classes=self.atomtype)
        #print(Feature.shape)
        return Feature



if __name__ == "__main__":
    train_ProteinDataset = TrainScPDB('/home/dbw/PythonProject/DeepPocket_pp/seg_scPDB_train0.types',data_root = "/home/dbw/ScPDB/scPDB", recmolcache = "scPDB_new.molcache2", cache_structs=True)

    loader = torch.utils.data.DataLoader(train_ProteinDataset)
    ratio = 0.0
    ratio2 = 0.0
    i=1
    for Batch in loader:
        input_tensor, riconv_tensor, mask_tensor, centers, _, feature, edge_pocket,protein_coords = Batch
        # edge,mask  = get_edge(protein_coords, edge_pocket, centers)
        # #visualize_voxels(edge)
        # if torch.sum(edge) !=0:
        #     print(torch.sum(edge_pocket))
        #    # print(torch.sum(mask * mask_tensor))
        #     print(torch.sum(mask))
        #     slice_data = mask_tensor[0, :, :,33, :]
        #     slice_data = np.squeeze(slice_data)
        #     slice_data = slice_data.numpy()
        #     input_tensor = input_tensor[0]
        #     input_tensor = torch.sum(input_tensor, dim=0, keepdim=True)
        #     slice_edge = edge[0, :, :,33, :]
        #     slice_edge = np.squeeze(slice_edge)
        #     slice_edge = slice_edge.numpy()
        #     # 创建热力图
        #
        #     # 添加颜色条
        #
        #     # 设置标题和坐标轴标签
        #     fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        #
        #     # 在第一个子图中绘制热力图
        #     axes[0].imshow(slice_data, cmap='viridis', origin='lower', interpolation='nearest')
        #     axes[0].set_title(f"Heatmap of Voxel Data on Slice 1")
        #     axes[0].set_xlabel("X")
        #     axes[0].set_ylabel("Y")
        #     axes[0].grid(False)
        #
        #     # 在第二个子图中绘制热力图
        #     axes[1].imshow(slice_edge, cmap='viridis', origin='lower', interpolation='nearest')
        #     axes[1].set_title(f"Heatmap of Voxel Data on Slice 2")
        #     axes[1].set_xlabel("X")
        #     axes[1].set_ylabel("Y")
        #     axes[1].grid(False)
        #     plt.imshow(slice_data, cmap='viridis', alpha=0.5, origin='lower', interpolation='nearest')
        #     plt.imshow(slice_edge, cmap='plasma', alpha=0.5, origin='lower', interpolation='nearest')
        #
        #     # 添加颜色条
        #
        #
        #     # 设置标题和坐标轴标签
        #     plt.title("Overlay of Two Heatmaps")
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #
        #     plt.show()
        #     # 调整布局，以确保子图之间不重叠
        #     plt.tight_layout()
        #     ratio += torch.sum(edge*mask_tensor)/torch.sum(edge)
        #     ratio2 += torch.sum(mask*mask_tensor)/torch.sum(mask)

        #     i=i+1
        # # if i >5:
        #     break
    print(ratio/i)
    print(ratio2/i)
    # print(ratio/len(loader))
    # print(ratio/len(loader))