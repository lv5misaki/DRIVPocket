import molgrid
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_edge_v2(coordinateset,binary_labels, center,pred, gaussian_radius_multiple=1, radii_multiple=1,mask_threshold=1.0,point_mean = False):
    # Create ground truth tensor
    # 假设你有两个张量，xyz_coordinates 和 binary_labels
    # xyz_coordinates 是一个形状为 (8, N, 3) 的张量，其中 N 是每个集合的点的数量，3 表示 XYZ 坐标的维度
    # binary_labels 是一个形状为 (8, N) 的张量，包含每个集合中点的二进制标签
    def find_centroid(voxel_data: object, threshold: object) -> object:
        indices = np.where(voxel_data >= threshold)
        centroid = np.mean(np.array(indices), axis=1)
        return centroid.astype(int)
    gmaker_img = molgrid.GridMaker(dimension=32, gaussian_radius_multiple=gaussian_radius_multiple, radius_scale=1 * radii_multiple)
    grid_nps = []
    # 输出结果
    for i, points in enumerate(coordinateset):
        grid_p = []
        marked_as_1_indices = torch.nonzero(binary_labels[i] == 1, as_tuple=True)
        points_marked_as_1 = points[marked_as_1_indices[0]]
        center_label =torch.mean(points_marked_as_1)
        for ii,point in enumerate(points_marked_as_1):
            #print(f"集合 {i + 1} 中坐标集合标记为1的点：", points_marked_as_1)
            point = point.unsqueeze(0)
            c2grid = molgrid.Coords2Grid(gmaker_img, center=molgrid.float3(float(center[0][i]), float(center[1][i]), float(center[2][i])))
            origtypes = torch.ones(1, 1)
            radii = torch.ones(1)
            grid_gen = c2grid(point, origtypes, radii)
            grid_np_i = grid_gen.numpy()
            #grid_np = binary_dilation(grid_np[0], cube(3))
            grid_np_i = grid_np_i.astype(float)
            mask = grid_np_i < 0.1
            # print(torch.max(grid_nps))
            grid_np_i[mask] = 0
            grid_p.append(grid_np_i[0])

        grid_p_ = torch.tensor(np.array(grid_p))
        if point_mean:
            center_label = find_centroid(pred[i],0.9)[1:]
        # center_label = find_centroid(pred[i],0.85)
        # grid_p_ = update_and_sum_voxels(grid_p_, center_label[1:])
        grid_p_ = update_and_sum_voxels(grid_p_, center_label)
        grid_nps.append(grid_p_)

    grid_nps = torch.stack(grid_nps,dim=0)
    mask = grid_nps >=mask_threshold
    #print(torch.max(grid_nps))
    grid_nps[mask] = mask_threshold
    mask2 = grid_nps > 0
    return grid_nps, mask2

def update_and_sum_voxels_v2(voxels,voxel_center):
    nonzero_indices = torch.nonzero(voxels, as_tuple=False)

    # 提取非零坐标对应的体素集索引和坐标
    voxel_set_indices = nonzero_indices[:, 0]
    nonzero_coords = nonzero_indices[:, 1:].float()

    # 计算这些坐标与输入坐标的距离
    distances = torch.linalg.norm(nonzero_coords - voxel_center, dim=1)
    max_values, max_indices_flat = voxels.view(voxels.shape[0], -1).max(dim=1)

    # 将扁平化的索引转换为3维坐标
    max_indices = torch.stack(torch.meshgrid(torch.arange(65), torch.arange(65), torch.arange(65)), -1).view(-1, 3)
    max_indices = max_indices[max_indices_flat]
    max_val_distances = torch.linalg.norm(max_indices.float() - voxel_center, dim=1)
    # 扩展max_val_distances到与nonzero_indices相同的数量
    max_val_distances_expanded = max_val_distances[voxel_set_indices]

    # 比较距离
    mask = distances < max_val_distances_expanded

    # 创建一个新的体素张量
    final_voxels = torch.zeros_like(voxels)
    final_voxels[
        voxel_set_indices[mask], nonzero_coords[mask, 0].long(), nonzero_coords[mask, 1].long(), nonzero_coords[
            mask, 2].long()] = 1
    return final_voxels * voxels


def update_and_sum_voxels(voxels, voxel_center):
    """
    处理体素数据并将每个点集的值相加，得到单个体素数组。

    参数:
    voxels (torch.Tensor): 形状为 (n, 64, 64, 64) 的体素数据，其中 n 是点集的数量。
    voxel_center (torch.Tensor): 体素中心点的坐标。

    返回:
    torch.Tensor: 形状为 (1, 64, 64, 64) 的更新后的体素数据。
    """
    def unravel_index(index, shape):
        """将一维索引转换为三维坐标"""
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))


    sum_voxels = torch.zeros((1, 65, 65, 65),dtype=torch.float64)
    num_point_sets = voxels.shape[0]

    for i in range(num_point_sets):


        # 找到中心点到体素中心的距离
        max_pos = torch.argmax(voxels[i])
        center = torch.tensor(unravel_index(max_pos, (65, 65, 65)), dtype=torch.float)
        distance_to_voxel_center = torch.linalg.norm(center - voxel_center)
        # voxels[i] = adjust_voxels_torch(voxels[i],center,distance_to_voxel_center)
        indices = torch.nonzero(voxels[i], as_tuple=False).float()

        # 计算每个点到中心点的距离
        distances = torch.linalg.norm(indices - voxel_center, dim=1)
        mask = (distances <= distance_to_voxel_center * 1 )

        valid_points = indices[mask].long()
        # for idx in valid_points:
        #     sum_voxels[0, idx[0], idx[1], idx[2]] += voxels[i, idx[0], idx[1], idx[2]]
        z, y, x = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
        selected_voxels = voxels[i, z, y, x]
        # 使用 scatter_add_ 累加值到 sum_voxels
        indices_ = z * 65 * 65 + y * 65 + x
        flat_sum_voxels = torch.zeros_like(voxels[0]).view(-1)
        # 使用 scatter_add_ 累加点的值到体素网格
        flat_sum_voxels.index_add_(0, indices_, selected_voxels)
        sum_voxels += flat_sum_voxels.view(1, 65, 65, 65)
        #flat_sum_voxels = flat_sum_voxels.view(1, 65, 65, 65)
    return sum_voxels
        #


def adjust_voxels_torch(voxels_shape, center, max_distance):
    # 确保输入为torch.Tensor

    # 生成每个维度的坐标网格
    z, y, x = torch.meshgrid(torch.arange(65),
                             torch.arange(65),
                             torch.arange(65),
                             indexing='ij')

    # 将坐标网格转换为float32，以进行浮点运算
    x, y, z = x.float(), y.float(), z.float()

    # 计算距离
    distance = torch.sqrt((x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2)
    distance[distance >= max_distance] = 0

    # 计算调整后的值
    adjusted_values = torch.clip(distance / max_distance, 0, 1)
    # 对于距离大于max_distance的点，将值设置为1


    return adjusted_values
    # slice_data = sum_voxels[0, :, 33, :]
    # slice_data = np.squeeze(slice_data)
    # slice_data = slice_data.numpy()
    # #slice_edge = flat_sum_voxels[0, :, 33, :]
    # slice_edge = torch.sum(voxels,dim=0)[ :, 33, :]
    # slice_edge = np.squeeze(slice_edge)
    # slice_edge = slice_edge.numpy()
    # plt.show()
    # # 创建热力图
    #
    # # 添加颜色条
    #
    # # 设置标题和坐标轴标签
    # fig, axes = plt.subplots(1,2, figsize=(15, 10))
    #
    # # 在第一个子图中绘制热力图
    # axes[0].imshow(slice_data, cmap='viridis', origin='lower', interpolation='nearest')
    # axes[0].set_title(f"Heatmap of Voxel Data on Slice 1")
    # axes[0].set_xlabel("X")
    # axes[0].set_ylabel("Y")
    # axes[0].grid(False)
    #
    # # 在第二个子图中绘制热力图
    # axes[1].imshow(slice_edge, cmap='viridis', origin='lower', interpolation='nearest')
    # axes[1].set_title(f"Heatmap of Voxel Data on Slice 2")
    # axes[1].set_xlabel("X")
    # axes[1].set_ylabel("Y")
    # axes[1].grid(False)
    # plt.show()







if __name__ == '__main__':
    import Protein_Dataset
    train_ProteinDataset = Protein_Dataset.TrainScPDB('/home/dbw/PythonProject/DeepPocket_pp/seg_scPDB_train0.types',
                                      data_root="/home/dbw/ScPDB/scPDB", recmolcache="scPDB_new.molcache2",
                                      cache_structs=True)
    ratio = 0.0
    ratio2 = 0.0
    i = 0
    loader = torch.utils.data.DataLoader(train_ProteinDataset)
    for Batch in loader:
        input_tensor, riconv_tensor, mask_tensor, centers, _, feature, edge_pocket, protein_coords = Batch

        edge, mask= get_edge_v2(protein_coords, edge_pocket, centers, mask_tensor, gaussian_radius_multiple=1, radii_multiple=3, mask_threshold=2)
        np.save("/home/dbw/PythonProject/DeepPocket_pp/saveimg/input"+str(i)+".npy", input_tensor.numpy())
        if torch.sum(edge) !=0:
            print(torch.sum(edge * mask_tensor)/torch.sum(edge))
            slice_data = mask_tensor[0, 0, :, 33, :]
            slice_data = np.squeeze(slice_data)
            slice_data = slice_data.numpy()
            input_tensor = input_tensor[0]
            input_tensor = torch.sum(input_tensor, dim=0, keepdim=True)
            slice_edge = edge[ 0, 0, :, 33, :]
            slice_edge = np.squeeze(slice_edge)
            slice_edge = slice_edge.numpy()
            # 创建热力图

            # 添加颜色条

            # 设置标题和坐标轴标签
            fig, axes = plt.subplots(1, 3, figsize=(15, 10))

            # 在第一个子图中绘制热力图
            axes[0].imshow(slice_data, cmap='viridis', origin='lower', interpolation='nearest')
            axes[0].set_title(f"Heatmap of Voxel Data on Slice 1")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].grid(False)

            # 在第二个子图中绘制热力图
            axes[1].imshow(slice_edge, cmap='viridis', origin='lower', interpolation='nearest')
            axes[1].set_title(f"Heatmap of Voxel Data on Slice 2")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].grid(False)
            plt.imshow(slice_data, cmap='viridis', alpha=0.5, origin='lower', interpolation='nearest')
            plt.imshow(slice_edge, cmap='plasma', alpha=0.5, origin='lower', interpolation='nearest')

            # 添加颜色条


            # 设置标题和坐标轴标签
            plt.title("Overlay of Two Heatmaps")
            plt.xlabel("X")
            plt.ylabel("Y")

            plt.show()
            # 调整布局，以确保子图之间不重叠
            plt.tight_layout()
            i=i+1
            if i >3:
                break
            ratio += torch.sum(edge*mask_tensor)/torch.sum(edge)
            ratio2 += torch.sum(mask*mask_tensor)/torch.sum(mask)
    print(ratio / i)
    print(ratio2 / i)

