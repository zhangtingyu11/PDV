import torch

from . import common_utils, voxel_aggregation_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


def find_num_points_per_part(batch_points, batch_boxes, grid_size):
    """
    Args:
        batch_points: (N, 4)
        batch_boxes: (B, O, 7)
        grid_size: G
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    assert grid_size > 0

    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    points_per_parts = []
    for i in range(batch_boxes.shape[0]):
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        points = batch_points[bs_mask]
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)
        points_in_boxes_mask = box_idxs_of_pts != -1
        box_for_each_point = boxes[box_idxs_of_pts.long()][points_in_boxes_mask]
        xyz_local = points[points_in_boxes_mask] - box_for_each_point[:, 0:3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local[:, None, :], -box_for_each_point[:, 6]
        ).squeeze(dim=1)
        # Change coordinate frame to corner instead of center of box
        xyz_local += box_for_each_point[:, 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        xyz_local = torch.min(xyz_local, box_for_each_point[:, 3:6] - 1e-5)
        xyz_local_grid = (xyz_local // (box_for_each_point[:, 3:6] / grid_size))
        xyz_local_grid = torch.cat((box_idxs_of_pts[points_in_boxes_mask].unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
        points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()
        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)


def find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=False):
    """
    Args:
        batch_points: (B*N, 5), [batch_idx, x, y, z, intensity]
        batch_boxes: (B, roi_num, 7), [x, y, z, dx, dy, dz, heading]
        grid_size: G
        max_num_boxes: M
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    assert grid_size > 0

    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    points_per_parts = []
    for i in range(batch_boxes.shape[0]):
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        points = batch_points[bs_mask]
        #* 因为roi可能有重合的, 返回为[N, roi个数, max_num_boxes], 如果在roi内就记录roi的索引, 否则填充-1
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_multi_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0), max_num_boxes).squeeze(0)
        #* [N, max_num_boxes, 7], 点所在的包围框的信息
        box_for_each_point = boxes[box_idxs_of_pts.long()]
        #* 将点云转换到所在的roi的坐标系下, [N, max_num_boxes, 3]
        xyz_local = points.unsqueeze(1) - box_for_each_point[..., 0:3]
        xyz_local_original_shape = xyz_local.shape
        #* [N, max_num_boxes, 3]->[N*max_num_boxes, 1, 3]
        xyz_local = xyz_local.reshape(-1, 1, 3)
        # Flatten for rotating points
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -box_for_each_point.reshape(-1, 7)[:, 6]
        )
        # Change coordinate frame to corner instead of center of box
        xyz_local_corner = xyz_local.reshape(xyz_local_original_shape) + box_for_each_point[..., 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        #* 去掉一些grid_point不在索引里面的点
        xyz_local_grid = (xyz_local_corner / (box_for_each_point[..., 3:6] / grid_size))
        points_out_of_range = ((xyz_local_grid < 0) | (xyz_local_grid >= grid_size) | (xyz_local_grid.isnan())).any(-1).flatten()
        #* [N, max_num_boxes, 4], [点所在的roi的索引, 点所在roi的grid x索引, 点所在roi的grid y索引, 点所在roi的grid z索引]
        xyz_local_grid = torch.cat((box_idxs_of_pts.unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        #*[N, max_num_boxes, 4]->[N*max_num_boxes, 4]
        xyz_local_grid = xyz_local_grid.reshape(-1, xyz_local_grid.shape[-1])
        # Filter based on valid box_idxs
        #* 去除不在roi中的点和在grid size范围外的点
        valid_points_mask = (xyz_local_grid[:, 0] != -1) & (~points_out_of_range)
        #* [valid_point_num, 4]
        xyz_local_grid = xyz_local_grid[valid_points_mask]

        if return_centroid:
            xyz_local = xyz_local[valid_points_mask].squeeze(1)
            centroids, part_idxs, points_per_part = voxel_aggregation_utils.get_centroid_per_voxel(xyz_local, xyz_local_grid)
            points_per_part = torch.cat((points_per_part.unsqueeze(-1), centroids), dim=-1)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size), dtype=points_per_part.dtype, device=points.device)
            else:
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1])).to_dense()
        else:
            #* 对同一个roi的grid idx进行去重, part_idxs为[unique_idx_num, 4], points_per_part为[unique_idx_num]
            part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
            # Sometimes no points in boxes, usually in the first few iterations. Return empty tensor in that case
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size), dtype=points_per_part.dtype, device=points.device)
            else:
                #* [roi_num, grid_size, grid_size, grid_size], 记录每个roi的每个grid里面有多少个点
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()

        points_per_parts.append(points_per_part_dense)
    """
    Returns:
        [batch_size, roi_num, grid_size, grid_size, grid_size], 记录每个roi的每个grid里面有多少个点
    """
    return torch.stack(points_per_parts)
