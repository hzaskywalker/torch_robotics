import torch
from .utils import batched_index_select


def update_dist(a, b):
    return (a[..., None] + b[:, None]).min(dim=2)


def value_iteration(cost, value=None, num_iter=15, inf=1e9):
    """
    :param cost: (b, n, n)
    :param value: (b, n) or None
    :param num_iter: number of value iteration
    :return:
        value (b, n), successor (b, n) if value is not None
        cost (b, n, n), successor (b, n, n) else
    """
    assert len(cost.shape) == 3
    b, n, _ = cost.shape
    eye = torch.eye(n, device=cost.device).float()
    cost = cost * (1-eye)[None,:].float()

    if value is not None:
        for i in range(num_iter):
            new_value = (cost + value[:, None, :]).min(dim=2)[0]
            value = torch.min(value, new_value)
    else:
        value = cost
        for i in range(num_iter):
            value = torch.min(value, (value[:,:,:,None] + value[:,None,:,:]).min(dim=2)[0])

    cost = cost + eye[None, :] * inf
    if len(cost.shape) == 3:
        cost = cost[..., None]
    successor = (cost + value[:, None, :] + 1e-10).min(dim=2)[1]

    return value, successor


def knn_distance(points, model, dst=None, k=10, inverse=False):
    """
    time complexity: batch_size x n x k  conv1d
    :param points: (batch_size, dim, n)
    :param model: f(start, goal) where start (batch, dim, n) goals (batch, dim, n)
    :param k: num of nearest neighbor
    :param dst:
    :param inverse:
    :return: distance (batch_size, n, n)
    """
    if dst is None:
        dst = points

    batch_size, dim, n = points.shape
    n2 = dst.shape[-1]

    dist = torch.norm(points[:, :, :, None] - dst[:, :, None, :], p=2, dim=1) # (b, n, n2)
    _, idx = torch.topk(-dist, k, dim=2)
    idx = idx.reshape(batch_size, -1)

    start = points[..., None].expand(-1, -1, -1, k).reshape(batch_size, dim, n*k)
    neighbor = batched_index_select(dst, 2, idx) #(batch_size, dim, n * k)

    if inverse:
        start, neighbor = neighbor, start
    dist = model(start, neighbor) #suppose the value is (batch_size, n, k)
    if len(dist.shape) == 3:
        assert dist.shape[1] == 1
        dist = dist[:, 0]
    dist = dist.reshape(batch_size, n, k)

    idx = idx.reshape(batch_size, n, k)
    ans = torch.zeros((batch_size, n, n2), device=points.device) + float('inf')
    return ans.scatter_(dim=2, index=idx, src=dist)


def test_knn_distance():
    points1 = torch.arange(10).float()[None, None, :]
    points2 = torch.arange(15).float()[None, None, :] - 5
    def model(a, b):
        return b-a

    print(knn_distance(points1, model, 3, points2, inverse=True))
    exit(0)


def find_path(points, successor, cur, target, max_len):
    """
    :param points: (b, dim, n)
    :param successor: (b, n, n) or (b, n)
    :param cur: (b, k) or (b,)
    :param target: (b, k) or (b,)
    :param max_len: integer
    :return: path （b, dim, k, n) or (b, dim, n) if k must be 1
    """
    flag = False
    if cur.dim() == 1:
        cur, target = cur[:, None], target[:, None]
        flag = True
    b, k = cur.shape
    n = points.shape[2]

    assert successor.shape[1] == successor.shape[2] == n
    successor = successor.view(b, n * n) # b x n*n

    length = torch.ones((b, k), device=cur.device).long() * max_len

    trajectory = []
    for i in range(max_len):
        trajectory.append(batched_index_select(points, 2, cur)) #(b, dim, k)
        mask = torch.eq(cur, target).long()
        length = torch.min(length, mask * (i+1) + (-mask + 1) * max_len) # (b, k)
        cur = batched_index_select(successor, 1, cur * n + target)

    trajectory = torch.stack(trajectory, dim=-1) #(b, dim, k, n)
    if flag:
        trajectory, length = trajectory[:,:,0], length[:, 0]
    return trajectory, length
