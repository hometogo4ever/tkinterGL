import torch

dev = torch.device("cuda")

def find_min_max(vertices):
    # 각 vertex 차원에서 최대값과 최소값 계산
    max_values = torch.max(vertices, dim=0).values
    min_values = torch.min(vertices, dim=0).values
    return max_values, min_values