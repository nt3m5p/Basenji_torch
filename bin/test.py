import torch


def pearson_correlation_coefficient(x, y):
    # 确保输入是一维张量
    x = x.squeeze()
    y = y.squeeze()
    
    # 计算均值
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    
    # 计算标准差
    sigma_x = torch.std(x)
    sigma_y = torch.std(y)
    
    # 计算Pearson相关系数
    with torch.no_grad():  # 确保不会计算梯度
        numerator = ((x - mu_x) * (y - mu_y)).sum()
        denominator = sigma_x * sigma_y
        corr_coef = numerator / denominator
    
    return corr_coef


# 示例使用
x = torch.randn(100)  # 随机生成示例数据
y = torch.randn(100)
corr_coef = pearson_correlation_coefficient(x, y)
print(corr_coef)  # 输出Pearson相关系数