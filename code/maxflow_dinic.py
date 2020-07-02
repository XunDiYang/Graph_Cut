# Dinic Algorithm

INF = 10000000


# ref: https://www.cnblogs.com/LUO77/p/6115057.html
# 使用BFS构件层次图
def Bfs(C, F, s, t):
    """

    :param C: 容量矩阵
    :param F:
    :param s:
    :param t:
    :return:
    """
    n = len(C)
    queue = []
    queue.append(s)
    global level
    # 初始化
    level = n * [0]
    level[s] = 1
    while queue:
        k = queue.pop(0)
        for i in range(n):
            if (F[k][i] < C[k][i]) and (level[i] == 0):  # 未被访问
                level[i] = level[k] + 1
                queue.append(i)
    return level[t] > 0


# DFS寻找增广路径
def Dfs(C, F, k, cp):
    tmp = cp
    if k == len(C) - 1:
        return cp
    for i in range(len(C)):
        if (level[i] == level[k] + 1) and (F[k][i] < C[k][i]):
            f = Dfs(C, F, i, min(tmp, C[k][i] - F[k][i]))
            F[k][i] = F[k][i] + f
            F[i][k] = F[i][k] - f
            tmp = tmp - f
    return cp - tmp


# 接口
def MaxFlow(C, s, t):
    """

    :param C: 容量矩阵
    :param s: 起点下标
    :param t: 终点下标
    :return:
    """
    n = len(C)
    # F为流量矩阵
    F = [n * [0] for i in range(n)]
    flow = 0
    while Bfs(C, F, s, t):
        flow = flow + Dfs(C, F, s, INF)
    return flow


if __name__ == '__main__':
    # -------------------------------------
    # make a capacity graph
    # node   s   o   p   q   r   t
    C = [[0, 3, 3, 0, 0, 0],  # s
         [0, 0, 2, 3, 0, 0],  # o
         [0, 0, 0, 0, 2, 0],  # p
         [0, 0, 0, 0, 4, 2],  # q
         [0, 0, 0, 0, 0, 2],  # r
         [0, 0, 0, 0, 0, 3]]  # t

    source = 0  # A
    sink = 5  # F
    print("Dinic's Algorithm")
    max_flow_value = MaxFlow(C, source, sink)
    print("max_flow_value is", max_flow_value)
