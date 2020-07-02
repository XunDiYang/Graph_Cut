import cv2
import numpy as np
import tqdm
import maxflow

is_click = 0
is_finish = 0
# 前一时刻鼠标坐标
pre_x, pre_y = None, None
# 物体和背景的颜色
OBJ_COLOR = (0, 0, 255)
BKG_COLOR = (0, 255, 0)
# 当前画图的模式
OBJ_MODE = 1
BKG_MODE = 0
MODE = OBJ_MODE


class GraphCut:
    def __init__(self, img_path, lambda_=0, sigma=50.):
        self.img = cv2.imread(img_path)
        # 用来交互的图片
        # 交互的时候会修改图片像素！
        self.paint_img = self.img.copy()
        self.h, self.w, self.channel = self.img.shape
        print(self.w, self.h)
        # 选择种子点
        self.background_seeds = []
        self.object_seeds = []

        # 常数
        self.lambda_ = lambda_
        self.sigma = sigma
        self.K = None

        # 像素之间边的权值
        self.boundary_penalties = {}
        # 像素与s t点之间的权值
        self.region_penalties_obj = {}
        self.region_penalties_bkg = {}

        # 边和点的个数
        self.num_nodes = self.w * self.h
        self.num_edges = 0

        self.tot = 0

    def _all_pixels(self):
        # 遍历所有点的生成器
        # 生成 (i, j)类型的坐标
        for i in range(self.h):
            for j in range(self.w):
                yield (i, j)

    def _check_point(self, point):
        # 检测点的合法性
        x, y = point
        return 0 <= x < self.h and 0 <= y < self.w

    def scribe(self, mode, point):
        # 没使用copy，to_add 相当于原list的别名
        # point: (x, y)
        x, y = point
        if self._check_point(point):
            to_add = self.object_seeds if mode == OBJ_MODE else self.background_seeds
            # opencv提供的坐标与numpy的下标有区别
            to_add.append((y - 1, x - 1))

    def cut(self):
        if len(self.background_seeds) == 0 or len(self.object_seeds) == 0:
            print('未设置前景或背景')
            return

        # print('BKG\n', len(self.background_seeds), '\n', self.background_seeds)
        # print('OBJ\n', len(self.object_seeds), '\n', self.object_seeds)

        self.get_boundary_penalties()
        print('tot', self.tot)
        self.get_region_penalties()
        self.build_cut_graph()

    # 计算边界能量
    ###########################################################################
    def _get_neighbors(self, p):
        # 得到当前点的周围点
        x, y = p
        if x == self.h - 1 or y == self.w - 1:
            return []

        # 因为是无向图，每个节点只用考虑两个方向
        dirs = [(0, 1), (1, 0)]
        neighbors = []
        for dir in dirs:
            q_x, q_y = x + dir[0], y + dir[1]
            if self._check_point((q_x, q_y)):
                # 检查点是否合法
                neighbors.append((q_x, q_y))
        return neighbors

    def _get_boundary_penalty(self, p, q):
        # 计算两个像素点之间的权重
        # 注意：这里self.img[p]是uint8，如果结果小于0会溢出！所以需要转化
        delta = self.img[p].astype(np.int32) - self.img[q].astype(np.int32)
        dist = abs(p[0] - q[0]) + abs(p[1] - q[1])
        # 这里注意也有可能溢出
        tmp = - np.sum(delta ** 2) / (2 * self.sigma ** 2)
        penalty = np.exp(tmp) / dist
        # 这个是原论文的公式
        # return penalty
        # 这个公式参考https://github.com/NathanZabriskie/GraphCut
        penalty = 1 / (1 + np.sum(np.power(delta, 2)))

        self.tot += penalty
        return penalty

    def get_boundary_penalties(self):
        print('正在生成像素之间边权...')
        # 用于计算K值
        maximum = -1
        # 计算所有的B{p, q}
        for (i, j) in tqdm.tqdm(self._all_pixels()):
            p = (i, j)
            self.boundary_penalties[p] = {}
            tmp = 0
            for q in self._get_neighbors(p):
                # 其中p, q均是(x, y)类型的tuple
                self.num_edges += 1
                penalty = self._get_boundary_penalty(p, q)
                self.boundary_penalties[p][q] = penalty
                tmp += penalty
            maximum = max(maximum, tmp)
        # 这个是原论文公式，但实际效果不咋地
        # self.K = 1 + maximum
        self.K = 1000000000

    # 计算区域能量
    ###########################################################################

    def _get_gaussian(self, seeds):
        """
        :param seeds: list
        :return: 高斯分布的mu，sigma
        """
        # 通过设定的seed计算出 OBJ 和 bkg的高斯分布
        # TODO:方便起见，只使用一个通道进行分布计算
        values = [self.img[p][0] for p in seeds]
        # 注意：如果只有一个点，std为0的话后续计算会出现除零错误！
        return np.mean(values), max(np.std(values), 0.00001)

    def _get_gaussian_pro(self, x, mu, sigma):
        # 计算高斯分布下的概率
        factor = (1. / (abs(sigma) * np.sqrt(2 * np.pi)))
        return factor * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    def _get_region_penalty(self, point, mu, sigma):
        # 计算单个像素点的区域能量
        prob = max(self._get_gaussian_pro(self.img[point][0], mu, sigma), 0.00001)
        return - self.lambda_ * np.log(prob)

    def get_region_penalties(self):
        print('正在生产端点边权...')
        # 计算前景和背景的分布
        obj_mu, obj_sigma = self._get_gaussian(self.object_seeds)
        bkg_mu, bkg_sigma = self._get_gaussian(self.background_seeds)

        for p in tqdm.tqdm(self._all_pixels()):
            i, j = p
            if p in self.object_seeds:
                # 该点是前景
                self.region_penalties_bkg[p] = self.K
                self.region_penalties_obj[p] = 0
            elif p in self.background_seeds:
                # 背景点
                self.region_penalties_bkg[p] = 0
                self.region_penalties_obj[p] = self.K
            else:
                # 其他不确定的点
                self.region_penalties_bkg[p] = self._get_region_penalty(p, bkg_mu, bkg_sigma)
                self.region_penalties_obj[p] = self._get_region_penalty(p, obj_mu, obj_sigma)

    # 根据计算的边权建立图并切割
    ###########################################################################

    def _get_idx(self, p):
        return p[0] * self.w + p[1]

    def build_cut_graph(self):
        print('正在创建图...')
        print('nodes: %d edges: %d' % (self.num_nodes, self.num_edges))
        g = maxflow.Graph[float](self.num_nodes, self.num_edges)
        # nodes是一维的np.array
        nodes = g.add_nodes(self.num_nodes)
        for p in self._all_pixels():
            # 增加到s, k点的边
            p_idx = self._get_idx(p)
            g.add_tedge(p_idx, self.region_penalties_obj[p], self.region_penalties_bkg[p])
            # if self.region_penalties_obj[p] + self.region_penalties_bkg[p]:
            #     print(p_idx, self.region_penalties_obj[p], self.region_penalties_bkg[p])
            edges = self.boundary_penalties[p]
            for q, penalty in edges.items():
                q_idx = self._get_idx(q)
                g.add_edge(p_idx, q_idx, penalty, penalty)
                # print(p_idx, q_idx, penalty, penalty)

        print('正在分割...')
        flow = g.maxflow()

        # 白色背景板
        cut_img1 = np.zeros(self.img.shape, dtype=np.uint8)
        cut_img1.fill(255)

        cut_img2 = cut_img1.copy()
        for p in self._all_pixels():
            idx = self._get_idx(p)
            if g.get_segment(idx) == 1:
                cut_img1[p] = (0, 0, 255)
                cut_img2[p] = self.img[p]

        print('分割结束')
        cv2.imshow(window_name + ' result1', cut_img1)
        cv2.imshow(window_name + ' result2', cut_img2)
        cv2.imshow('origin', self.img)

        cv2.waitKey(0)


def draw(event, x, y, flags, param):
    global is_click, pre_x, pre_y
    # 根据模式选择颜色和list
    color = OBJ_COLOR if MODE == OBJ_MODE else BKG_COLOR

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键点击
        is_click = True
        cv2.rectangle(graphcut.paint_img, (x - 1, y - 1), (x + 1, y + 1), color, -1)
        # 这个坐标和numpy遍历是反的
        # TODO:为啥这个坐标是从 0到600...
        graphcut.scribe(MODE, (x, y))
        pre_x, pre_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # 左键松开
        is_click = False
    elif event == cv2.EVENT_MOUSEMOVE:
        # 鼠标移动
        if is_click:
            # 为了防止断点出现，使用line连接当前点和前一位置的点
            cv2.line(graphcut.paint_img, (pre_x, pre_y), (x, y), color, 2)
            graphcut.scribe(MODE, (x, y))
            pre_x, pre_y = x, y


if __name__ == '__main__':
    # graphcut = GraphCut("../image/yxd1.jpg")
    # graphcut = GraphCut("../image/yxd2.jpg")
    graphcut = GraphCut("../resource/hat.jpg")
    window_name = 'lambda = %.2f sigma = %.2f' % (graphcut.lambda_, graphcut.sigma)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw)

    while True:
        # 要不停显示img
        cv2.imshow(window_name, graphcut.paint_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        elif key == ord('g'):
            # 生成图片
            graphcut.cut()
        elif key == ord('t'):
            # 改变模式
            MODE = ~MODE
        elif key == ord('s'):
            # 保存图片
            pass

    cv2.destroyAllWindows()
