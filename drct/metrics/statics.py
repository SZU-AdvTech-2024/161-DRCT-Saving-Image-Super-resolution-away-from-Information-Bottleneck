import numpy as np

LIST_LIGHTS = np.array([[0.49630392, -0.46488675, 0.73318678],
                        [0.24442559, -0.13698725, 0.95994306],
                        [-0.03561741, -0.17536889, 0.98385835],
                        [-0.0934782,  -0.44171128, 0.89227402],
                        [-0.31681037, -0.50538874, 0.80262911],
                        [-0.10819103, -0.56019807, 0.82126296],
                        [0.28274414, -0.42263067, 0.86106861],
                        [0.10290347, -0.43157396, 0.89618909],
                        [0.20942622, -0.3361263,  0.91823733],
                        [0.09123674, -0.33245227, 0.93869662],
                        [0.13206232, -0.04645855, 0.99015212],
                        [-0.14061956, -0.36067477, 0.92203033]])


class Box:
    """
    box_ori: 中心原始图片在padding的坐标
    box_full: 正式裁取下来的分块在padding的坐标
    box_core: 会应用到结果中的核心部分在padding中的坐标
    box_real_core: 核心部分在分块内部的局部坐标
    box_real_full: 核心部分在重建后全图的坐标
    """
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0

    def __init__(self, x_min=0, x_max=0, y_min=0, y_max=0):
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.y_min = int(y_min)
        self.y_max = int(y_max)

    def __str__(self):
        return str([self.x_min, self.x_max, self.y_min, self.y_max])

    def load_from_list(self, list_in):
        """load from [x_min, x_max, y_min, y_max]"""
        self.x_min = int(list_in[0])
        self.x_max = int(list_in[1])
        self.y_min = int(list_in[2])
        self.y_max = int(list_in[3])

    def get_list(self):
        """return [x_min, x_max, y_min, y_max]"""
        list_out = [self.x_min, self.x_max, self.y_min, self.y_max]
        return list_out
