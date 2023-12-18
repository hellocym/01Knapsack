import random


def brutal(C:int, w:list, v:list)->None:
    """
    使用暴力法解决01背包问题
    C: 背包承重
    w_i: 第i个物品的重量
    v_i: 第i个物品的价值
    """
    if len(w) != len(v):
        raise ValueError("w和v长度必须相同")
    n = len(w)
    max_x = None
    max_value = 0
    for i in range(2**n):
        x = [0 for _ in range(n)]
        weight = 0
        value = 0
        for j in range(n):
            if (i >> j) & 1:
                weight += w[j]
                value += v[j]
                x[j] = 1
        if weight <= C:
            if value > max_value:
                max_value = value
                max_x = x.copy()
    return {
        'Max Value': max_value, 
        'Selection': max_x
    }


def DP(C:int, w:list, v:list)->None:
    """
    使用动态规划解决01背包问题
    C: 背包承重
    w_i: 第i个物品的重量
    v_i: 第i个物品的价值
    """
    if len(w) != len(v):
        raise ValueError("w和v长度必须相同")
    n = len(w)
    dp = [[0 for _ in range(C+1)] for _ in range(n+1)]
    for j in range(0, min(C, w[n-1])):
        dp[n][j] = 0
    for j in range(w[n-1], C+1):
        dp[n][j] = v[n-1]
    for i in range(n-1, 0, -1):
        for j in range(0, min(C, w[i-1])):
            dp[i][j] = dp[i+1][j]
        for j in range(w[i-1], C+1):
            dp[i][j] = max(dp[i+1][j], dp[i+1][j-w[i-1]]+v[i-1])
    
    # 构造最优解
    x = []
    remain = C
    for i in range(1, n):
        if dp[i][remain] == dp[i+1][remain]:
            x.append(0)
        else:
            x.append(1)
            remain -= w[i-1]
    if dp[n][remain]:
        x.append(1)
    else:
        x.append(0)
    return {
        'Max Value': dp[1][C], 
        'Selection': x
    }

class GA:
    def __init__(self, C:int, w:list, v:list):
        """
        使用遗传算法解决01背包问题
        C: 背包承重
        w_i: 第i个物品的重量
        v_i: 第i个物品的价值
        x: 选择的物品，基因型
        适应度：背包价值
        """
        if len(w) != len(v):
            raise ValueError("w和v长度必须相同")
        self.n = len(w)
        self.w = w
        self.v = v
        self.C = C
        # 种群大小
        self.pop_size = 256
        # 竞赛选择规模
        self.tournament_size = 2
        # 拥挤因子
        self.crowding_factor = 3
        # 交叉父辈数量
        self.num_parents = 3
        # 随机种子
        seed = 42
        random.seed(seed)
        # 初始化种群
        print('Initializing Population...')
        self.pop = []
        for i in range(self.pop_size):
            x = [0 for _ in range(self.n)]
            for j in range(self.n):
                if random.random() < 0.5:
                    x[j] = 1
            self.pop.append(x)
        # print(self.pop)

    # 适应度函数
    def fitness(self, x):
        weight = 0
        value = 0
        for i in range(self.n):
            if x[i]:
                weight += self.w[i]
                value += self.v[i]
        if weight > self.C:
            return 0
        else:
            return value
    
    # 竞赛选择
    def tournament_selection(self, pop, tournament_size):
        selected = []
        for i in range(len(pop)):
            selected.append(random.sample(pop, tournament_size))
        selected = [max(s, key=self.fitness) for s in selected]
        return selected

    # 交叉（多父辈交叉，位点平均）
    def multi_parent_crossover(self, parents, num_parents):
        # 随机选择父辈
        selected_parents = random.sample(parents, num_parents)

        # 确保所有父辈长度一致
        length = len(selected_parents[0])
        assert all(len(parent) == length for parent in selected_parents), "所有父辈长度必须一致"

        # 生成后代，位点平均
        child = [sum([parent[i] for parent in selected_parents]) / num_parents for i in range(length)]
        child = [round(c) for c in child]
        return child

    # 变异（交换变异）
    def swap_mutation(self, x):
        # 随机选择两个位点
        i, j = random.sample(range(self.n), 2)
        # 交换
        x[i], x[j] = x[j], x[i]
        return x

    def crowding(self, children, crowding_factor):
            """
            拥挤策略
            """
            for child in children:
                # 选择一组随机个体
                candidates = random.sample(self.pop, crowding_factor)
                # 选择最相似的个体（汉明距离）
                most_similar = min(candidates, key=lambda x: sum([abs(x[i]-child[i]) for i in range(self.n)]))
                # 如果child更优，则替换most_similar
                if self.fitness(child) > self.fitness(most_similar):
                    # print(f'child: {self.fitness(child)}, most_similar: {self.fitness(most_similar)}')
                    self.pop.remove(most_similar)
                    self.pop.append(child)

    def run(self):
        # 构建生成器
        while True:
            a = self.pop.copy()
            # print('Selecting...')
            children = self.tournament_selection(self.pop, self.tournament_size)
            # print('Crossovering...')
            children.extend([self.multi_parent_crossover(self.pop, self.num_parents) for _ in range(self.pop_size)])
            # print('Mutating...')
            children.extend([self.swap_mutation(child.copy()) for child in children])
            self.crowding(children, self.crowding_factor)
            print(f'Overall Best: {(best:=max(self.pop, key=self.fitness))}, {self.fitness(best)}')
            yield self.pop
    

if __name__ == "__main__":
    C = 10
    w = [7, 3, 4, 5]
    v = [42, 12, 40, 25]
    # print(brutal(C, w, v))
    # print(DP(C, w, v))
    # GA(C, w, v)
    # exit()
    C = 25
    w = [7,5,3,2,4,8,6,9]
    v = [42,31,12,7,40,41,25,40]
    # print(brutal(C, w, v))
    # print(DP(C, w, v))
    # ga = GA(C, w, v)
    # generation = ga.run()
    # for i in range(20):
    #     next(generation)
    C = 1000
    w = [80,82,85,70,72,70,82,75,78,45,49,76,45,35,94,49,76,79,84,74,76,63,35,26,52,12,56,78,16,52,16, 42,18,46,39,80,41,41,16,35,70,72,70,66,50,55,25, 50,55,40]
    v = [200,208,198,192,180,180,168,176,182,168,187,138,184,154,168,175,198,184,158,148,174,135,126,156,123,145,164,145,134,164,134,174,102,149,134,156,172,164,101,154,192,180,180,165,162, 160,158,155,130,125]
    # print(brutal(C, w, v))
    print(DP(C, w, v))
    ga = GA(C, w, v)
    generation = ga.run()
    for i in range(300):
        next(generation)