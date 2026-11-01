# Gudhi 的 Coxeter 三角化是什么,我们为什么没用它

回答问题 2。来源:(a) **直接读教授的代码**(`InfPlanner.cpp`,行号见下);
(b) 论文 Boissonnat / Kachanovich / Wintraecken,*Tracing Isomanifolds in Rᵈ in
Time Polynomial in d Using Coxeter–Freudenthal–Kuhn Triangulations*,
SoCG 2021 / SIAM J. Comput. 2023。
注:`gudhi.inria.fr` 在本会话的网络策略里不可达,所以 API 细节取自她的代码本身
(那才是"她实际用了什么"的权威来源),不是取自文档页。

---

## 一句话回答

**它不是"3D 空间里的三角化"。它是 `d` 维空间的三角化,`d` = 墙所在空间的维数。**
在她的代码里 `d = si_->getStateDimension()`,即**机器人的自由度**(可以是 2、3、5、7);
在我们的项目里 `d = 2`(页坐标 `(y, θ)`)。

说得更准确:它是**任意维度下的 marching cubes**。
marching squares 在 2D 网格上找等值线,marching cubes 在 3D 网格上找等值面;
Coxeter 三角化 + 流形追踪在 **d 维**空间上找等值**超曲面**,
而且不用网格、不占内存、可并行。

---

## 1. Coxeter 三角化本身

**是什么:** 把整个 `Rᵈ` 剖分成全等的 `d`-单形(d=2 是三角形,d=3 是四面体,
d≥4 是它们的高维推广)的一种**周期性**剖分。
Freudenthal–Kuhn 剖分把格子立方体切成 `d!` 个单形;再做一个线性变换就得到
Ã_d 型的 Coxeter 三角化,它在这类空间填充剖分里**长宽比(thickness)最好**
—— 这对数值稳健性要紧,因为瘦长单形会让"流形穿过哪条边"的判断失效。

**关键性质:它是隐式的,根本不存储。**
给定一个点,用 O(d²) 次算术就能算出"它落在哪个单形里"以及"那个单形的顶点是谁"。
没有网格、没有邻接表、没有内存。
这正是 `d=7` 还能活下去的唯一原因:`res 61` 的七维网格是 61⁷ ≈ 3×10¹²
个格子,而隐式三角化只在**流形附近**生成单形。

她代码里对应的两行(`InfPlanner.cpp:1843–1845`):

```cpp
Coxeter_triangulation<> cox_tr(oracle.amb_d());               // d = 状态维数
cox_tr.change_offset(Eigen::VectorXd::Random(oracle.amb_d())); // 随机平移
cox_tr.change_matrix(lambda_ * cox_tr.matrix());               // 缩放 = 分辨率
```

- `change_offset(Random)`:**随机错位**。如果流形刚好和某个单形的面共面,
  "穿过哪条边"就无法判定;随机平移以概率 1 避开这种退化。
- `change_matrix(lambda_ * ...)`:`lambda_` 就是**分辨率旋钮**。
  她在 `checkManifold` 里每跑完一轮没短路的检查就 `lambda_ *= 0.9`
  (`:1280`),即**自适应细化**。这对应我们 Algorithm 4 的
  `net_d_min *= 0.5`(细分面)那一支。

## 2. 流形追踪(manifold tracing)

墙是一个**隐式**曲面:`F(x) = 0`,这里 `F` 就是 SVM 判别式
(她的 `Manifold_surface` 仿函数,`:1688`;`cod_d() == 1` 说明余维 1,
`amb_d()` 是环境维数)。算法:

1. 从若干**种子点**出发(`lastSeeds_`,取自落在墙上的那些点)。
2. 找种子所在的单形。
3. **相交判定(oracle)**:余维 1 时流形横截地穿过三角化的**棱**(1-单形)。
   棱的两个端点上 `F` 异号 ⇒ 流形穿过这条棱 ⇒ 解出交点。
   `make_precise_oracle(fun)`(`:1838`)就是这个判定器。
4. **广度优先走开**:从一条被穿过的棱,经它所在的三角形走到相邻的棱
   (她的 `save2DRNtriangulation` 里 `s.coface_range(2)` → `cof.face_range(1)`
   就是"棱 → 它的 2-面(三角形)→ 那些三角形的 1-面(棱)",`:1770–1772`),
   把所有被穿过的棱收集起来。
5. 结果:一张 `被穿过的单形 → 交点` 的表(`Out_simplex_map`)。
   对每条被穿过的棱取**对偶胞腔**,就得到墙的分片线性逼近:
   - `d=2`:对偶是**线段** ⇒ 墙是**折线**
   - `d=3`:对偶是**三角形** ⇒ 墙是**三角网格曲面**
   - 一般地:墙是 `(d−1)` 维的 PL 超曲面,胞腔是 `(d−1)`-单形

她把第 4、5 步并行化了:`manifold_tracing_algorithm_par_libcuckoo`(`:1852`)
用 libcuckoo 并发哈希表,`construct_n_save_simplices_par`(`:1891`)
用 `boost::asio::thread_pool(hardware_concurrency())`。

**为什么 IJRR 需要这一步:** Definition 1 要求把学到的流形变成一个**显式的闭多胞形**,
然后 Algorithm 3 **逐面**验证每个面都在 `C_obs` 里。
"面"在 `d` 维就是 `(d−1)`-单形 —— 不做三角化就没有面可查。

**代价结构(论文的结论):** 复杂度对 `d` 和分辨率 `1/D` 是**多项式**,
但对流形的**内蕴维数 n** 是**指数**。余维 1 时 `n = d − 1`,
所以 d 变大时指数项就是主项 —— 这和她报告"三角化占 >90% 时间"一致。

---

## 3. 我们为什么没用它

| | `d = 2`(本项目) | `d = 3`(待办 8) | `d = 7`(她的高 DOF 实验) |
|---|---|---|---|
| 墙是 | 一条曲线 | 一张曲面 | 6 维超曲面 |
| 面是 | **线段** | 三角形 | 6-单形 |
| 网格可行吗 | 61² = 3 721 ✅ | 61³ = 226 981 ⚠️ | 61⁷ ≈ 3×10¹² ❌ |
| 三角化占时间 | ~0% | 开始要紧 | **>90%** |

**`d=2` 时 IJRR §4.3 的切向 Delaunay 复形退化成"把零集连成有序折线",
而这正是 `reconstruct_wall` 用 `plt.contour` 做的事**,顺带还验证了闭合性
(Definition 1 (I))。实测:`reconstruct_wall` 在 EXP 3 里 11 次调用共 1.99 s,
其中几乎全部是在 281×421 网格上求 118 301 次判别式 —— **不是三角化的代价**。

**而且她自己也有 2-D 专用分支**,不走高维那条路:

| 她的 2-D 专用函数 | 行 | 做什么 |
|---|---|---|
| `save2DRNtriangulation` | 1737 | `d==2` 时单独把线段存成 CSV |
| `save2Dmanifold` | 1796 | `d==2` 时用更细的三角化再画一遍 |
| `checkSimplex2D` | 2260 | `d==2` 时的面检查,**完全不算穿透深度** |

`generateCT` 里那两行就是显式的维度分派(`:1863–1865`):

```cpp
if (si_->getStateDimension() == 2) save2DRNtriangulation(...);
if (si_->getStateDimension() == 2) save2Dmanifold(ptc);
```

所以"用她的方法"和"我们现在做的"在 `d=2` 上**不是两种方法,是同一种方法的
同一个退化分支** —— 她写了 Gudhi 那条路是为了 `d≥3`,而不是为了 `d=2`。

### 那"既然教授已经用她的方法做出了不可行性证明,我们为什么不直接用她的代码?"

三个理由,按重要性排:

1. **她的代码解的不是我们的问题。** 她在 `Q`(关节空间,`d = DOF`)里做证明;
   我们在 `Y⁺`(显式流形的坐标空间,`d = 2`)里做。
   `Y⁺` 的页/门结构 —— 八页、奇异集 Σ、过门换页、泛洪 —— 她的代码里**完全没有**,
   因为 `Q` 里没有这回事。那是 [IROS] 的贡献,也是我们这篇的全部新内容。
   她的 `InfPlannerPRM` 换到 `Y⁺` 上要重写的恰好是我们已经写完的那部分。
2. **依赖代价。** 要跑起来需要 OMPL + Gudhi + ThunderSVM(CUDA)+ nlopt +
   amino + libcuckoo + Eigen + boost。换来的是三件我们在 Python 里也能做的事
   (见 `docs/seven_questions.md` Q2:并发 anytime 结构、Alg 2/3 线程池、
   `checkSimplex2D` 的便宜面检查)。
3. **我们在一处比发布的实现更贴近论文。** 她的 `trainingSetup()`(`:3597`)把
   `param_.gamma` **固定为 1.0**、`C=100`,而 Algorithm 1 的 γ 递增循环在她代码里
   是**注释掉的**(`:3579–3588`)。我们实现的是论文写的那个递增版本
   (`train_wall`,γ 升到训练精度 1)。汇报时这一点值得讲。

### 什么时候必须换成她那条路

**待办 8:`d = 3`**(IROS 的 2D 曲面任务)。那时候:
网格彻底不能用、三角化变成主开销、面从线段变成三角形、
碰撞检测也该换成真实 STL 网格(FCL)。
那是 Gudhi + OMPL + FCL + C++ 的主场,也是**唯一**值得付那份依赖代价的时候。

---

## 4. 一张对照表

| IJRR 的步骤 | 她的实现 | 我们的实现 | `d=2` 上等价吗 |
|---|---|---|---|
| Alg 1 学墙 | ThunderSVM,γ 固定 1.0 | sklearn,γ 递增到精度 1 | 我们更贴论文 |
| Alg 2 采墙 | nlopt(解析梯度) | 闭式梯度 Newton(§17) | 等价,我们快 99× |
| §4.3 重建 | Gudhi Coxeter + 流形追踪 | `plt.contour` + 闭合性校验 | **等价**(她也有 2-D 分支) |
| Alg 3 查面 | 线程池 + `checkSimplex2D` | 串行 + `pd ≥ r` | 我们更严格,她更快 |
| Alg 4 外循环 | `lambda_ *= 0.9` + `triangulationLocalFix` | `net_d_min *= 0.5` + 回灌重训 | 等价 |
| 训练标签 | `makeTrainingDataFromGraph`(按连通分量) | `RoadmapLabeller`(同) | 等价(§21/§25 已搬) |
| 并发 | 三线程 anytime + 三个线程池 | **没有** | **这是真缺的那一块** |
