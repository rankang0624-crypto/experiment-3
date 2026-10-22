 # 一、实验目的
通过 SIFT 算法提取并匹配两幅图像的局部特征点，结合 Lowe 比率测试筛选候选匹配，并利用 RANSAC 估计单应矩阵，实现图像配准与拼接。进一步比较无融合、线性融合、固定宽度平滑融合以及 OpenCV Stitcher API 的输出效果，理解不同拼接实现的特点与局限。
二、实验内容
1. 图像拼接的一般步骤
1.读取两张具有共同视野和重叠区域的图像，并进行灰度化处理。
2.使用 SIFT 提取关键点及其描述符。
3.使用 FLANN 进行 KNN 特征匹配，通过 Lowe 比率测试筛选优质匹配点。
4.使用 RANSAC 去除错误匹配并估计单应矩阵，将两张图像映射到统一坐标系。
5.对配准后的图像分别进行无融合、线性融合和固定宽度平滑融合，并与 OpenCV Stitcher 自动拼接结果进行比较。
2. 输入图像
本次输入为用户提供的两张餐桌照片。两张图像均包含相同的桌面、餐盘、面包及主菜等局部结构，具有较明显的重叠区域，适合用于图像特征匹配和拼接。
<img width="696" height="522" alt="image" src="https://github.com/user-attachments/assets/20a6b669-5001-4da0-95e3-d6cd0b631e29" />

<img width="504" height="640" alt="image" src="https://github.com/user-attachments/assets/4428a461-0d2d-4164-97ea-551391690c59" />

    import cv2
    import numpy as np
 
    a = cv2.imread("a.jpg")
    b = cv2.imread("b.jpg")
 
    if a is None or b is None:
    raise FileNotFoundError("请将 a.jpg、b.jpg 放在脚本同目录")
三、实验原理与实现
1. SIFT 特征提取
SIFT（尺度不变特征变换）能够在不同尺度和一定旋转、光照变化下提取具有区分性的局部特征。程序调用 cv2.SIFT_create()，分别计算两张图像的关键点和描述符。关键点数量可用于描述检测到的局部特征规模，但数量多并不必然意味着拼接质量更好。
本次检测结果：图像 A 提取 4,959 个关键点，图像 B 提取 3,558 个关键点。

       sift = cv2.SIFT_create()
 
       gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
       gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
 
       ka, da = sift.detectAndCompute(gray_a, None)
       kb, db = sift.detectAndCompute(gray_b, None)
 
       print("图像 A 关键点数量：", len(ka))
       print("图像 B 关键点数量：", len(kb))
 
       cv2.imwrite("03_keypoints_a.jpg",
            cv2.drawKeypoints(a, ka, None))
       cv2.imwrite("04_keypoints_b.jpg",
            cv2.drawKeypoints(b, kb, None))
3. FLANN 特征匹配与 Lowe 比率测试
使用 FLANN 近似最近邻匹配器，对图像 B 的描述符在图像 A 的描述符集合中进行 KNN 查询（k=2）。对最近邻距离 d1 和次近邻距离 d2，保留满足 d1 < 0.7 × d2 的匹配，以降低局部描述子相似但实际位置不对应的错误匹配。
本次共进行 3,558 组 KNN 查询，筛选得到 1,113 对优质匹配点。
<img width="744" height="351" alt="image" src="https://github.com/user-attachments/assets/574be593-abc5-4fdd-aabc-0df2a4dd8f5a" />

图 3  FLANN 与 Lowe 比率测试后的匹配结果

# FLANN 的 KD-Tree 参数适用于 SIFT 浮点型描述符
    flann = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5),
    dict(checks=100)
    )
 
    pairs = flann.knnMatch(db, da, k=2)
 
    # 只保留最近邻距离明显小于次近邻距离的匹配
    good = []
    for pair in pairs:
    if len(pair) == 2:
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good.append(m)
 
    print("优质匹配点数量：", len(good))
 
    match_img = cv2.drawMatches(
    b, kb, a, ka, good, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
      )
    cv2.imwrite("01_sift_flann_matches.jpg", match_img)


3. RANSAC 与单应矩阵估计
由于初始匹配仍可能包含误匹配，实验使用 RANSAC 对候选匹配进行几何一致性检验。程序以图像 B 的关键点作为源点、图像 A 的关键点作为目标点，调用 cv2.findHomography(..., cv2.RANSAC, 4.0) 估计单应矩阵，并将投影误差小于设定阈值的匹配作为内点。
本次得到 1,086 个 RANSAC 内点，内点比例为 97.57%（1,086 / 1,113）。这表明筛选后的匹配点具有较强的几何一致性。
<img width="744" height="351" alt="image" src="https://github.com/user-attachments/assets/c8f7e80b-3172-4203-be82-a19517e0926a" />

图 4  RANSAC 几何内点匹配结果

    if len(good) < 4:
    raise RuntimeError("优质匹配点不足，无法估计单应矩阵")
 
    # 从图像 B 的匹配点映射到图像 A
    src_pts = np.float32(
    [kb[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
    [ka[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)
 
    H, mask = cv2.findHomography(
    src_pts, dst_pts, cv2.RANSAC, 4.0
    )
    if H is None:
     raise RuntimeError("RANSAC 单应矩阵估计失败")
 
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good)
    print("RANSAC 内点数量：", inliers)
    print("RANSAC 内点比例：{:.2%}".format(inlier_ratio))
 
    inlier_matches = [
    m for m, ok in zip(good, mask.ravel()) if ok
    ]
    inlier_img = cv2.drawMatches(
    b, kb, a, ka, inlier_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("02_ransac_inliers.jpg", inlier_img)
4. 图像配准与融合策略
根据估计的单应矩阵，将图像 B 投影到图像 A 的坐标系，并根据变换后两张图像的边界计算拼接画布。随后比较以下融合策略：
noBlending：不对重叠区域进行权重混合，直接覆盖像素，方法简单，但边界可能明显。
linearBlending：对重叠区域使用固定的 50% / 50% 权重进行线性融合，可减弱接缝，但可能出现重影或模糊。
linearBlendingWithConstant：依据到图像边界的距离计算融合权重，使边界附近的权重变化更平滑。该实现是对报告所述固定宽度融合思想的近似复现，并非原仓库中省略函数的逐行复刻。
<img width="744" height="559" alt="image" src="https://github.com/user-attachments/assets/482d3865-59fb-4918-902d-4944fa777445" />

        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
 
        corners_b = np.float32([
       [0, 0], [0, hb], [wb, hb], [wb, 0]
       ]).reshape(-1, 1, 2)
 
       warped_corners = cv2.perspectiveTransform(corners_b, H)
       corners_a = np.float32([
       [0, 0], [0, ha], [wa, ha], [wa, 0]
       ]).reshape(-1, 1, 2)
 
       all_corners = np.concatenate([warped_corners, corners_a])
       xmin, ymin = np.floor(
       all_corners.min(axis=0).ravel()
       ).astype(int)
       xmax, ymax = np.ceil(
       all_corners.max(axis=0).ravel()
       ).astype(int)
 
       W, Hh = xmax - xmin, ymax - ymin
       T = np.array([
       [1, 0, -xmin],
       [0, 1, -ymin],
       [0, 0, 1]
       ], dtype=np.float64)
 
       warped_b = cv2.warpPerspective(b, T @ H, (W, Hh))
       warped_a = np.zeros_like(warped_b)
       mask_a = np.zeros((Hh, W), dtype=np.uint8)
 
       x0, y0 = -xmin, -ymin
       warped_a[y0:y0+ha, x0:x0+wa] = a
       mask_a[y0:y0+ha, x0:x0+wa] = 255
 
       mask_b = cv2.warpPerspective(
       np.ones((hb, wb), dtype=np.uint8) * 255,
       T @ H, (W, Hh)
图 5  noBlending 无融合结果

<img width="744" height="559" alt="image" src="https://github.com/user-attachments/assets/64052062-467c-41ca-ad14-0bdb10165d1d" />

# 直接将图像 A 覆盖到变换后的图像 B 上
    out = warped_b.copy()
    out[mask_a > 0] = warped_a[mask_a > 0]
    cv2.imwrite("05_noBlending.jpg", out)
图 6  linearBlending 线性融合结果

<img width="744" height="559" alt="image" src="https://github.com/user-attachments/assets/7eec6699-7512-4370-8524-f454b9d8c2aa" />

    out = warped_b.copy()
    only_a = (mask_a > 0) & (mask_b == 0)
    overlap = (mask_a > 0) & (mask_b > 0)
 
    out[only_a] = warped_a[only_a]
# 重叠区域两幅图像各占 50% 权重
    out[overlap] = cv2.addWeighted(
    warped_a[overlap].reshape(-1, 1, 3), 0.5,
    warped_b[overlap].reshape(-1, 1, 3), 0.5,
    0
    ).reshape(-1, 3)
 
    cv2.imwrite("06_linearBlending.jpg", out)
    
图 7  linearBlendingWithConstant 平滑融合结果
5. OpenCV Stitcher API
为了进行实现方式对比，实验调用 cv2.Stitcher_create(cv2.Stitcher_PANORAMA).stitch([a, b]) 运行 OpenCV 自动拼接流程。本次返回状态码 0（Stitcher_OK），表示 API 成功完成拼接。

<img width="744" height="559" alt="image" src="https://github.com/user-attachments/assets/58d8760f-aba5-4520-9697-24f7b8fb898c" />

     dist_a = cv2.distanceTransform(
    (mask_a > 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    dist_b = cv2.distanceTransform(
    (mask_b > 0).astype(np.uint8), cv2.DIST_L2, 3
    )
 
    alpha = (dist_a / (dist_a + dist_b + 1e-6))[..., None]
    out = (
    warped_a.astype(np.float32) * alpha
    + warped_b.astype(np.float32) * (1 - alpha)
    ).clip(0, 255).astype(np.uint8)
 
    only_a = (mask_a > 0) & (mask_b == 0)
    only_b = (mask_b > 0) & (mask_a == 0)
    out[only_a] = warped_a[only_a]
    out[only_b] = warped_b[only_b]
 
    cv2.imwrite("07_linearBlendingWithConstant.jpg", out)
图 8  OpenCV Stitcher 自动拼接结果
四、实验结果与分析
实验指标	实际结果
图像 A 的 SIFT 关键点	4,959
图像 B 的 SIFT 关键点	3,558
KNN 查询组数	3,558
优质匹配点	1,113
RANSAC 内点	1,086
RANSAC 内点比例	97.57%
OpenCV Stitcher 状态码	0（成功）
从实验数据可知，FLANN 与 Lowe 比率测试得到较多候选匹配，RANSAC 又进一步验证了其中大部分匹配符合统一的几何变换关系。OpenCV Stitcher API 也成功完成自动拼接，说明两张餐桌照片具备足够的共同内容用于配准。
融合结果方面，无融合策略实现直接，但拼接边界可能较突出；线性融合能够对重叠区域的像素进行平均，降低亮度突变，但当图像存在轻微错位时可能出现模糊或重影；基于距离的平滑融合可使权重随空间位置变化，通常有助于减轻硬边界。OpenCV Stitcher 的内部流程还可能包含曝光补偿、缝隙选择等处理，因此其输出不一定与手动融合结果完全一致。
五、对比实验与讨论
本次对比了三种手动融合策略及 OpenCV Stitcher API。由于输入只有同一场景的两张静态照片，本实验没有复现 GitHub 原报告中的“纯静态裁剪”和“微动态物体移动”两套独立场景，也没有对原仓库 baseline 代码进行逐行运行。因此，本报告仅对当前两张输入图像的实际运行结果作出结论，不将其描述为完整复现了原报告的所有对比场景。
另外，GitHub 报告正文没有完整展示 linearBlending() 和 linearBlendingWithConstantWidth() 的函数实现，而是引用了 Notebook 中的代码。本次三种融合图像依据报告中描述的算法思想重新实现，具体像素结果与原 Notebook 可能存在差异。
六、实验总结与体会
本次实验完成了 SIFT 特征提取、FLANN 特征匹配、Lowe 比率测试、RANSAC 几何筛选、单应矩阵估计以及多种图像融合方式的运行。实验中得到 1,113 对优质匹配点，其中 1,086 对被 RANSAC 判定为内点，内点比例为 97.57%；OpenCV Stitcher 状态码为 0，自动拼接成功。
通过对比可以看出，图像拼接的关键不仅在于检测到足够多的特征点，还在于匹配点是否满足同一几何变换关系，以及融合策略能否自然处理重叠区域。无融合方法实现直接，线性融合有助于减弱接缝但可能产生重影，平滑权重融合可以改善过渡，而 OpenCV Stitcher 将配准与多种拼接优化步骤集成起来，使用更方便。
本次实验的局限在于，融合函数为依据报告描述实现的近似版本，且未复现原报告中所有动态场景和 baseline 对比。后续可以进一步运行原始 Notebook，在相同输入和参数下对比手动实现、参考代码与 OpenCV API，并对拼接接缝、运行时间及图像失真程度进行定量评价。
七、参考资料
1. nihao230，experiment-3-of-Computer-Vision，exp-3_report.md，GitHub。
   https://github.com/nihao230/experiment-3-of-Computer-Vision/blob/main/exp-3_report.md
2. OpenCV 官方文档：Image Stitching。
   https://docs.opencv.org/
