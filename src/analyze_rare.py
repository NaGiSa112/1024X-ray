import pandas as pd

df = pd.read_csv("data/annotations.csv")
df["bw"] = (df.x2 - df.x1) / df.width
df["bh"] = (df.y2 - df.y1) / df.height
df["area"] = df.bw * df.bh
df["ar"] = df.bw / df.bh

header = f"{'class':<14}{'boxes':>6}{'imgs':>6}{'area_med':>10}{'p10':>8}{'p90':>8}{'ar_range':>14}{'shapes':>8}"
print(header)
for cls, g in sorted(df.groupby("class"), key=lambda kv: -len(kv[1])):
    ar = g.ar.round(1)
    print(
        f"{cls:<14}{len(g):>6}{g.image.nunique():>6}"
        f"{g.area.median():>10.4f}{g.area.quantile(.1):>8.4f}{g.area.quantile(.9):>8.4f}"
        f"{f'{ar.min():.1f}~{ar.max():.1f}':>14}{ar.nunique():>8}"
    )

print()
print("tool 每张图的框数分布 / 面积分布（判断是不是同一个物体反复出现）")
tool = df[df["class"] == "tool"]
per_img = tool.groupby("image").size()
print("  每图框数:", per_img.value_counts().sort_index().to_dict())
print("  面积(归一化)十分位:", [round(tool.area.quantile(q), 4) for q in (0, .25, .5, .75, 1)])
print("  长宽比十分位:", [round(tool.ar.quantile(q), 2) for q in (0, .25, .5, .75, 1)])

print()
print("图像级重复情况（同一张图被标注两遍的可能性）")
print("  唯一图片名:", df.image.nunique(), " 标注行数:", len(df))
