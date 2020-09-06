def to_latex(df, file_name: str, *args, **kwargs):
    kwargs["column_format"] = "|".join(
        [""] + ["l"] * df.index.nlevels + ["r"] * df.shape[1] + [""]
    )
    res = df.to_latex(*args, **kwargs)
    res = res.replace("toprule", "hline")
    res = res.replace("midrule", "hline")
    res = res.replace("bottomrule", "hline")
    with open(file_name, "w") as f:
        f.write(res)
    return res
