def deskew(src, zint, inplace=True, pattern="layer_(\d+)", z_col_header="z [nm]"):
    match = re.search(pattern, src)
    if not match:
        raise ValueError("unknown filename '{}'".format(src))
    z = match.group(1)
    try:
        z = int(z)
    except ValueError:
        fname = os.path.basename(src)
        raise ValueError("unable to extract Z index from filename '{}'".format(fname))

    df = pd.read_csv(src, header=0)
    df[z_col_header] = (z-1) * zint

    if not inplace:
        os.rename(src, "{}.old".format(src))
    df.to_csv(src, float_format='%g', index=False)
