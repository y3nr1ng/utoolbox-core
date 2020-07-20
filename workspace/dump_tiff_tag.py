from tifffile import TiffFile

filename = "flybrain_Iter_ch0_stack0000_640nm_0000000msec_0000531429msecAbs.tif"

with TiffFile(filename) as tiff:
    print(f"{len(tiff.pages)} pages")

    image = tiff.pages[0]
    print(f"{image.shape}, {image.dtype}")

    tags = image.tags
    print(f"{len(tags)} tags")
    for tag in tags:
        print(f"{tag.name} ({tag.code}) {tag.dtype} ({tag.count})")

        value = tag.value
        if tag.code == 32781:
            value = value.replace(b"\x80", b"\n")
            value = value.decode("ascii", "backslashreplace")
        print(value)

        print()
