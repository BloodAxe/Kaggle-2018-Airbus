
def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return rgb


def hex2bgr(value):
    rgb = hex2rgb(value)
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr
