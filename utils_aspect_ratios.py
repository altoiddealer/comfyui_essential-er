from math import sqrt, gcd

def round_to_precision(val, prec):
    return round(val / prec) * prec

def res_to_model_fit(avg, w, h, prec):
    mp = w * h
    mp_target = avg * avg
    scale = sqrt(mp_target / mp)
    w = int(round_to_precision(w * scale, prec))
    h = int(round_to_precision(h * scale, prec))
    return w, h

def dims_from_ar(avg, n, d, prec=64):
    prec = max(1, int(prec))
    doubleavg = avg * 2
    ar_sum = n+d
    # calculate width and height by factoring average with aspect ratio
    w = round((n / ar_sum) * doubleavg)
    h = round((d / ar_sum) * doubleavg)
    # Round to correct megapixel precision
    w, h = res_to_model_fit(avg, w, h, prec)
    return w, h

def avg_from_dims(w, h):
    avg = (w + h) // 2
    if (w + h) % 2 != 0:
        avg += 1
    return avg

def ar_parts_from_dims(w, h):
    divisor = gcd(w, h)
    simp_w = w // divisor
    simp_h = h // divisor
    return simp_w, simp_h

def ar_parts_from_str(ar_str:str):
    ratio_parts = tuple(map(int, ar_str.replace(':', '/').split('/')))
    return ratio_parts[0], ratio_parts[1]
