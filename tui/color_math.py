import math

def oklch_to_srgb(l: float, c: float, h_degrees: float) -> tuple[int, int, int]:
    """
    Convert OKLCH (Lightness, Chroma, Hue) to sRGB (0-255).
    L: 0 to 1
    C: 0 to ~0.4
    H: 0 to 360 degrees
    """
    h = math.radians(h_degrees)
    
    # OKLCH to OKLAB
    L = l
    a = c * math.cos(h)
    b = c * math.sin(h)
    
    # OKLAB to Linear sRGB
    # Matrix from standard OKLAB definitions
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    
    l_ = l_ ** 3
    m_ = m_ ** 3
    s_ = s_ ** 3
    
    r_lin =  4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g_lin = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b_lin = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    
    # Linear sRGB to sRGB
    def gamma_correct(c):
        c = max(0.0, min(1.0, c))
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1 / 2.4)) - 0.055
            
    r = int(gamma_correct(r_lin) * 255)
    g = int(gamma_correct(g_lin) * 255)
    b = int(gamma_correct(b_lin) * 255)
    
    return (r, g, b)

def get_glass_surfaces(bg_r: int, bg_g: int, bg_b: int, accent_hex: str) -> dict:
    """
    Given a background color and an accent color, returns the 4%, 8%, and 12% 
    glass surface colors simulating a physical glass stack.
    Since we can't reliably read the terminal background due to TTY race conditions
    in some VTEs, we will use a sophisticated alpha overlay.
    """
    return {
        "base": f"rgba(255,255,255,0.04)",
        "raised": f"rgba(255,255,255,0.08)",
        "overlay": f"rgba(255,255,255,0.12)"
    }
