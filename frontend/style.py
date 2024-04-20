css = """
h1, h2, p {
    text-align: center;
    display: block;
}

p {
    text-align: center;
    font-size: 16px;
}

.result-real {
    color: green;
}

.result-fake {
    color: red;
}
"""


def percent_color(num):
    """
    Helper function to color fake review percentage based on range.
    """
    if num < 25:
        return f"<h1 style='color: green;'>{num}%</h1>"
    elif num < 50:
        return f"<h1 style='color: yellow;'>{num}%</h1>"
    elif num < 75:
        return f"<h1 style='color: orange;'>{num}%</h1>"
    else:
        return f"<h1 style='color: red;'>{num}%</h1>"


def rating_color(num):
    """
    Helper function to color adjusted rating based on range. Rating 1-5 stars
    """
    if num >= 4.0:
        return f"<h1 style='color: green;'>{num}</h1>"
    elif num >= 3.0:
        return f"<h1 style='color: yellow;'>{num}</h1>"
    elif num >= 2.0:
        return f"<h1 style='color: orange;'>{num}</h1>"
    else:
        return f"<h1 style='color: red;'>{num}</h1>"