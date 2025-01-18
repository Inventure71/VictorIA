
# names of variable are a bit misleading, but it's just a because it was specialized
def calculate_intersection(m_left, b_left, m_top, b_top):
    denominator = 1 - m_top * m_left
    if denominator == 0:
        raise ValueError("Left and Top borders are parallel and do not intersect.")
    y = (m_top * b_left + b_top) / denominator
    x = m_left * y + b_left
    return x, y

"""
top_left = calculate_intersection_left_top(left_m, left_b, top_m, top_b)
top_right = calculate_intersection_right_top(right_m, right_b, top_m, top_b)
bottom_left = calculate_intersection_left_bottom(left_m, left_b, bottom_m, bottom_b)
bottom_right = calculate_intersection_right_bottom(right_m, right_b, bottom_m, bottom_b)
"""