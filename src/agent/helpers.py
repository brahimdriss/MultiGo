from MultiGo.src.gotypes import Point

def is_point_an_eye(board, point, color):
    if board.get(point) is not None:        # Eye must be empty
        return False
    for neighbor in point.neighbors():      # All adjacent points must contain friendly stones
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color :
                return False

    friendly_corners = 0                    # control 3 out of 4 if in middle / all on the edge 
    off_board_corners = 0
    corners = [
        Point(point.row - 1 , point.col - 1 ),
        Point(point.row - 1 , point.col + 1),
        Point(point.row + 1 , point.col - 1),
        Point(point.row + 1 , point.col + 1),
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4 # point on the edge or corner
    return friendly_corners >= 3            # point in the middle