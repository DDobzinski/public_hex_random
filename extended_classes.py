# HexEnvironment/extended_class.py
import numpy as np

from src.Colour import Colour
from src.Tile import Tile
from src.Board import Board


class ExtendedTile(Tile):
    def __init__(self, x, y, colour=None):
        super().__init__(x, y, colour)
        # Flags for edge connections: [Top, Bottom, Left, Right] for each color
        # For example, if the tile is connected to the top board edge for the RED color,
        # then self.edge_connections[Colour.RED][0]=True
        # TOP and BOTTOM edges are for RED color
        # LEFT and RIGHT edges are for BLUE color

        self.edge_connections = {
            Colour.RED: [False, False, False, False],
            Colour.BLUE: [False, False, False, False],
        }

    def set_edge_connection(self, color, edge, value):
        self.edge_connections[color][edge] = value

    def get_edge_connection(self, color, edge):
        return self.edge_connections[color][edge]


class ExtendedBoard(Board):
    def __init__(self, board_size=11, padding=2):
        self.padding = padding
        padded_board_size = board_size + 2 * self.padding
        super().__init__(padded_board_size)
        self._tiles = [
            [ExtendedTile(i, j) for j in range(padded_board_size)]
            for i in range(padded_board_size)
        ]

        # Set padding colors
        for i in range(padded_board_size):
            for j in range(padded_board_size):
                # Top and bottom padding rows
                if i < self.padding or i >= board_size + self.padding:
                    self._tiles[i][j].set_colour(Colour.RED)
                # Left and right padding columns
                elif j < self.padding or j >= board_size + self.padding:
                    self._tiles[i][j].set_colour(Colour.BLUE)

        self.update_edge_connections()

    def update_edge_connections(self):
        # Clear previous connections and visits
        for row in self._tiles:
            for tile in row:
                tile.clear_visit()
                for color in [Colour.RED, Colour.BLUE]:
                    for edge in range(4):
                        tile.set_edge_connection(color, edge, False)

        # Update edge connections for tiles at the edges
        # TOP edge
        tile = self._tiles[0][0]
        self._update_connections_flood_fill(tile, Colour.RED, 0)
        self._unvisit_all()
        # RIGHT edge
        tile = self._tiles[self.padding][self._board_size - 1]
        self._update_connections_flood_fill(tile, Colour.BLUE, 1)
        self._unvisit_all()
        # BOTTOM edge
        tile = self._tiles[self._board_size - 1][0]
        self._update_connections_flood_fill(tile, Colour.RED, 2)
        self._unvisit_all()
        # LEFT edge
        tile = self._tiles[self.padding][0]
        self._update_connections_flood_fill(tile, Colour.BLUE, 3)
        self._unvisit_all()

    def _unvisit_all(self):
        for row in self._tiles:
            for tile in row:
                tile.clear_visit()

    def _update_connections_flood_fill(self, tile, color, edge):
        if tile.is_visited() or tile.get_colour() != color:
            return

        tile.visit()
        tile.set_edge_connection(color, edge, True)

        # Propagate edge connections to neighboring tiles
        for idx in range(Tile.NEIGHBOUR_COUNT):
            x_n = tile.get_x() + tile.I_DISPLACEMENTS[idx]
            y_n = tile.get_y() + tile.J_DISPLACEMENTS[idx]

            # Check for bounds
            if 0 <= x_n < self._board_size and 0 <= y_n < self._board_size:
                neighbor = self._tiles[x_n][y_n]
                self._update_connections_flood_fill(neighbor, color, edge)

    def get_3d_representation(self):
        # Convert the board state into a 3D representation
        board_3d = np.zeros(
            (self._board_size, self._board_size, 6), dtype=np.float32
        )  # 6 channels for each tile (2 colors, 4 edges)

        for i in range(self._board_size):
            for j in range(self._board_size):
                tile = self._tiles[i][j]

                # Set the color channels
                if tile.get_colour() == Colour.RED:
                    board_3d[i, j, 0] = 1  # Red color presence
                    
                elif tile.get_colour() == Colour.BLUE:
                    board_3d[i, j, 1] = 1  # Blue color presence
                    

                # Set the edge connection channels
                for edge in range(4):
                    if tile.get_edge_connection(Colour.RED, edge):
                        board_3d[i, j, 2 + edge] = 1
                    if tile.get_edge_connection(Colour.BLUE, edge):
                        board_3d[i, j, 2 + edge] = 1

        return board_3d

    def update_board(self, game_move):
        """
        Synchronize the ExtendedBoard state with the actual game board state.
        """

        i, j = game_move.get_x(), game_move.get_y()
        extended_tile = self._tiles[i + self.padding][j + self.padding]
        extended_tile.set_colour(game_move.colour)

        self.update_edge_connections()
