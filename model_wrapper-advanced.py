import socket
import numpy as np
from custom_dqn import MaskedActionsDQN

from extended_classes import ExtendedBoard

from src.Colour import Colour
from src.Move import Move

from custom_dqn import MaskedActionsDQN

from src.Colour import Colour
from src.Move import Move


class HexAIWrapper():
    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, model_path):
        self._board_size = 0
        self._board = []
        self._colour = ""
        self._turn_count = 1

        # Load the model with the custom feature extractor
        self.model = MaskedActionsDQN.load(model_path)

    def run(self):
        states = {
            1: HexAIWrapper._connect,
            2: HexAIWrapper._wait_start,
            3: HexAIWrapper._make_move,
            4: HexAIWrapper._wait_message,
            5: HexAIWrapper._close
        }
        res = states[1](self,)
        while res != 0:
            res = states[res](self)

    def _connect(self):
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((HexAIWrapper.HOST, HexAIWrapper.PORT))
        return 2

    def _wait_start(self):
        data = self._receive_data().split(";")
        if data[0] == "START":
            self._board_size = int(data[1])
            self._colour = data[2]
            self._extended_board = ExtendedBoard(board_size=self._board_size)
            # print("we create a board here", {tile for tile in self._extended_board._tiles})
            return 4 if self._colour == "B" else 3
        else:
            print("ERROR: No START message received.")
            return 0
        
    def _make_move(self):
        

        action = self._decide_action()
        if action == "SWAP":
            msg = "SWAP\n"
        else:
            if(self._colour == "B"):
                action = self._transform_move(action[0], action[1])
            msg = f"{action[0]},{action[1]}\n"
        
        self._send_data(msg)
        return 4

    def _wait_message(self):
        self._turn_count += 1
        data = self._receive_data().split(";")
        if data[0] == "END" or data[-1] == "END":
            return 5
        else:
            self._process_board_change(data[1], data[2])
            return 3 if data[-1] == self._colour else 4

    def _decide_action(self):

        if(self._turn_count == 1):
            if(self._colour == "R"):
                return 2, 1
        elif(self._turn_count == 2 and self._colour == "B"):
            return "SWAP"
        # AI
        observation = self._extended_board.get_3d_representation()  # Full 15x15 board
        observation = observation[np.newaxis, ...]  # Add batch dimension

        action, _ = self.model.predict(observation, deterministic=True)
        x, y = divmod(action, self._board_size)  # Assuming the action is a flattened index
        return x[0], y[0]




    def _transform_move(self, x, y):
        # Transform the row and column for a 180-degree rotation
        x = self._board_size - 1 - x
        y = self._board_size - 1 - y
        return x, y

    def _process_board_change(self, action, board_str):
        if action != "SWAP":
            x, y = map(int, action.split(","))
            self._extended_board.update_board(Move(self._colour, x, y))
        else:
            self._colour = "B" if(self._colour == "R") else "R"
        self._update_board_from_string(board_str)


    def _update_board_from_string(self, board_str):
        rows = board_str.split(",")
        if(self._colour == "R"):
            for i, row in enumerate(board_str.split(",")):
                for j, tile in enumerate(row):
                    if tile == 'R':
                        self._extended_board._tiles[i + 2][j + 2].set_colour(Colour.RED)
                    elif tile == 'B':
                        self._extended_board._tiles[i + 2][j + 2].set_colour(Colour.BLUE)
        else:
            for i, row in enumerate(reversed(rows)):
                transformed_row = row[::-1]
                for j, tile in enumerate(transformed_row):
                    if tile == 'B':
                        self._extended_board._tiles[i + 2][j + 2].set_colour(Colour.RED)
                    elif tile == 'R':
                        self._extended_board._tiles[i + 2][j + 2].set_colour(Colour.BLUE)
        
        self._extended_board.update_edge_connections()

    def _receive_data(self):
        return self._s.recv(1024).decode("utf-8").strip()

    def _send_data(self, data):
        self._s.sendall(bytes(data, "utf-8"))

    def _close(self):
        self._s.close()
        return 0

if __name__ == "__main__":
    agent = HexAIWrapper(model_path="model_460000.zip")
    agent.run()