import chess
import chess.engine
import torch
import numpy as np
from functools import lru_cache
from chessModel import piece_to_index, ChessValueNetwork

class HashableBoard(chess.Board):
    def __hash__(self):
        return hash(self.fen())

    def __eq__(self, other):
        if isinstance(other, HashableBoard):
            return self.fen() == other.fen()
        return False
    
chess.Board = HashableBoard

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.4f}s")
        return result
    return wrapper
def count_calls(func):
    """
    装饰器：统计函数被调用的次数
    """
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1  # 增加计数器
        return func(*args, **kwargs)

    wrapper.call_count = 0  # 初始化计数器
    return wrapper

chess.Board

class NeuralNetPlayer:
    def __init__(self, model, max_depth=3, device="cpu"):
        """
        初始化自定义玩家。
        :param model: 已加载的神经网络模型
        :param max_depth: 搜索的最大深度
        :param device: PyTorch 设备 (CPU/GPU)
        """
        self.model = model.to(device)
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        self.max_depth = max_depth
        self.device = device

    def board2vec(self, board, flip=False):
        """
        将棋盘状态转换为神经网络输入。
        :param board: chess.Board 对象
        :param flip: 是否翻转棋盘（黑色视角）
        :return: torch.Tensor (12, 8, 8)
        """
        vec = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_index = piece_to_index[piece.symbol()]
                row, col = divmod(square, 8)
                if flip:
                    row = 7 - row
                    if piece_index < 6:  
                        piece_index += 6 
                    else: 
                        piece_index -= 6  
                vec[piece_index, row, col] = 1
        return torch.tensor(vec, dtype=torch.float32).to(self.device)
    
    @lru_cache
    def evaluate_position(self, board, flip):
        """
        评估当前棋盘状态的分数。
        :param board: chess.Board 对象
        :param flip: 是否翻转棋盘（黑色视角）
        :return: 分数值
        """
        vec = self.board2vec(board, flip)
        vec = vec.unsqueeze(0)  # 添加 batch 维度
        with torch.no_grad():
            score = self.model(vec).item()  # 假设模型输出一个标量分数
        return score
    
    def evaluate_position_tranditional(self, board, flip):
        """
        评估当前棋盘状态的分数。
        :param board: chess.Board 对象
        :param flip: 是否翻转棋盘（黑色视角）
        :return: 对于当前行动方的分数值
        """
        score = 0

        # 遍历棋盘上的所有格子
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # 获取棋子的分数
                piece_value = self.piece_values[piece.piece_type]
                # 根据棋子颜色调整分数
                if piece.color == chess.WHITE:
                    score += piece_value
                else:
                    score -= piece_value

        # 如果需要翻转棋盘视角（黑色视角）
        if flip:
            score = -score

        return score
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        CHECKMATE_SCORE = 1e10
        """
        Minimax 实现，用于评估并搜索最佳走法。
        :param board: 当前棋盘状态
        :param depth: 剩余搜索深度
        :param alpha: Alpha 值
        :param beta: Beta 值
        :param maximizing_player: 是否为最大化玩家 (True 表示白方, False 表示黑方)
        :return: (最佳分数, 最佳走法)
        """
        if board.is_game_over():
            if board.is_checkmate():
                return (-CHECKMATE_SCORE + depth if maximizing_player else CHECKMATE_SCORE - depth), None
            return 0, None  # 平局或其他结束情况

        if depth == 0:
            # 当前行动方为 maximizing_player，如果是黑方，则 flip = True
            flip = not maximizing_player
            return (1 if maximizing_player else -1) * self.evaluate_position(board, flip=flip), None

        best_move = None

        if maximizing_player:
            best_value = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                value, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # 剪枝
        else:
            best_value = float('inf')
            for move in board.legal_moves:
                board.push(move)
                value, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if value < best_value:
                    best_value = value
                    best_move = move

                beta = min(beta, best_value)
                if alpha >= beta:
                    break  # 剪枝

        return best_value, best_move
    
    def minimax_first_max(self, board, depth, alpha, beta, maximizing_player):
        CHECKMATE_SCORE = 1e10
        if board.is_game_over():
            if board.is_checkmate():
                return (-CHECKMATE_SCORE + depth if maximizing_player else CHECKMATE_SCORE - depth), None
            return 0, None
        if depth == 0:
            # 当前行动方为 maximizing_player，如果是黑方，则 flip = True
            flip = not maximizing_player
            return (1 if maximizing_player else -1) * self.evaluate_position(board, flip=flip), None
        best_move = None
        moves = sorted(board.legal_moves, key=lambda x: self.evaluate_move(board, x, maximizing_player), reverse=True)
        if maximizing_player:
            best_value = float('-inf')
            for move in moves:
                board.push(move)
                value, _ = self.minimax_first_max(board, depth - 1, alpha, beta, False)
                board.pop()

                if value > best_value:
                    best_value = value
                    best_move = move

                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # 剪枝
        else:
            best_value = float('inf')
            for move in moves:
                board.push(move)
                value, _ = self.minimax_first_max(board, depth - 1, alpha, beta, True)
                board.pop()

                if value < best_value:
                    best_value = value
                    best_move = move

                beta = min(beta, best_value)
                if alpha >= beta:
                    break  # 剪枝

        return best_value, best_move
    def evaluate_move(self, board, move, turn):
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return 1e10
        value = self.evaluate_position(board, not turn)
        board.pop()
        return value
    def negascout(self, board, depth, alpha, beta, maximizing_player):
        CHECKMATE_SCORE = 1e10
        if board.is_game_over():
            if board.is_checkmate():
                return (-CHECKMATE_SCORE + depth if maximizing_player else CHECKMATE_SCORE - depth), None
            return 0, None
        if depth == 0:
            # 当前行动方为 maximizing_player，如果是黑方，则 flip = True
            flip = not maximizing_player
            return (1 if maximizing_player else -1) * self.evaluate_position(board, flip=flip), None
        
        best_move = None
        a = alpha
        b = beta
        first_child = True
        # 对合法走法排序以优化搜索效率
        moves = sorted(board.legal_moves, key=lambda x: self.evaluate_move(board, x, maximizing_player), reverse=True)

        for move in moves:
            board.push(move)
            t = -self.negascout(board, depth - 1, -b, -a, not maximizing_player)[0]
            board.pop()
            if t > a and t < beta and not first_child and depth > 1:
                # Re-search
                board.push(move)
                t = -self.negascout(board, depth - 1, -beta, -t, not maximizing_player)[0]
                board.pop()
            if t > a:
                a = t   
                best_move = move
            if a >= beta:
                return a, best_move  # Cut-off
            b = a + 1  # Null window
            first_child = False
        return a, best_move

    def get_best_move(self, board):
        """
        获取最佳走法。
        :param board: 当前棋盘状态
        :return: 最佳走法 (chess.Move)
        """
        best_value, best_move = self.negascout(board, depth=self.max_depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=board.turn)
        print("eval:", best_value)

        return best_move
    
import chess.engine

import sys
import time
class UCIPlayer:
    def __init__(self, model, max_depth=3, device="cpu"):
        self.player = NeuralNetPlayer(model, max_depth=max_depth, device=device)
        self.name = "NeuralNetBot"
        self.author = "YourName"
        # 扩展 UCI 选项
        self.options = {
            "Hash": {"type": "spin", "default": 64, "min": 1, "max": 1024},
            "MultiPV": {"type": "spin", "default": 1, "min": 1, "max": 500},
            "Threads": {"type": "spin", "default": 1, "min": 1, "max": 512},
            "MaxDepth": {"type": "spin", "default": max_depth, "min": 1, "max": 100},
            "Move Overhead": {"type": "spin", "default": 30, "min": 0, "max": 5000},
            "Minimum Thinking Time": {"type": "spin", "default": 20, "min": 0, "max": 5000},
            "Ponder": {"type": "check", "default": False},
        }
        self.board = chess.Board()
        self.debug_mode = True
        # 存储选项的当前值
        self.current_values = {name: properties["default"] for name, properties in self.options.items()}
    def start(self):
        """
        启动 UCI 引擎并处理命令。
        """
        while True:
            try:
                command = input()
                self._process_command(command.strip())
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"info string Error: {str(e)}")
    def _process_command(self, command):
        """
        处理 UCI 命令。
        """
        if command == "quit":
            sys.exit(0)
            
        elif command == "uci":
            self._send_uci_info()
            
        elif command == "debug on":
            self.debug_mode = True
            
        elif command == "debug off":
            self.debug_mode = False
            
        elif command == "isready":
            print("readyok")
            
        elif command == "ucinewgame":
            self.board = chess.Board()
            
        elif command.startswith("setoption"):
            self._handle_setoption(command)
            
        elif command.startswith("position"):
            self._handle_position(command)
            
        elif command.startswith("go"):
            self._handle_go(command)
            
        elif command == "stop":
            # 实现停止思考的逻辑
            pass
            
        elif command == "ponderhit":
            # 实现潜思命中的逻辑
            pass
            
        elif command == "register":
            # 如果需要注册逻辑
            print("registration ok")
    def _send_uci_info(self):
        """
        发送引擎信息和选项。
        """
        print(f"id name {self.name}")
        print(f"id author {self.author}")
        
        # 发送选项信息
        for name, properties in self.options.items():
            option_str = f"option name {name} type {properties['type']}"
            if "default" in properties:
                option_str += f" default {properties['default']}"
            if "min" in properties:
                option_str += f" min {properties['min']}"
            if "max" in properties:
                option_str += f" max {properties['max']}"
            print(option_str)
            
        print("uciok")
    def _handle_setoption(self, command):
        """
        处理选项设置命令，增强的版本。
        """
        parts = command.split()
        if len(parts) >= 5 and parts[1] == "name":
            # 重建选项名称（处理包含空格的选项名）
            name_parts = []
            value_index = -1
            for i in range(2, len(parts)):
                if parts[i] == "value":
                    value_index = i
                    break
                name_parts.append(parts[i])
            
            if value_index == -1:
                return
                
            option_name = " ".join(name_parts)
            value = " ".join(parts[value_index + 1:])

            if option_name in self.options:
                try:
                    if self.options[option_name]["type"] == "spin":
                        value = int(value)
                        if self.options[option_name]["min"] <= value <= self.options[option_name]["max"]:
                            self.current_values[option_name] = value
                            if option_name == "MaxDepth":
                                self.player.max_depth = value
                    elif self.options[option_name]["type"] == "check":
                        self.current_values[option_name] = value.lower() == "true"
                    elif self.options[option_name]["type"] == "string":
                        self.current_values[option_name] = value
                    
                    if self.debug_mode:
                        print(f"info string Option {option_name} set to {value}")
                        
                except ValueError:
                    if self.debug_mode:
                        print(f"info string Invalid value for option {option_name}")

    def _get_time_allocation(self, wtime, btime, winc, binc, movestogo):
        """
        计算应该使用的思考时间。
        """
        # 获取当前方的时间和增量s
        if self.board.turn:
            time_left = wtime
            inc = winc
        else:
            time_left = btime
            inc = binc

        # 如果没有有效的时间信息，返回默认值
        if time_left < 0:
            return self.current_values["Minimum Thinking Time"]

        # 计算基本时间分配
        if movestogo > 0:
            time_per_move = time_left / movestogo
        else:
            # 假设还有大约20步要走
            time_per_move = time_left / 20

        # 考虑增量
        if inc > 0:
            time_per_move += inc * 0.75

        # 应用移动开销
        move_overhead = self.current_values["Move Overhead"]
        time_per_move = max(time_per_move - move_overhead, 
                          self.current_values["Minimum Thinking Time"])

        return min(time_per_move, time_left - move_overhead)
    def _handle_position(self, command):
            """
            处理位置设置命令。
            """
            parts = command.split()
            moves_index = -1

            if "startpos" in command:
                self.board = chess.Board()
                moves_index = parts.index("startpos") + 1
            elif "fen" in command:
                fen_index = parts.index("fen") + 1
                fen_parts = []
                for i in range(fen_index, len(parts)):
                    if parts[i] == "moves":
                        moves_index = i
                        break
                    fen_parts.append(parts[i])
                
                if moves_index == -1:
                    moves_index = len(parts)
                
                fen = " ".join(fen_parts)
                try:
                    self.board = chess.Board(fen)
                except ValueError:
                    if self.debug_mode:
                        print("info string Invalid FEN string")
                    return

            # 应用移动
            if moves_index < len(parts) and parts[moves_index] == "moves":
                for move in parts[moves_index + 1:]:
                    try:
                        self.board.push_uci(move)
                    except ValueError:
                        if self.debug_mode:
                            print(f"info string Invalid move: {move}")
                        break
    def _handle_go(self, command):
        """
        处理走棋命令，增强的版本。
        """
        parts = command.split()
        
        # 解析时间控制参数
        wtime = btime = winc = binc = movestogo = depth = movetime = -1
        infinite = False
        
        i = 1
        while i < len(parts):
            if parts[i] == "infinite":
                infinite = True
                i += 1
            elif parts[i] == "wtime" and i + 1 < len(parts):
                wtime = int(parts[i + 1])
                i += 2
            elif parts[i] == "btime" and i + 1 < len(parts):
                btime = int(parts[i + 1])
                i += 2
            elif parts[i] == "winc" and i + 1 < len(parts):
                winc = int(parts[i + 1])
                i += 2
            elif parts[i] == "binc" and i + 1 < len(parts):
                binc = int(parts[i + 1])
                i += 2
            elif parts[i] == "movestogo" and i + 1 < len(parts):
                movestogo = int(parts[i + 1])
                i += 2
            elif parts[i] == "depth" and i + 1 < len(parts):
                depth = int(parts[i + 1])
                i += 2
            elif parts[i] == "movetime" and i + 1 < len(parts):
                movetime = int(parts[i + 1])
                i += 2
            else:
                i += 1

        # 计算思考时间
        if not infinite and movetime == -1:
            think_time = self._get_time_allocation(wtime, btime, winc, binc, movestogo)
        else:
            think_time = movetime if movetime > 0 else None

        # 设置搜索深度
        if depth > 0:
            self.player.max_depth = min(depth, self.options["MaxDepth"]["max"])

        # 开始搜索最佳走法
        start_time = time.time()
        
        try:
            best_move = self.player.get_best_move(self.board)
            
            # 计算搜索用时
            search_time = int((time.time() - start_time) * 1000)  # 转换为毫秒
            
            # 发送搜索信息
            print(f"info depth {self.player.max_depth} time {search_time}")
            
            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove 0000")  # 没有合法移动时的特殊情况
                
        except Exception as e:
            if self.debug_mode:
                print(f"info string Error during search: {str(e)}")
            print("bestmove 0000")

model = ChessValueNetwork()
model.load_state_dict(torch.load("1.79.pth"))
model.eval()

if __name__ == "__main__":
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        uci_player = UCIPlayer(model, max_depth=3, device=device)
        uci_player.start()



# def test_play(player):
#     # 初始化棋盘
#     board = chess.Board()
#     while not board.is_game_over():
#         if board.turn:  # 玩家回合
#             print(board)
#             move = input("Your move (e.g., e2e4): ")
#             move = chess.Move.from_uci(move)
#             if move in board.legal_moves:
#                 board.push()
#             else:
#                 raise ValueError("Invalid move")
#         else:  # AI 回合
#             print("AI is thinking...")
#             ai_move = player.get_best_move(board)
#             board.push(ai_move)
#             print(f"AI played: {ai_move}")
