import chess
import chess.engine
import random
import pickle
import os
import threading
from typing import Dict, List, Tuple


class ChessAI:
    def __init__(self, engine_path: str = "stockfish"):
        """
        Initialize the Chess AI with Q-learning and minimax capabilities.
        
        Args:
            engine_path: Path to UCI chess engine (default: "stockfish")
        """
        self.engine = None
        self.engine_path = engine_path
        self._initialize_engine()
        self.lock = threading.Lock()
        
        # Q-learning parameters
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.3  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.max_q_table_size = 40000 
        self.q_table_cleanup_threshold = 70000
        self.experience_buffer = []
        self.buffer_size = 1000
        
        # Game parameters
        self.minimax_depth = 3
        self.max_moves_per_game = 200
        
        # Game state
        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self.game_states: List[Tuple[str, chess.Move]] = []
        
        # Load existing Q-table if available
        self.load_q_table()
    
    def _initialize_engine(self):
        """Initialize the chess engine with error handling."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            print(f"Successfully initialized engine: {self.engine_path}")
        except Exception:
            self.engine = None
    
    def _cleanup_q_table(self):
        """Remove least used entries when Q-table gets too large."""
        if len(self.q_table) >= self.q_table_cleanup_threshold:
            # Remove states with lowest total Q-values (least explored)
            states_to_remove = []
            for state, moves in self.q_table.items():
                total_q_value = sum(abs(q) for q in moves.values())
                if total_q_value < 0.1:  # Very low exploration
                    states_to_remove.append(state)
            
            for state in states_to_remove[:len(self.q_table) - self.max_q_table_size]:
                del self.q_table[state]

    def get_board_state(self, board: chess.Board) -> str:
        """Get a string representation of the board state."""
        # Use FEN but remove move counters to reduce state space
        fen_parts = board.fen().split()
        return ' '.join(fen_parts[:4])  # Position, turn, castling, en passant
    
    def get_move(self, board: chess.Board, use_exploration: bool = True) -> chess.Move:
        """ Get the best move using Q-learning with epsilon-greedy exploration."""

        state = self.get_board_state(board)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Initialize Q-values for this state if not seen before
        if state not in self.q_table:
            self.q_table[state] = {move.uci(): 0.0 for move in legal_moves}
        
        # Add any new legal moves to existing state
        for move in legal_moves:
            if move.uci() not in self.q_table[state]:
                self.q_table[state][move.uci()] = 0.0
        
        # Clean up q_table
        if len(self.q_table) > self.q_table_cleanup_threshold:
            self._cleanup_q_table()

        # Epsilon-greedy action selection
        if use_exploration and random.random() < self.epsilon:
            return random.choice(legal_moves)
        else:
            # Choose move with highest Q-value
            move_values = {move: self.q_table[state].get(move.uci(), 0.0) 
                          for move in legal_moves}
            best_move = max(move_values, key=move_values.get)
            return best_move
    
    def evaluate_board(self, board: chess.Board) -> float:
        """Evaluate the board position."""
        # Game over evaluations
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Try engine evaluation first
        if self.engine:
            try:
                info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
                score = info["score"].relative.score()
                return score / 100.0 if score is not None else self._material_evaluation(board)
            except:
                pass
        
        # Fallback to material + positional evaluation
        return self._material_evaluation(board) + self._positional_evaluation(board)
    
    def _material_evaluation(self, board: chess.Board) -> float:
        """Calculate material balance."""
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        
        material_balance = 0
        for piece_type in piece_values:
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            material_balance += (white_pieces - black_pieces) * piece_values[piece_type]
        
        return material_balance
    
    def _positional_evaluation(self, board: chess.Board) -> float:
        """Basic positional evaluation."""
        score = 0
        
        # Center control
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                score += 0.3 if piece.color == chess.WHITE else -0.3
        
        # Piece mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK  
        black_mobility = len(list(board.legal_moves))
        board.turn = not board.turn  # Restore original turn
        
        score += (white_mobility - black_mobility) * 0.1
        
        return score
    
    def minimax(self, board: chess.Board, depth: int, alpha: float = -float('inf'), 
                beta: float = float('inf'), maximizing_player: bool = True) -> float:
        """Minimax algorithm with alpha-beta pruning."""

        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
         
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval
    
    def get_best_minimax_move(self, board: chess.Board) -> chess.Move:
        """Get the best move using minimax algorithm."""
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            move_value = self.minimax(board, self.minimax_depth - 1, 
                                    maximizing_player=(not board.turn))
            board.pop()
            
            if board.turn == chess.WHITE and move_value > best_value:
                best_value = move_value
                best_move = move
            elif board.turn == chess.BLACK and move_value < best_value:
                best_value = move_value
                best_move = move
        
        return best_move or random.choice(list(board.legal_moves))
    
    def update_q_values(self, game_states: List[Tuple[str, chess.Move]], 
                   final_reward: float):
        """Q-value update with experience replay."""
        if not game_states:
            return
        
        # Reverse iterate for temporal difference
        for i, (state, move) in enumerate(reversed(game_states)):
            if state not in self.q_table or move.uci() not in self.q_table[state]:
                continue
            
            # Calculate discounted reward
            steps_from_end = i
            discounted_reward = final_reward * (self.gamma ** steps_from_end)
            
            # Add exploration bonus for less visited states
            visit_bonus = 0.1 / (1 + sum(1 for _ in self.q_table[state].values()))
            total_reward = discounted_reward + visit_bonus
            
            # Update with momentum (running average)
            old_q = self.q_table[state][move.uci()]
            momentum = 0.9
            self.q_table[state][move.uci()] = (
                momentum * old_q + (1 - momentum) * 
                (old_q + self.alpha * (total_reward - old_q))
            )

    def train_single_game(self):
        """Train a single game and return the result."""
        self.reset_game()
        game_states: List[Tuple[str, chess.Move]] = []
        
        # Play one training game
        while (not self.board.is_game_over() and 
               len(game_states) < self.max_moves_per_game):
            
            state = self.get_board_state(self.board)
            move = self.get_move(self.board, use_exploration=True)
            game_states.append((state, move))
            self.board.push(move)
        
        # Determine game outcome and update Q-values
        result = self.board.result()
        if result == "1-0":  # White wins
            final_reward = 1.0
            outcome = "win"
        elif result == "0-1":  # Black wins  
            final_reward = -1.0
            outcome = "loss"
        else:  # Draw
            final_reward = 0.0
            outcome = "draw"
        
        self.update_q_values(game_states, final_reward)
        return outcome
    
    def save_q_table(self, filename: str = "q_table.pkl"):
        """Save the Q-table to a file."""
        def save_operation():
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            return True
        
        return self._safe_operation(save_operation, "Failed to save Q-table") or False

    def load_q_table(self, filename: str = "q_table.pkl"):
        """Load the Q-table from a file."""
        if not os.path.exists(filename):
            self.q_table = {}
            return False
        
        def load_operation():
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            return True
        
        return self._safe_operation(load_operation, "Failed to load Q-table") or False

    def _safe_operation(self, operation, error_message="Operation failed"):
        """Safely execute an operation with error handling."""
        try:
            return operation()
        except Exception as e:
            print(f"{error_message}: {e}")
            self.q_table = {} if "load" in error_message.lower() else self.q_table
            return None
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = chess.Board()
        self.move_history = []
        self.game_states = []
    
    def undo_move(self):
        """Undo the last move(s)."""
        if len(self.board.move_stack) >= 2:
            last_move = self.board.peek()  # Get last move
            # Undo both AI and human moves
            self.board.pop()  # AI move
            self.board.pop()  # Human move
            # Update move history
            if len(self.move_history) >= 2:
                self.move_history = self.move_history[:-2]
                return last_move if len(self.board.move_stack) > 0 else None
            return 
        
    def get_stats(self):
        """Get current AI statistics."""
        stats = {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "total_positions": sum(len(moves) for moves in self.q_table.values())
        }
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.engine:
                self.engine.quit()
                self.engine = None
        except Exception as e:
            print(f"Error during cleanup: {e}")
        try:
            self.save_q_table()  # Save Q-table on exit
        except Exception as e:
            print(f"Error saving Q-table during: {e}")    
