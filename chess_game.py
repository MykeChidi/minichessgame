import chess
import pygame
import sys
import threading
import time
from typing import Tuple, Optional
from enum import Enum
import random
from chess_ai import ChessAI

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 640
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 50
SQUARE_SIZE = BOARD_SIZE // 8

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
HIGHLIGHT = (255, 255, 0, 128)
LEGAL_MOVE = (0, 255, 0, 128)
LAST_MOVE = (255, 165, 0, 128)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 149, 237)
TEXT_COLOR = (50, 50, 50)
BG_COLOR = (245, 245, 245)

class GameState(Enum):
    MENU = "menu"
    HUMAN_VS_AI = "human_vs_ai"
    TRAINING = "training"
    HUMAN_VS_HUMAN = "human_vs_human"
    PAUSED = "paused"

class Controller:
    """Game action controls and logic"""
    def __init__(self):
        self.game_state = GameState.MENU
        self.ai = ChessAI()
        self.player_color = chess.WHITE
        self.training_thread = None
        self.manager = None
        self.ai_move_timer_id = pygame.USEREVENT + 1
        self.timer_active = False
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.ai_thinking = False
        self.training_progress = {"games": 0, "total": 0, "wins": 0, "draws": 0, "losses": 0}

    def undo_move(self):
        """Undo the last move if possible."""
        if self.game_state == GameState.HUMAN_VS_AI:
            if not self.ai.board.is_game_over():
                undone_move = self.ai.undo_move()
                if undone_move:
                    self.last_move = self.ai.board.peek() if len(self.ai.board.move_stack) > 0 else None
                    self.selected_square = None
                    self.legal_moves = []
                    self.show_message(f"Move undone", 2000)
                    if self.ai.board.turn != self.player_color:
                        # If it's the not player's turn after undo, make AI move
                        self.make_ai_move()
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.ai.reset_game()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.ai_thinking = False
        self.message = ""
        self.message_timer = 0
        if self.game_state == GameState.HUMAN_VS_AI:
            self.start_human_game(self.player_color)

    def return_to_menu(self):
        """Return to the main menu."""
        self.game_state = GameState.MENU
        self.ai.reset_game()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.ai_thinking = False
    
    def show_message(self, message, duration=3000):
        """Show a message through the manager."""
        if self.manager:
            self.manager.show_message(message, duration)
    
    def _schedule_ai_move(self, delay_ms=3000):
        """Schedule an AI move with a timer."""
        if not self.timer_active:
            pygame.time.set_timer(self.ai_move_timer_id, delay_ms)
            self.timer_active = True
    
    def _cancel_ai_timer(self):
        """Cancel the AI move timer."""
        if self.timer_active:
            pygame.time.set_timer(self.ai_move_timer_id, 0)
            self.timer_active = False
    
    def quit_game(self):
        """Quit the application."""
        if self.training_thread and self.training_thread.is_alive():
            self.game_state = GameState.MENU
            self.training_thread.join(timeout=2.0)  # Wait for training to finish
        
        pygame.time.set_timer(pygame.USEREVENT + 1, 0)  
        # Clean up resources
        self.ai.cleanup()
        pygame.quit()
        sys.exit()

    def safe_operation(self, operation, error_message="Operation failed"):
        """Safely execute an operation with error handling."""
        try:
            return operation()
        except Exception as e:
            print(f"{error_message}: {e}")
            self.show_message(f"Error: {error_message}")
            return None
        
    def make_ai_move(self):
        """Make AI move in a separate thread."""
        if self.ai_thinking or self.ai.board.is_game_over():
            return
        
        # Move counter to prevent infinte loops
        if len(self.ai.board.move_stack) > 200:
            self.show_message("Game too long - ending in draw")
            return
        
        self.ai_thinking = True
        
        def ai_move_worker():
            try:
                with self.ai.lock:
                    if self.game_state == GameState.HUMAN_VS_AI:
                        # Human vs AI - use Q-learning
                        move = self.ai.get_move(self.ai.board, use_exploration=False)
                
                self.ai.board.push(move)
                self.ai.move_history.append(move)
                self.last_move = move
                    
            except Exception as e:
                print(f"AI move error: {e}")
            finally:
                self.ai_thinking = False
                
        threading.Thread(target=ai_move_worker, daemon=True).start()


    def start_human_vs_human(self):
        """Start game between two players"""
        self.ai.reset_game()
        self.game_state = GameState.HUMAN_VS_HUMAN
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None

    def start_human_game(self, player_color: chess.Color):
        """Start a game between human and AI."""
        self.ai.reset_game()
        self.game_state = GameState.HUMAN_VS_AI
        self.player_color = player_color
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.ai_thinking = False
        
        # If AI goes first, make its move
        if self.ai.board.turn != player_color:
            self.make_ai_move()

    def stop_training(self):
        """Stop the current training."""
        self.game_state = GameState.MENU
        self.show_message("Training stopped.")

    def validate_game_state_transition(self, new_state):
        """Validate if state transition is allowed."""
        valid_transitions = {
            GameState.MENU: [GameState.TRAINING, GameState.HUMAN_VS_AI,GameState.HUMAN_VS_HUMAN],
            GameState.TRAINING: [GameState.MENU],
            GameState.HUMAN_VS_AI: [GameState.MENU, GameState.PAUSED],
            GameState.HUMAN_VS_HUMAN: [GameState.MENU, GameState.PAUSED],
            GameState.PAUSED: [GameState.HUMAN_VS_AI, GameState.HUMAN_VS_HUMAN, GameState.MENU]
        }
        
        return new_state in valid_transitions.get(self.game_state, [])

    def set_game_state(self, new_state):
        """Safely change game state."""
        if self.validate_game_state_transition(new_state):
            self.game_state = new_state
            return True
        else:
            print(f"Invalid state transition from {self.game_state} to {new_state}")
            return False

    def start_training(self, num_games=100):
        """Start AI training in a separate thread."""
        if self.training_thread and self.training_thread.is_alive():
            return
        
        self.game_state = GameState.TRAINING
        self.training_progress = {"games": 0, "total": num_games, "wins": 0, "draws": 0, "losses": 0}
        
        def training_worker():
            batch_size = 5  # Process in batches
            outcomes = []
            
            for game_num in range(num_games):
                if self.game_state != GameState.TRAINING:
                    break
                
                outcome = self.ai.train_single_game()
                outcomes.append(outcome)
                
                # Update progress in batches
                if (game_num + 1) % batch_size == 0 or game_num == num_games - 1:
                    wins = outcomes.count("win")
                    draws = outcomes.count("draw")
                    losses = outcomes.count("loss")
                    
                    self.training_progress.update({
                        "games": game_num + 1,
                        "wins": self.training_progress["wins"] + wins,
                        "draws": self.training_progress["draws"] + draws,
                        "losses": self.training_progress["losses"] + losses
                    })
                    outcomes.clear()
                
                # Adaptive epsilon decay
                if game_num % 25 == 0 and game_num > 0:
                    self.ai.epsilon = max(
                        self.ai.epsilon_min, 
                        self.ai.epsilon * self.ai.epsilon_decay
                    )
                
                time.sleep(0.01)  # Reduced delay
            
            # Training complete
            if self.game_state == GameState.TRAINING:
                self.ai.save_q_table()
                self.show_message("Training completed!")
                self.game_state = GameState.MENU
        
        self.training_thread = threading.Thread(target=training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def _is_valid_move(self, move):
        """Validate special moves like castling and en passant."""
        try:
            # Create a copy of the board to test the move
            test_board = self.ai.board.copy()
            test_board.push(move)
            return True
        except ValueError:
            return False
    
    def _handle_pawn_promotion(self, move, square):
        """Handle pawn promotion (Queen or Rook)"""
        if move.promotion is None and self.ai.board.piece_at(self.selected_square).piece_type == chess.PAWN:
            # Check if pawn is reaching promotion rank
            if (self.ai.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
            (self.ai.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                # Randomly choose between Queen and Rook
                promotion_piece = random.choice([chess.QUEEN, chess.ROOK])
                move = chess.Move(move.from_square, move.to_square, promotion=promotion_piece)
                
                piece_name = "Queen" if promotion_piece == chess.QUEEN else "Rook"
                self.show_message(f"Pawn promoted to {piece_name}!")
        
        return move
    
    def handle_square_click(self, square: int):
        """Handle clicking on a chess square."""
        if not (0 <= square <= 63):
            return
        
         # Handle Human vs Human mode
        if self.game_state == GameState.HUMAN_VS_HUMAN:
            if self.ai_thinking or self.ai.board.is_game_over():
                return
            
            piece = self.ai.board.piece_at(square)
            
            # If clicking on own piece (current player's turn), select it
            if piece and piece.color == self.ai.board.turn:
                self.selected_square = square
                self.legal_moves = [move for move in self.ai.board.legal_moves 
                                if move.from_square == square]
            
            # If we have a piece selected and click on a legal move destination
            elif self.selected_square is not None:
                move = None
                for legal_move in self.legal_moves:
                    if legal_move.to_square == square:
                        move = legal_move
                        break
                
                if move:
                    # Validate special moves
                    if not self._is_valid_move(move):
                        self.show_message("Invalid move!", 2000)
                        self.selected_square = None
                        self.legal_moves = []
                        return
                    
                     # Handle pawn promotion (queen or rook)
                    move = self._handle_pawn_promotion(move, square)

                    # Make the move
                    try:
                        self.ai.board.push(move)
                        self.ai.move_history.append(move)
                        self.last_move = move
                        self.selected_square = None
                        self.legal_moves = []
                            
                    except ValueError:
                        # Invalid move
                        pass
                else:
                    # Clear selection if clicking elsewhere
                    self.selected_square = None
                    self.legal_moves = []
            
            return  

        if self.game_state != GameState.HUMAN_VS_AI or self.ai_thinking:
            return
        
        if self.ai.board.turn != self.player_color or self.ai.board.is_game_over():
            return
        
        piece = self.ai.board.piece_at(square)
        
        # If clicking on own piece, select it
        if piece and piece.color == self.player_color:
            self.selected_square = square
            self.legal_moves = [move for move in self.ai.board.legal_moves 
                              if move.from_square == square]
        
        # If we have a piece selected and click on a legal move destination
        elif self.selected_square is not None:
            move = None
            for legal_move in self.legal_moves:
                if legal_move.to_square == square:
                    move = legal_move
                    break
            
            if move:
                # Validate special moves
                if not self._is_valid_move(move):
                    self.show_message("Invalid move!", 2000)
                    self.selected_square = None
                    self.legal_moves = []
                    return
                
                # Handle pawn promotion (queen or rook)
                move = self._handle_pawn_promotion(move, square)

                # Make the move
                try:
                    self.ai.board.push(move)
                    self.ai.move_history.append(move)
                    self.last_move = move
                    self.selected_square = None
                    self.legal_moves = []
                    
                    # Make AI response
                    if not self.ai.board.is_game_over():
                        self.make_ai_move()
                        
                except ValueError:
                    # Invalid move
                    pass
            else:
                # Clear selection if clicking elsewhere
                self.selected_square = None
                self.legal_moves = []


class RenderUI:
    """Game UI renderer"""
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Let's Play Chess")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 18)
        
        self.selected_square = None
        self.game_state = GameState.MENU
        self.legal_moves = []
        self.buttons = []
        self.last_move = None
        self.controller = Controller()
        self.manager = None
        self.player_color = chess.WHITE
        self.ai = ChessAI()
        
        self.ai_thinking = False

        try:
            self._load_piece_images_from_files()
        except:
            self._draw_piece_images()

    def _draw_piece_images(self):
        """Load chess piece images - draws recognizable piece shapes."""
        self.piece_images = {}
        
        for piece_type in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']:
            is_white = piece_type.isupper()
            piece_color = WHITE if is_white else BLACK
            outline_color = BLACK if is_white else WHITE
            
            # Create surface for piece
            piece_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            
            # Draw piece based on type
            center_x, center_y = SQUARE_SIZE // 2, SQUARE_SIZE // 2
            
            if piece_type.upper() == 'P':  # Pawn
                # Draw pawn shape
                pygame.draw.circle(piece_surface, outline_color, (center_x, center_y - 8), 12)
                pygame.draw.circle(piece_surface, piece_color, (center_x, center_y - 8), 10)
                pygame.draw.rect(piece_surface, outline_color, (center_x - 8, center_y + 2, 16, 12))
                pygame.draw.rect(piece_surface, piece_color, (center_x - 6, center_y + 4, 12, 8))
                
            elif piece_type.upper() == 'R':  # Rook
                # Draw rook shape
                pygame.draw.rect(piece_surface, outline_color, (center_x - 12, center_y - 8, 24, 20))
                pygame.draw.rect(piece_surface, piece_color, (center_x - 10, center_y - 6, 20, 16))
                # Crenellations
                for i in range(0, 5):
                    x = center_x - 10 + i * 4
                    if i % 2 == 0:
                        pygame.draw.rect(piece_surface, outline_color, (x, center_y - 12, 4, 6))
                        pygame.draw.rect(piece_surface, piece_color, (x + 1, center_y - 11, 2, 4))
                
            elif piece_type.upper() == 'N':  # Knight
                # Draw knight shape (simplified horse head)
                points = [
                    (center_x - 8, center_y + 8),
                    (center_x - 12, center_y - 4),
                    (center_x - 8, center_y - 12),
                    (center_x + 4, center_y - 8),
                    (center_x + 8, center_y - 4),
                    (center_x + 12, center_y + 8)
                ]
                pygame.draw.polygon(piece_surface, outline_color, points)
                # Inner shape
                inner_points = [(x + (1 if x < center_x else -1), y + 1) for x, y in points[:-1]]
                pygame.draw.polygon(piece_surface, piece_color, inner_points)
                
            elif piece_type.upper() == 'B':  # Bishop
                # Draw bishop shape
                pygame.draw.circle(piece_surface, outline_color, (center_x, center_y - 8), 8)
                pygame.draw.circle(piece_surface, piece_color, (center_x, center_y - 8), 6)
                pygame.draw.polygon(piece_surface, outline_color, [
                    (center_x - 6, center_y - 2), 
                    (center_x + 6, center_y - 2),
                    (center_x + 10, center_y + 10),
                    (center_x - 10, center_y + 10)
                ])
                pygame.draw.polygon(piece_surface, piece_color, [
                    (center_x - 4, center_y), 
                    (center_x + 4, center_y),
                    (center_x + 8, center_y + 8),
                    (center_x - 8, center_y + 8)
                ])
                # Cross on top
                pygame.draw.line(piece_surface, outline_color, (center_x, center_y - 14), (center_x, center_y - 10), 2)
                pygame.draw.line(piece_surface, outline_color, (center_x - 2, center_y - 12), (center_x + 2, center_y - 12), 2)
                
            elif piece_type.upper() == 'Q':  # Queen
                # Draw queen shape
                pygame.draw.polygon(piece_surface, outline_color, [
                    (center_x - 12, center_y + 8),
                    (center_x - 8, center_y - 4),
                    (center_x - 4, center_y - 8),
                    (center_x, center_y - 12),
                    (center_x + 4, center_y - 8),
                    (center_x + 8, center_y - 4),
                    (center_x + 12, center_y + 8)
                ])
                pygame.draw.polygon(piece_surface, piece_color, [
                    (center_x - 10, center_y + 6),
                    (center_x - 6, center_y - 2),
                    (center_x - 2, center_y - 6),
                    (center_x, center_y - 10),
                    (center_x + 2, center_y - 6),
                    (center_x + 6, center_y - 2),
                    (center_x + 10, center_y + 6)
                ])
                # Crown points
                for i in range(-2, 3):
                    x = center_x + i * 4
                    pygame.draw.circle(piece_surface, outline_color, (x, center_y - 10 - abs(i) * 2), 2)
                    
            elif piece_type.upper() == 'K':  # King
                # Draw king shape
                pygame.draw.polygon(piece_surface, outline_color, [
                    (center_x - 10, center_y + 8),
                    (center_x - 6, center_y - 2),
                    (center_x + 6, center_y - 2),
                    (center_x + 10, center_y + 8)
                ])
                pygame.draw.polygon(piece_surface, piece_color, [
                    (center_x - 8, center_y + 6),
                    (center_x - 4, center_y),
                    (center_x + 4, center_y),
                    (center_x + 8, center_y + 6)
                ])
                # Crown
                pygame.draw.rect(piece_surface, outline_color, (center_x - 8, center_y - 8, 16, 6))
                pygame.draw.rect(piece_surface, piece_color, (center_x - 6, center_y - 6, 12, 2))
                # Cross on top
                pygame.draw.line(piece_surface, outline_color, (center_x, center_y - 14), (center_x, center_y - 8), 3)
                pygame.draw.line(piece_surface, outline_color, (center_x - 3, center_y - 11), (center_x + 3, center_y - 11), 3)
            
            self.piece_images[piece_type] = piece_surface

    def _load_piece_images_from_files(self):
        """Load chess piece images from files."""
        self.piece_images = {}
        
        # Dictionary mapping piece symbols to filenames
        piece_files = {
            'K': 'pieces/king_white.png',   'Q': 'pieces/queen_white.png',   'R': 'pieces/rook_white.png',
            'B': 'pieces/bishop_white.png', 'N': 'pieces/knight_white.png',  'P': 'pieces/pawn_white.png',
            'k': 'pieces/king_black.png',   'q': 'pieces/queen_black.png',   'r': 'pieces/rook_black.png',
            'b': 'pieces/bishop_black.png', 'n': 'pieces/knight_black.png',  'p': 'pieces/pawn_black.png'
        }
        
        for piece_symbol, filename in piece_files.items():
            try:
                # Try to load the image file
                image = pygame.image.load(filename)
                # Scale to fit square size
                scaled_image = pygame.transform.scale(image, (SQUARE_SIZE - 4, SQUARE_SIZE - 4))
                
                # Create surface with proper positioning
                piece_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                piece_surface.blit(scaled_image, (2, 2))  # Center with 2px margin
                
                self.piece_images[piece_symbol] = piece_surface
            except pygame.error:
                # Fallback to text if image not found
                self._draw_piece_images()

    def square_to_pos(self, square: int) -> Tuple[int, int]:
        """Convert chess square to screen position."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        return x, y
    
    def pos_to_square(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert screen position to chess square."""
        x, y = pos
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            file = x // SQUARE_SIZE
            rank = 7 - (y // SQUARE_SIZE)
            return chess.square(file, rank)
        return None

    def draw_board(self):
        """Draw the chess board."""
        for rank in range(8):
            for file in range(8):
                x = file * SQUARE_SIZE
                y = rank * SQUARE_SIZE
                color = LIGHT_BROWN if (rank + file) % 2 == 0 else DARK_BROWN
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Draw coordinates
                square = chess.square(file, 7 - rank)
                if file == 0:  # Rank labels
                    rank_label = str(8 - rank)
                    text = self.small_font.render(rank_label, True, TEXT_COLOR)
                    self.screen.blit(text, (x + 2, y + 2))
                if rank == 7:  # File labels
                    file_label = chr(ord('A') + file)
                    text = self.small_font.render(file_label, True, TEXT_COLOR)
                    self.screen.blit(text, (x + SQUARE_SIZE - 12, y + SQUARE_SIZE - 15))
    
    def draw_highlights(self):
        """Draw square highlights."""
        # Highlight selected square
        if self.selected_square is not None:
            x, y = self.square_to_pos(self.selected_square)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT)
            self.screen.blit(highlight_surface, (x, y))
        
        # Highlight legal moves
        for move in self.legal_moves:
            x, y = self.square_to_pos(move.to_square)
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(LEGAL_MOVE)
            self.screen.blit(highlight_surface, (x, y))
        
        # Highlight last move
        if self.last_move:
            for square in [self.last_move.from_square, self.last_move.to_square]:
                x, y = self.square_to_pos(square)
                highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                highlight_surface.fill(LAST_MOVE)
                self.screen.blit(highlight_surface, (x, y))
    
    def draw_pieces(self):
        """Draw chess pieces on the board."""
        for square in chess.SQUARES:
            piece = self.ai.board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                x, y = self.square_to_pos(square)
                self.screen.blit(self.piece_images[symbol], (x, y))

    def draw_sidebar(self):
        """Draw the sidebar with game information."""

        if self.controller:
            self.game_state = self.controller.game_state
            self.ai_thinking = self.controller.ai_thinking
            self.player_color = self.controller.player_color
            self.selected_square = self.controller.selected_square
            self.legal_moves = self.controller.legal_moves
            self.last_move = self.controller.last_move
        
        sidebar_x = BOARD_SIZE
        pygame.draw.rect(self.screen, BG_COLOR, (sidebar_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        
        y_offset = 20
        
        # Title
        title = self.title_font.render("Chess Game", True, TEXT_COLOR)
        self.screen.blit(title, (sidebar_x + 10, y_offset))
        y_offset += 50
        
        # Game state info
        if self.game_state == GameState.HUMAN_VS_AI:
            player1_color_text = "White" if self.player_color else "Black"
            player2_color_text = "Black" if self.player_color else "White"
            
            text = self.font.render(f"Player: {player1_color_text}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            text = self.font.render(f"AI: {player2_color_text}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            # Current turn
            current_player = "White" if self.ai.board.turn == self.player_color else "Black"
            if self.ai_thinking:
                current_player = "AI thinking..."
                

            text = self.font.render(f"Current player: {current_player}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 40

        elif self.game_state == GameState.HUMAN_VS_HUMAN:
            player1_color_text = "White" 
            player2_color_text = "Black"
            
            text = self.font.render(f"Player_1: {player1_color_text}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            text = self.font.render(f"Player_2: {player2_color_text}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            # Current turn
            current_player = "White" if self.ai.board.turn == chess.WHITE else "Black"

            text = self.font.render(f"Current player: {current_player}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 40
        
        elif self.game_state == GameState.TRAINING:
            text = self.font.render("Training in Progress", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 30
            
            progress = self.controller.training_progress
            text = self.font.render(f"Games: {progress['games']}/{progress['total']}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 20
            
            text = self.font.render(f"Wins: {progress['wins']} Draws: {progress['draws']} Losses: {progress['losses']}", True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 40
        
        # Game status
        if self.ai.board.is_game_over():
            result = self.ai.board.result()
            if result == "1-0":
                status = "White Wins!"
            elif result == "0-1":
                status = "Black Wins!"
            else:
                status = "Draw!"
             
            # Round winner status
            text = self.font.render(status, True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 30
        
        # Message display
        if self.manager.message and self.manager.message_timer > 0:
            text = self.small_font.render(self.manager.message, True, TEXT_COLOR)
            self.screen.blit(text, (sidebar_x + 10, y_offset))
            y_offset += 25
        
        self.buttons = [] #clear buttons list 
        # Control buttons
        y_offset = max(y_offset, WINDOW_HEIGHT - 200)
        
        buttons = []
        if self.game_state == GameState.MENU:
            buttons = [
                ("Train AI", self.controller.start_training),
                ("Play as white ", lambda: self.controller.start_human_game(chess.WHITE)),
                ("Play as black", lambda: self.controller.start_human_game(chess.BLACK)),
                ("Two Players", self.controller.start_human_vs_human),
            ]

        elif self.game_state == GameState.PAUSED:
            buttons = [
                ("Resume Game", lambda: self.controller.set_game_state(GameState.HUMAN_VS_AI)),
                ("New Game", self.controller.reset_game),
                ("Return to Menu", self.controller.return_to_menu),
            ]

        elif self.game_state == GameState.HUMAN_VS_AI:
            buttons = [
                 ("Undo ", self.controller.undo_move),
                ("New Game", self.controller.reset_game),
                ("Return to Menu", self.controller.return_to_menu),
            ]

        elif self.game_state == GameState.HUMAN_VS_HUMAN:
            buttons = [
                ("New Game", self.controller.reset_game),
                ("Return to Menu", self.controller.return_to_menu),
            ]

        elif self.game_state == GameState.TRAINING:
            buttons = [
                ("Stop Training", self.controller.stop_training)
            ]
        
        for i, (button_text, callback) in enumerate(buttons):
            button_rect = pygame.Rect(sidebar_x + 10, y_offset + i * 35, 200, 30)
            mouse_pos = pygame.mouse.get_pos()
            button_color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, BLACK, button_rect, 2)
            
            text = self.font.render(button_text, True, WHITE)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
            # Store button info for click handling
            self.buttons.append((button_rect, callback))
    
    def draw_menu(self):
        """Draw the main menu."""
        self.screen.fill(BG_COLOR)
        self.buttons = []

        # Title
        title = pygame.font.Font(None, 48).render("Chess Game", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font.render("Select Game options", True, TEXT_COLOR)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 140))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu buttons
        buttons = [
            ("Train AI", self.controller.start_training),
            ("Play as White", lambda: self.controller.start_human_game(chess.WHITE)),
            ("Play as Black", lambda: self.controller.start_human_game(chess.BLACK)),
            ("Two Players", self.controller.start_human_vs_human),
            ("Exit", self.controller.quit_game)
            ]

        for i, (button_text, callback) in enumerate(buttons):
            button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 200 + i * 50, 200, 40)
            mouse_pos = pygame.mouse.get_pos()
            button_color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
            
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, BLACK, button_rect, 2)
        
            text = self.font.render(button_text, True, WHITE)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
            self.buttons.append((button_rect, callback))
        
        footer = WINDOW_HEIGHT - 100
        text = self.small_font.render(f"Designed and developed by - MykeChidi", True, TEXT_COLOR)
        self.screen.blit(text, (20, footer + 50))

        # Message display
        if self.manager and self.manager.message and self.manager.message_timer > 0:
            text = self.font.render(self.manager.message, True, TEXT_COLOR)  
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 40))
            self.screen.blit(text, text_rect)
        

class Handler:
    """Game events and action handler """
    def __init__(self):
        self.controller = Controller()
        self.renderer = RenderUI()
        self.ai = ChessAI()
        self.game_state = GameState.MENU
        self.manager = ManageUI()

        # Cross references
        self.controller.manager = self.manager
        self.renderer.controller = self.controller
        self.renderer.manager = self.manager
        self.manager.controller = self.controller

    def handle_button_click(self, pos):
        """Handle button clicks."""
        if not hasattr(self.renderer, 'buttons') or not self.renderer.buttons:
            return
        
        for button_rect, callback in self.renderer.buttons:
            if button_rect.collidepoint(pos):
                callback()
                break

    def handle_event(self, event):
        """Handle pygame events."""
        self.game_state = self.controller.game_state
        if event.type == pygame.QUIT:
            self.controller.quit_game()
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = pygame.mouse.get_pos()
                
                # Check if clicking on board
                if pos[0] < BOARD_SIZE and self.controller.game_state in [GameState.HUMAN_VS_AI, GameState.HUMAN_VS_HUMAN]:
                    square = self.renderer.pos_to_square(pos)
                    if square is not None:
                        self.controller.handle_square_click(square)
                
                # Check button clicks
                self.handle_button_click(pos)
        
        elif event.type == self.controller.ai_move_timer_id:  # AI timer #:
            self.controller._cancel_ai_timer()  # Cancel timer# 
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if self.game_state != GameState.MENU:
                    self.controller.return_to_menu()
                else:
                    self.controller.quit_game()
            
            elif event.key == pygame.K_r and self.game_state in [GameState.HUMAN_VS_AI, GameState.HUMAN_VS_HUMAN]:
                # Reset game
                if self.game_state == GameState.HUMAN_VS_AI:
                    self.controller.start_human_game(self.renderer.player_color)
                elif self.game_state == GameState.HUMAN_VS_HUMAN:
                    self.controller.start_human_vs_human()
                
class ManageUI:
    """Game UI event manager"""
    def __init__(self):
        self.message = ""
        self.message_timer = 0
        self.controller = None
        self.input_dialog_active = False
        self.input_text = "" 
        self.input_cursor = 0

    def show_message(self, message: str, duration: int = 3000):
        """Show a temporary message."""
        self.message = message
        self.message_timer = duration
    
    def update(self, dt):
        """Update game state."""
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""
                self.message_timer = 0

class ChessGameApp:
    """Main entry into application"""
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.message_timer = 3000
        
        
        self.handler = Handler()
        self.manager = self.handler.manager
        self.renderer = self.handler.renderer
        self.controller = self.handler.controller
        
        self.game_state = GameState.MENU
        
        # Initialize engine
        engine_paths = ["engine\\stockfish-windows-x86-64-avx2.exe",
                    "stockfish",
                    "engine/stockfish",
                    "/usr/bin/stockfish",
                    "/usr/local/bin/stockfish"]
        
        self.ai = None
        for engine_path in engine_paths:
            try:
                self.ai = ChessAI(engine_path)
                break
            except:
                continue
        
        if self.ai is None:
            print("Unable to initialize engine, Using a dummy engine by default")
            self.ai = ChessAI("dummy_engine")
        
        # Update the handler's AI reference
        self.handler.ai = self.ai
        self.controller.ai = self.ai
        self.renderer.ai = self.ai
        
    def load_game(self):
        """Draw the entire game."""
        if self.controller.game_state == GameState.MENU:
            self.renderer.draw_menu()
        else:
            self.screen.fill(BG_COLOR)
            self.renderer.draw_board()
            self.renderer.draw_highlights()
            self.renderer.draw_pieces()
            self.renderer.draw_sidebar()

    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            dt = self.clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                self.handler.handle_event(event)
            
            # Update
            self.manager.update(dt)
            
            # Start
            self.load_game()
            pygame.display.flip()
        
        self.ai.cleanup()
        pygame.quit()

def main():
    """Main function to start the game."""
    try:
        game = ChessGameApp()
        game.run()
    except Exception as e:
        print(f"Error starting game: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
