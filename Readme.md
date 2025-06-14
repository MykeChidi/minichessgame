# Chess Game with AI

A mini chess game built with Python and Pygame, featuring an AI opponent that learns through Q-learning and uses minimax algorithms for strategic play.

## Features

### Game Modes
- **Human vs AI**: Play against an intelligent AI opponent
- **Human vs Human**: Two-player local chess game
- **AI Training**: Train the AI to improve its performance
- **Color Selection**: Choose to play as White or Black against AI

### AI Capabilities
- **Q-Learning**: The AI learns from games and improves over time
- **Minimax Algorithm**: Strategic move evaluation with alpha-beta pruning
- **Stockfish Integration**: Enhanced position evaluation when Stockfish engine is available
- **Adaptive Learning**: Epsilon-greedy exploration with decay
- **Persistent Learning**: Q-table saves and loads automatically
- **Position Evaluation**: Material, positional, and engine-based scoring
- **Experience Replay**: Efficient learning from game history

### Game Features
- **Full Chess Rules**: Complete implementation including castling, en passant, and pawn promotion
- **Visual Interface**: Intuitive graphical board with piece highlighting
- **Move Validation**: Legal move checking and game state detection
- **Game History**: Move tracking and undo functionality
- **Real-time Feedback**: Game status, turn indicators, and messages
- **Memory Management**: Q-table cleanup prevents memory overflow

### Menu Options
1. **Train AI**: Run training sessions to improve AI performance
2. **Play as White**: Start game as White pieces
3. **Play as Black**: Start game as Black pieces  
4. **Two Players**: Local multiplayer mode
5. **Exit**: Quit the application

### Game Controls
- **Mouse**: Click to select pieces and make moves
- **ESC**: Return to menu or quit game
- **R**: Reset current game
- **Visual Cues**:
  - Yellow highlight: Selected piece
  - Green highlight: Legal moves
  - Orange highlight: Last move made

## Installation

### Prerequisites
- Python 3.10 or higher
- Required Python packages:
  ```bash
  pip install pygame python-chess
  ```

### Optional: Stockfish Engine
For enhanced AI performance, install Stockfish:
- **Windows**: Download from [Stockfish website](https://stockfishchess.org/download/)
- **Linux**: `sudo apt-get install stockfish`
- **macOS**: `brew install stockfish`

Place the Stockfish executable in one of these locations:
- `engine/stockfish-windows-x86-64-avx2.exe` (Windows)
- `engine/stockfish` (Linux/Mac)
- System PATH as `stockfish`

## Usage

### Running the Game
```bash
python chess_game.py
```

## AI Training

The AI uses Q-learning to improve its chess playing:

### Training Process
- Games are played automatically between AI instances
- Q-values are updated based on game outcomes
- Learning rate and exploration decrease over time
- Progress is displayed in real-time

### Training Parameters
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.9
- **Exploration Rate (ε)**: Starts at 0.3, decays to 0.01
- **Q-table Size Limit**: 40,000 positions

### Performance Tips
- Train for 100+ games for noticeable improvement
- Longer training sessions yield better results
- Q-table is automatically saved and loaded
- Close other applications during training
- Use SSD storage for faster Q-table loading/saving

## Technical Details

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install pygame and python-chess
2. **Stockfish Not Found**: Game works without it, but AI is weaker
3. **Slow Performance**: Reduce minimax depth or train with fewer games
4. **Memory Usage**: Q-table automatically cleans up old positions

## Credits

**Designed and developed by MykeChidi**

### Dependencies
- [Pygame](https://pygame.org/) - Game development library
- [python-chess](https://python-chess.readthedocs.io/) - Chess game logic
- [Stockfish](https://stockfishchess.org/) - Optional chess engine

---

*Enjoy playing chess and watching the AI learn and improve!*