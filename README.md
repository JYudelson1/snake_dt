# Snake Game Decision Transformer

A reference implementation showing how to use the [decision-transformer-dfdx](https://github.com/JYudelson1/decision-transformer-dfdx) crate. This project implements a simple Snake game environment and trains a Decision Transformer to play it.

## Overview

This project serves as a complete example of how to implement the necessary traits and structures for using decision-transformer-dfdx. It includes:

- A basic Snake game implementation (`snake.rs`)
- Configuration setup (`dt_trait.rs`)
- Training data generation (`data.rs`)
- Training loop example (`main.rs`)

## Implementation Details

### Game State (`snake.rs`)
- 3x3 board representation
- Simple snake mechanics (movement, apple collection)
- Four possible actions (Up, Down, Left, Right)

### Decision Transformer Setup (`dt_trait.rs`)
```rust
// Example configuration
struct SnakeConfig;

impl DTModelConfig for SnakeConfig {
    const NUM_ATTENTION_HEADS: usize = 4;
    const HIDDEN_SIZE: usize = 16;
    const MLP_INNER: usize = 4 * 16;
    const SEQ_LEN: usize = 4;
    const MAX_EPISODES_IN_GAME: usize = 1024;
    const NUM_LAYERS: usize = 4;
}
```

### Training Data Generation (`data.rs`)
Shows multiple strategies for generating training data:
- Random movement
- Smart movement (avoiding immediate death)
- Path-finding towards apple
- Expert policy
- Direct movement towards apple

## Key Features Demonstrated

1. **State Representation**: Shows how to convert a game state into tensors
```rust
impl DTState<f32, AutoDevice, SnakeConfig> for SnakeBoard {
    const STATE_SIZE: usize = 7 * BOARD_SIZE * BOARD_SIZE + 4;
    const ACTION_SIZE: usize = 4;
    // ...
}
```

2. **Action Space**: Example of discrete action space implementation
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Moves {
    Up,
    Down,
    Left,
    Right,
}
```

3. **Reward Function**: Example reward shaping
```rust
fn get_reward(&self, action: Self::Action) -> f32 {
    // Apple collection reward
    if new.points > self.points {
        1.0
    } else {
        // Distance-based reward
        let old_distance = /* ... */;
        let new_distance = /* ... */;
        0.1 * (old_distance - new_distance)
    }
}
```

4. **Multiple Training Data Sources**: Demonstrates how to implement `GetOfflineData`
```rust
impl GetOfflineData<f32, AutoDevice, SnakeConfig> for SnakeBoard {
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>) {
        // Randomly choose between different play strategies
        match rng.gen_range(0..4) {
            0 => play_game_random(),
            1 => play_game_smart(),
            2 => play_game_towards(),
            3 => play_game_well(),
            _ => unreachable!(),
        }
    }
}
```

## Running the Example

```bash
# Clone the repository
git clone https://github.com/JYudelson1/snake_dt
cd snake_dt

# Run the training example
cargo run --release
```

## Learning from this Example

When implementing your own environment using decision-transformer-dfdx, pay attention to:

1. **State Encoding**: See how the snake board is encoded into a tensor
2. **Action Space**: How discrete actions are handled
3. **Reward Design**: The balance between immediate rewards (apple collection) and shaping rewards (distance to apple)
4. **Training Data**: Multiple strategies for generating training data
5. **Visualization**: Implementation of `HumanEvaluatable` for debugging

## References

- Main crate: [decision-transformer-dfdx](https://github.com/JYudelson1/decision-transformer-dfdx)
- Original paper: [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)