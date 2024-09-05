use decision_transformer_dfdx::DTState;
use dfdx::tensor::Cpu;

use crate::{
    dt_trait::SnakeConfig,
    snake::{Moves, SnakeBoard},
};

pub fn play_game_random() -> (
    Vec<SnakeBoard>,
    Vec<<SnakeBoard as DTState<f32, Cpu, SnakeConfig>>::Action>,
) {
    let mut states = vec![];
    let mut actions = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());
        //state.print();
        let moves = state.get_actions();
        let next_move = moves.iter().next().unwrap();
        actions.push(next_move.clone());
        //println!("{next_move:?}");
        state.apply_action(*next_move);
    }
    (states, actions)
}

pub fn play_game_up() -> (
    Vec<SnakeBoard>,
    Vec<<SnakeBoard as DTState<f32, Cpu, SnakeConfig>>::Action>,
) {
    let mut states = vec![];
    let mut actions = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());
        //state.print();
        let next_move = Moves::Up;
        actions.push(next_move.clone());
        //println!("{next_move:?}");
        state.apply_action(next_move);
    }
    (states, actions)
}

pub fn play_game_smart() -> (
    Vec<SnakeBoard>,
    Vec<<SnakeBoard as DTState<f32, Cpu, SnakeConfig>>::Action>,
) {
    let mut states = vec![];
    let mut actions = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());
        //state.print();
        let moves = state.get_actions();

        let mut moved = false;
        for next_move in moves.iter() {
            let mut maybe_state = state.clone();
            maybe_state.apply_action(next_move.clone());
            if maybe_state.still_playing {
                actions.push(next_move.clone());
                state.apply_action(*next_move);
                moved = true;
                break;
            }
        }

        if !moved {
            let next_move = moves.iter().next().unwrap();
            actions.push(next_move.clone());
            state.apply_action(*next_move);
        }
    }
    (states, actions)
}

pub fn play_game_well() -> (
    Vec<SnakeBoard>,
    Vec<<SnakeBoard as DTState<f32, Cpu, SnakeConfig>>::Action>,
) {
    let mut states = vec![];
    let mut actions = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());
        //state.print();
        let moves = state.get_actions();

        // Check for apple
        let mut moved = false;
        for next_move in moves.iter() {
            let mut maybe_state = state.clone();
            maybe_state.apply_action(next_move.clone());
            if maybe_state.points > state.points {
                actions.push(next_move.clone());
                state.apply_action(*next_move);
                moved = true;
                break;
            }
        }

        // Check for survival
        if !moved {
            for next_move in moves.iter() {
                let mut maybe_state = state.clone();
                maybe_state.apply_action(next_move.clone());
                if maybe_state.still_playing {
                    actions.push(next_move.clone());
                    state.apply_action(*next_move);
                    moved = true;
                    break;
                }
            }
        }

        // Make any random move
        if !moved {
            let next_move = moves.iter().next().unwrap();
            actions.push(next_move.clone());
            state.apply_action(*next_move);
        }
    }
    (states, actions)
}

pub fn play_game_towards() -> (
    Vec<SnakeBoard>,
    Vec<<SnakeBoard as DTState<f32, Cpu, SnakeConfig>>::Action>,
) {
    let mut states = vec![];
    let mut actions = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());

        let (head_x, head_y) = state.snake_head;
        let (apple_x, apple_y) = state.apple;

        let move_candidate = if apple_x < head_x {
            Moves::Left
        } else if apple_x > head_x {
            Moves::Right
        } else {
            if apple_y < head_y {
                Moves::Up
            } else {
                Moves::Down
            }
        };
        let moves = state.get_actions();
        if moves.contains(&move_candidate) {
            actions.push(move_candidate.clone());
            state.apply_action(move_candidate);
        } else {
            let next_move = moves.iter().next().unwrap();
            actions.push(next_move.clone());
            state.apply_action(*next_move);
        }
    }
    (states, actions)
}
