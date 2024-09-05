use decision_transformer_dfdx::{DTModelConfig, DTState, GetOfflineData, HumanEvaluatable};
use dfdx::prelude::*;

use crate::{
    data::{play_game_random, play_game_smart, play_game_towards, play_game_up, play_game_well},
    snake::{Moves, SnakeBoard, BOARD_SIZE},
};

pub struct SnakeConfig;

impl DTModelConfig for SnakeConfig {
    const NUM_ATTENTION_HEADS: usize = 4;
    const HIDDEN_SIZE: usize = 512;
    const MLP_INNER: usize = 4 * 512;
    const SEQ_LEN: usize = 4;
    const MAX_EPISODES_IN_GAME: usize = 1024;
    const NUM_LAYERS: usize = 4;
}

impl DTState<f32, AutoDevice, SnakeConfig> for SnakeBoard {
    type Action = Moves;
    const STATE_SIZE: usize = { BOARD_SIZE * BOARD_SIZE + 4 };
    const ACTION_SIZE: usize = 4;

    fn apply_action(&mut self, action: Self::Action) {
        self.make_move(action)
    }

    fn get_reward(&self, action: Self::Action) -> f32 {
        let mut new = self.clone();
        new.apply_action(action);

        if new.points > self.points {
            1.0
        } else {
            0.0
        }
    }

    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), f32, AutoDevice>{
        let mut t: Tensor<(Const<{ Self::STATE_SIZE }>,), f32, Cpu> = Cpu::default().zeros();
        t = t - 1.0;
        for (i, row) in self.board.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                let pos = j + (row.len() * i);
                t[[pos]] = x.to_usize() as f32;
            }
        }

        t[[{ BOARD_SIZE * BOARD_SIZE }]] = self.snake_head.0 as f32;
        t[[{ BOARD_SIZE * BOARD_SIZE + 1 }]] = self.snake_head.1 as f32;
        t[[{ BOARD_SIZE * BOARD_SIZE + 2 }]] = self.apple.0 as f32;
        t[[{ BOARD_SIZE * BOARD_SIZE + 3 }]] = self.apple.1 as f32;

        let dev: AutoDevice = Default::default();
        t.to_device(&dev)
    }

    fn action_to_index(action: &Self::Action) -> usize {
        action.to_usize()
    }

    fn index_to_action(action: usize) -> Self::Action {
        match action {
            0 => Moves::Up,
            1 => Moves::Down,
            2 => Moves::Left,
            3 => Moves::Right,
            _ => panic!("{action} is not a valid move!"),
        }
    }

    fn new_random<R: rand::Rng + ?Sized>(_: &mut R) -> Self {
        Self::new_random()
    }
}

impl GetOfflineData<f32, AutoDevice, SnakeConfig> for SnakeBoard {
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>) {
        let x = rng.gen_range(0..4);
        let (games, actions) = match x {
            0 => play_game_random(),
            1 => play_game_smart(),
            2 => play_game_towards(),
            3 => play_game_well(),
            _ => unreachable!(),
        };
        // if games.last().unwrap().points >= 2 {
        //     println!("{} points", games.last().unwrap().points);
        // }
        (games, actions)
    }
}

impl HumanEvaluatable<f32, AutoDevice, SnakeConfig> for SnakeBoard {
    fn print(&self) {
        self.print();
    }

    fn print_action(action: &Self::Action) {
        println!("{action:?}")
    }

    fn is_still_playing(&self) -> bool {
        self.still_playing
    }
}
