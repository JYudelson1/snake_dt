#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use decision_transformer_dfdx::{DTState, GetOfflineData};
use dfdx::{
    optim::{Adam, AdamConfig, Sgd, SgdConfig, WeightDecay},
    tensor::AutoDevice,
};
use rand::SeedableRng;
use snake::SnakeBoard;

mod data;
mod dt_trait;
mod snake;

fn play_one_game() -> Vec<SnakeBoard> {
    let mut states = vec![];

    let mut state = SnakeBoard::new_random();
    while state.still_playing {
        states.push(state.clone());
        //state.print();
        let moves = state.get_actions();
        let next_move = moves.iter().next().unwrap();
        //println!("{next_move:?}");
        state.apply_action(*next_move);
    }
    //state.print();
    states.push(state.clone());
    states
}

fn print_series(states: Vec<SnakeBoard>) {
    for state in states {
        //print!("Score: {}", state.points);
        println!("{:?}", state.facing);
        state.print();
    }
}

fn main() {
    // loop {
    //     let states = play_one_game();
    //     if states.last().unwrap().points >= 6 {
    //         print_series(states);
    //         return;
    //     }
    // }

    let dev: AutoDevice = Default::default();

    let mut model = SnakeBoard::build_model();

    // let adam_config = AdamConfig {
    //     lr: 2e-2,
    //     ..Default::default()
    // };
    // let config = adam_config;
    // let mut optimizer = Adam::new(&model.0, config);

    let mut optimizer = Sgd::new(
        &model.0,
        SgdConfig {
            lr: 1e-3,
            weight_decay: Some(WeightDecay::L2(1e-4)),
            ..Default::default()
        },
    );

    let temp = 0.5;

    let mut rng = rand::prelude::StdRng::from_seed([42; 32]);
    for i in 0..128 {
        let (batch, actions) = SnakeBoard::get_batch::<32, _>(&mut rng);

        for epoch in 0..32 {
            let loss = model.train_on_batch(batch.clone(), actions, &mut optimizer, &dev);
            println!("Loss at batch {i} epoch {epoch}: {loss:.3} (offline learn)\r");
        }

        if i % 64 == 0 {
            println!("Attempting to achieve 0 points:");
            model.evaluate(SnakeBoard::new_random(), temp, 0.0);
            println!("Attempting to achieve 2 points:");
            model.evaluate(SnakeBoard::new_random(), temp, 2.0);
            println!("Attempting to achieve 4 points:");
            model.evaluate(SnakeBoard::new_random(), temp, 4.0);
            println!("Attempting to achieve 6 points:");
            model.evaluate(SnakeBoard::new_random(), temp, 6.0);
            println!("Attempting to achieve 8 points:");
            model.evaluate(SnakeBoard::new_random(), temp, 8.0);
        }
    }

    // let rng = &mut thread_rng();
    // for i in 0..10 {
    //     let loss = model.online_learn::<100, _>(1.0, 3.0, &mut optimizer, &dev, rng);
    //     println!("Loss at epoch {i}: {loss:.3} (online learn)");
    // }

    // model.evaluate(SnakeBoard::new_random());
    // model.evaluate(SnakeBoard::new_random());
    // model.evaluate(SnakeBoard::new_random());
}
