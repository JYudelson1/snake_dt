use std::collections::HashSet;

use rand::Rng;

pub const BOARD_SIZE: usize = 8;

#[derive(Debug, Clone, Copy)]
pub enum Cell {
    Apple,
    Empty,
    Snake(Option<Moves>),
}

impl Cell {
    pub fn to_usize(&self) -> usize {
        match self {
            Cell::Apple => 1,
            Cell::Empty => 0,
            Cell::Snake(dir) => match dir {
                Some(d) => match d {
                    Moves::Up => 3,
                    Moves::Down => 4,
                    Moves::Left => 5,
                    Moves::Right => 6,
                },
                None => 2,
            },
        }
    }
    pub fn print(&self) -> String {
        match self {
            Cell::Apple => "🍏".to_string(),
            Cell::Empty => "⬛".to_string(),
            Cell::Snake(_) => "⬜".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnakeBoard {
    pub board: [[Cell; BOARD_SIZE]; BOARD_SIZE],
    pub points: usize,
    pub facing: Option<Moves>,
    pub snake_head: (usize, usize),
    pub snake_tail: (usize, usize),
    pub apple: (usize, usize),
    pub still_playing: bool,
}

impl SnakeBoard {
    pub fn new_random() -> Self {
        let mut board = [[Cell::Empty; BOARD_SIZE]; BOARD_SIZE];

        let rand_x_snake = rand::thread_rng().gen_range(0..BOARD_SIZE);
        let rand_y_snake = rand::thread_rng().gen_range(0..BOARD_SIZE);

        board[rand_y_snake][rand_x_snake] = Cell::Snake(None);

        let mut rand_x = rand::thread_rng().gen_range(0..BOARD_SIZE);
        let rand_y = rand::thread_rng().gen_range(0..BOARD_SIZE);

        if rand_x == rand_x_snake {
            rand_x = (BOARD_SIZE - 1) - rand_x_snake;
        }

        board[rand_y][rand_x] = Cell::Apple;

        Self {
            board,
            points: 0,
            facing: None,
            snake_head: (rand_x_snake, rand_y_snake),
            snake_tail: (rand_x_snake, rand_y_snake),
            apple: (rand_x, rand_y),
            still_playing: true,
        }
    }

    pub fn get_actions(&self) -> HashSet<Moves> {
        let mut moves = HashSet::from([Moves::Down, Moves::Up, Moves::Left, Moves::Right]);

        match self.facing {
            Some(direction) => match direction {
                Moves::Up => {
                    moves.remove(&Moves::Down);
                }
                Moves::Down => {
                    moves.remove(&Moves::Up);
                }
                Moves::Left => {
                    moves.remove(&Moves::Right);
                }
                Moves::Right => {
                    moves.remove(&Moves::Left);
                }
            },
            None => (),
        }

        moves
    }

    pub fn make_move(&mut self, action: Moves) {
        if !self.still_playing {
            return;
        }

        self.facing = Some(action);

        let (mut head_x, mut head_y) = self.snake_head;

        self.board[head_y][head_x] = Cell::Snake(Some(action));

        match action {
            Moves::Up => {
                if head_y == 0 {
                    self.still_playing = false;
                    return;
                }
                head_y -= 1;
            }
            Moves::Down => {
                if head_y == (BOARD_SIZE - 1) {
                    self.still_playing = false;
                    return;
                }
                head_y += 1;
            }
            Moves::Left => {
                if head_x == 0 {
                    self.still_playing = false;
                    return;
                }
                head_x -= 1;
            }
            Moves::Right => {
                if head_x == (BOARD_SIZE - 1) {
                    self.still_playing = false;
                    return;
                }
                head_x += 1;
            }
        }
        if matches!(self.board[head_y][head_x], Cell::Snake(Some(_))) {
            self.still_playing = false;
            return;
        }

        if (head_x, head_y) == self.apple {
            self.points += 1;
            if self.points == { BOARD_SIZE * BOARD_SIZE - 1 } {
                self.still_playing = false;
                return;
            }

            let (rand_x, rand_y) = loop {
                let rand_x = rand::thread_rng().gen_range(0..BOARD_SIZE);
                let rand_y = rand::thread_rng().gen_range(0..BOARD_SIZE);

                if matches!(self.board[rand_y][rand_x], Cell::Empty) {
                    break (rand_x, rand_y);
                }
            };

            self.board[rand_y][rand_x] = Cell::Apple;
            self.apple = (rand_x, rand_y);
        } else {
            // Update the tail
            let (mut tail_x, mut tail_y) = self.snake_tail;
            let direction = match self.board[self.snake_tail.1][self.snake_tail.0] {
                Cell::Apple => panic!(),
                Cell::Empty => panic!(),
                Cell::Snake(dir) => dir.unwrap(),
            };
            self.board[self.snake_tail.1][self.snake_tail.0] = Cell::Empty;
            match direction {
                Moves::Up => {
                    tail_y -= 1;
                }
                Moves::Down => {
                    tail_y += 1;
                }
                Moves::Left => {
                    tail_x -= 1;
                }
                Moves::Right => {
                    tail_x += 1;
                }
            }
            self.snake_tail = (tail_x, tail_y);
        }

        self.board[head_y][head_x] = Cell::Snake(None);
        self.snake_head = (head_x, head_y);
    }

    pub fn print(&self) {
        println!("Score: {}", self.points);
        for row in self.board {
            for x in row {
                print!("{}", x.print());
            }
            println!()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Moves {
    Up,
    Down,
    Left,
    Right,
}

impl Moves {
    pub fn to_usize(&self) -> usize {
        match self {
            Moves::Up => 0,
            Moves::Down => 1,
            Moves::Left => 2,
            Moves::Right => 3,
        }
    }
}
