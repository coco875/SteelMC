use crate::{ai, api, map};
use serde::Deserialize;
use steel_plugin_api::{
    BlockPos, Difficulty, GameMode, Operation, OperationState, PlayerId, PlayerInfo, Position,
    Server, WorldSettings,
};

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub lobby_seconds: u64,
    pub round_seconds: u64,
    pub ending_seconds: u64,
    pub players: usize,
    pub seed: i64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            lobby_seconds: 20,
            round_seconds: 180,
            ending_seconds: 5,
            players: 4,
            seed: 0,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if !(2..=8).contains(&self.players) {
            return Err("Spleef players must be between 2 and 8".into());
        }
        if !(1..=300).contains(&self.lobby_seconds)
            || !(10..=1800).contains(&self.round_seconds)
            || !(1..=30).contains(&self.ending_seconds)
        {
            return Err("Spleef timers are outside their supported bounds".into());
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    PreparingLobby,
    Lobby { deadline: Option<u64> },
    PreparingArena,
    Transferring { deadline: u64 },
    Playing { deadline: u64 },
    Ending { deadline: u64 },
    Returning,
    Retiring,
}
impl Phase {
    const fn label(self) -> &'static str {
        match self {
            Self::Idle => "ready",
            Self::PreparingLobby => "preparing the lobby",
            Self::Lobby { .. } => "waiting in the lobby",
            Self::PreparingArena => "building the arena",
            Self::Transferring { .. } => "entering the arena",
            Self::Playing { .. } => "playing",
            Self::Ending { .. } => "round finished",
            Self::Returning => "returning players",
            Self::Retiring => "cleaning up",
        }
    }
}
struct Member {
    id: PlayerId,
    home: String,
    join_sent: bool,
    arrived: bool,
    leaving: bool,
    return_sent: bool,
    done: bool,
}
struct Contestant {
    id: PlayerId,
    bot: bool,
    transfer_sent: bool,
    ready: bool,
    alive: bool,
}
pub struct Game {
    config: Config,
    next_room: u64,
    phase: Phase,
    domain: String,
    lobby: String,
    arena: String,
    members: Vec<Member>,
    bots: Vec<PlayerId>,
    contestants: Vec<Contestant>,
    operation: Option<Operation>,
    domain_created: bool,
}
impl Game {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            next_room: 0,
            phase: Phase::Idle,
            domain: String::new(),
            lobby: String::new(),
            arena: String::new(),
            members: Vec::new(),
            bots: Vec::new(),
            contestants: Vec::new(),
            operation: None,
            domain_created: false,
        }
    }
    pub fn status(&self) -> String {
        format!(
            "Spleef: {}; {}/{} human players. /spleef join | leave | status",
            self.phase.label(),
            self.members.iter().filter(|m| !m.leaving).count(),
            self.config.players
        )
    }
    pub fn join(&mut self, server: &Server<'_>, id: PlayerId) -> Result<(), String> {
        if self.members.iter().any(|m| m.id == id) {
            return Err("You are already in Spleef; use /spleef leave".into());
        }
        if !matches!(
            self.phase,
            Phase::Idle | Phase::PreparingLobby | Phase::Lobby { .. }
        ) {
            return Err("A round is already running. Join after it finishes.".into());
        }
        if self.members.len() >= self.config.players {
            return Err("The Spleef room is full".into());
        }
        let players = server.players();
        let player = info(players.as_slice(), id)
            .ok_or("Player is changing worlds; try joining when it completes")?;
        let home = domain(player.world.as_str()).to_owned();
        if server.domain_is_temporary(home.as_str().into()) {
            return Err("Leave your current temporary game before joining Spleef".into());
        }
        if !server.player_ready(id) {
            return Err("Wait until your current world has loaded".into());
        }
        if self.phase == Phase::Idle {
            self.next_room = self
                .next_room
                .checked_add(1)
                .ok_or("Spleef room identities exhausted")?;
            self.domain = format!("spleef_{}", self.next_room);
            self.lobby = format!("{}:lobby", self.domain);
            self.arena = format!("{}:arena", self.domain);
            let operation = api!(server.create_temporary_world(
                self.lobby.as_str().into(),
                "spleef:lobby".into(),
                WorldSettings {
                    seed: self.config.seed,
                    spawn: spawn(),
                    default_game_mode: GameMode::Adventure,
                    difficulty: Difficulty::Peaceful
                },
                true
            ))?;
            self.operation = Some(operation);
            self.phase = Phase::PreparingLobby;
        }
        self.members.push(Member {
            id,
            home,
            join_sent: false,
            arrived: false,
            leaving: false,
            return_sent: false,
            done: false,
        });
        Ok(())
    }
    pub fn leave(&mut self, id: PlayerId) -> Result<(), String> {
        let member = self
            .members
            .iter_mut()
            .find(|m| m.id == id)
            .ok_or("You are not in Spleef")?;
        member.leaving = true;
        if let Some(contestant) = self.contestants.iter_mut().find(|p| p.id == id) {
            contestant.alive = false;
        }
        Ok(())
    }
    pub fn begin_return(&mut self) {
        if matches!(self.phase, Phase::Idle | Phase::Retiring | Phase::Returning) {
            return;
        }
        for member in &mut self.members {
            member.leaving = true;
        }
        self.phase = Phase::Returning;
    }
    fn poll_creation(&mut self, server: &Server<'_>) -> Result<bool, String> {
        let Some(operation) = self.operation else {
            return Err("Spleef world operation is missing".into());
        };
        let status = api!(server.operation_status(operation))?;
        match status.state {
            OperationState::Pending => Ok(false),
            OperationState::Ready => {
                self.operation = None;
                self.domain_created = true;
                Ok(true)
            }
            OperationState::Failed => {
                self.operation = None;
                Err(status.value.to_string())
            }
        }
    }
    fn update_members(
        &mut self,
        server: &Server<'_>,
        players: &[PlayerInfo],
    ) -> Result<(), String> {
        for member in &mut self.members {
            if !server.is_connected(member.id) {
                member.done = true;
                continue;
            }
            if member.leaving && !member.join_sent {
                member.done = true;
                continue;
            }
            let Some(player) = info(players, member.id) else {
                continue;
            };
            if !server.player_ready(member.id) {
                continue;
            }
            let in_game = domain(player.world.as_str()) == self.domain;
            if member.arrived && !in_game {
                member.done = true;
                continue;
            }
            if member.leaving {
                if !in_game {
                    member.done = true;
                    continue;
                }
                if !member.return_sent {
                    api!(server.switch_domain(member.id, member.home.as_str().into()))?;
                    member.return_sent = true;
                }
                continue;
            }
            if !member.join_sent && self.domain_created {
                api!(server.switch_domain(member.id, self.domain.as_str().into()))?;
                member.join_sent = true;
                continue;
            }
            if in_game && !member.arrived {
                api!(server.set_game_mode(member.id, GameMode::Adventure))?;
                api!(server.clear_inventory(member.id))?;
                api!(server.message(member.id,"Welcome to Spleef! Dig snow under your opponents; falling eliminates you. /spleef leave restores your normal world.".into()))?;
                member.arrived = true;
            }
        }
        self.members.retain(|member| !member.done);
        Ok(())
    }
    fn fill_bots(&mut self, server: &Server<'_>) -> Result<(), String> {
        self.bots.retain(|id| server.is_connected(*id));
        let desired = self.config.players - self.members.iter().filter(|m| !m.leaving).count();
        while self.bots.len() > desired {
            if let Some(id) = self.bots.pop() {
                api!(server.despawn_bot(id))?;
            }
        }
        while self.bots.len() < desired {
            let number = self.bots.len() + 1;
            let name = format!("SpleefBot{number}");
            let id = api!(server.spawn_bot(
                self.lobby.as_str().into(),
                name.as_str().into(),
                start_position(number)
            ))?;
            api!(server.set_game_mode(id, GameMode::Adventure))?;
            self.bots.push(id);
        }
        Ok(())
    }
    fn transfer_contestants(
        &mut self,
        server: &Server<'_>,
        players: &[PlayerInfo],
    ) -> Result<bool, String> {
        for (index, actor) in self.contestants.iter_mut().enumerate() {
            if !server.is_connected(actor.id) {
                actor.alive = false;
                continue;
            }
            let Some(player) = info(players, actor.id) else {
                continue;
            };
            if !server.player_ready(actor.id) {
                continue;
            }
            if player.world.as_str() != self.arena {
                if !actor.transfer_sent {
                    api!(server.select_world(actor.id, self.arena.as_str().into()))?;
                    actor.transfer_sent = true;
                }
                continue;
            }
            if !actor.ready {
                api!(server.teleport(actor.id, start_position(index)))?;
                api!(server.clear_inventory(actor.id))?;
                api!(server.equip_item(actor.id, "minecraft:diamond_shovel".into()))?;
                api!(server.set_game_mode(actor.id, GameMode::Survival))?;
                actor.ready = true;
            }
        }
        Ok(self.contestants.iter().all(|p| !p.alive || p.ready))
    }
    fn announce(&self, server: &Server<'_>, message: &str) -> Result<(), String> {
        for member in &self.members {
            if server.is_connected(member.id) {
                api!(server.message(member.id, message.into()))?;
            }
        }
        Ok(())
    }
    fn play(
        &mut self,
        server: &Server<'_>,
        players: &[PlayerInfo],
        tick: u64,
    ) -> Result<(), String> {
        for actor in &mut self.contestants {
            if !actor.alive {
                continue;
            }
            let player = info(players, actor.id);
            let lost = player.is_none_or(|p| {
                p.world.as_str() != self.arena
                    || p.position.y < f64::from(map::FLOOR - 8)
                    || p.health <= 0.0
            });
            if !lost {
                continue;
            }
            actor.alive = false;
            if let Some(player) = player
                && player.world.as_str() == self.arena
                && player.health > 0.0
            {
                api!(server.set_game_mode(actor.id, GameMode::Spectator))?;
                api!(server.teleport(
                    actor.id,
                    Position {
                        x: 0.5,
                        y: f64::from(map::FLOOR + 12),
                        z: 0.5
                    }
                ))?;
                if !actor.bot {
                    api!(
                        server.message(
                            actor.id,
                            "Eliminated! You can watch until the round ends or use /spleef leave."
                                .into()
                        )
                    )?;
                }
            }
        }
        let alive: Vec<_> = self
            .contestants
            .iter()
            .filter(|p| p.alive)
            .filter_map(|p| info(players, p.id))
            .collect();
        for actor in self.contestants.iter().filter(|p| p.alive && p.bot) {
            if let Some(player) = info(players, actor.id) {
                ai::step(server, &self.arena, player, &alive, tick)?;
            }
        }
        if alive.len() <= 1 {
            let text = alive.first().map_or_else(
                || "Spleef ended in a draw.".to_owned(),
                |p| format!("{} wins Spleef!", p.name),
            );
            self.announce(server, &text)?;
            self.phase = Phase::Ending {
                deadline: tick + self.config.ending_seconds * 20,
            };
        }
        Ok(())
    }
    pub fn tick(&mut self, server: &Server<'_>, tick: u64) -> Result<(), String> {
        if self.phase == Phase::Idle {
            return Ok(());
        }
        let players = server.players();
        let players = players.as_slice();
        self.update_members(server, players)?;
        if self.members.iter().all(|m| m.leaving)
            && !matches!(self.phase, Phase::Returning | Phase::Retiring)
        {
            self.begin_return();
        }
        match self.phase {
            Phase::Idle => {}
            Phase::PreparingLobby => {
                if self.poll_creation(server)? {
                    self.phase = Phase::Lobby { deadline: None };
                }
            }
            Phase::Lobby { mut deadline } => {
                if self.members.iter().any(|m| m.arrived && !m.leaving) {
                    self.fill_bots(server)?;
                    if deadline.is_none() {
                        deadline = Some(tick + self.config.lobby_seconds * 20);
                        self.announce(
                            server,
                            &format!(
                                "Spleef starts in {} seconds. Empty places are filled with bots.",
                                self.config.lobby_seconds
                            ),
                        )?;
                    }
                }
                if deadline.is_some_and(|end| tick >= end) {
                    self.operation = Some(api!(server.create_temporary_world(
                        self.arena.as_str().into(),
                        "spleef:arena".into(),
                        WorldSettings {
                            seed: self.config.seed,
                            spawn: spawn(),
                            default_game_mode: GameMode::Survival,
                            difficulty: Difficulty::Peaceful
                        },
                        false
                    ))?);
                    self.phase = Phase::PreparingArena;
                } else {
                    self.phase = Phase::Lobby { deadline };
                }
            }
            Phase::PreparingArena => {
                if self.poll_creation(server)? {
                    self.contestants = self
                        .members
                        .iter()
                        .filter(|m| !m.leaving)
                        .map(|m| Contestant {
                            id: m.id,
                            bot: false,
                            transfer_sent: false,
                            ready: false,
                            alive: true,
                        })
                        .chain(self.bots.iter().map(|id| Contestant {
                            id: *id,
                            bot: true,
                            transfer_sent: false,
                            ready: false,
                            alive: true,
                        }))
                        .collect();
                    self.phase = Phase::Transferring {
                        deadline: tick + 600,
                    };
                }
            }
            Phase::Transferring { deadline } => {
                if self.transfer_contestants(server, players)? {
                    self.announce(server, "Spleef begins! Last player on the snow wins.")?;
                    self.phase = Phase::Playing {
                        deadline: tick + self.config.round_seconds * 20,
                    };
                } else if tick >= deadline {
                    self.announce(
                        server,
                        "Spleef cancelled: players did not finish loading the arena.",
                    )?;
                    self.begin_return();
                }
            }
            Phase::Playing { deadline } => {
                self.play(server, players, tick)?;
                if tick >= deadline && matches!(self.phase, Phase::Playing { .. }) {
                    self.announce(server, "Time is up! Spleef ended in a draw.")?;
                    self.phase = Phase::Ending {
                        deadline: tick + self.config.ending_seconds * 20,
                    };
                }
            }
            Phase::Ending { deadline } => {
                if tick >= deadline {
                    self.begin_return();
                }
            }
            Phase::Returning => {
                for id in std::mem::take(&mut self.bots) {
                    if server.is_connected(id) {
                        api!(server.despawn_bot(id))?;
                    }
                }
                if self.operation.is_some() {
                    let _ = self.poll_creation(server)?;
                    return Ok(());
                }
                if self.members.is_empty() {
                    if self.domain_created {
                        self.operation = Some(api!(
                            server.destroy_temporary_domain(self.domain.as_str().into())
                        )?);
                        self.phase = Phase::Retiring;
                    } else {
                        self.reset();
                    }
                }
            }
            Phase::Retiring => {
                let Some(operation) = self.operation else {
                    return Err("Spleef cleanup operation missing".into());
                };
                let status = api!(server.operation_status(operation))?;
                match status.state {
                    OperationState::Pending => {}
                    OperationState::Ready => self.reset(),
                    OperationState::Failed => return Err(status.value.to_string()),
                }
            }
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.operation = None;
        self.domain_created = false;
        self.members.clear();
        self.bots.clear();
        self.contestants.clear();
        self.domain.clear();
        self.lobby.clear();
        self.arena.clear();
    }
}
fn domain(world: &str) -> &str {
    world.split_once(':').map_or(world, |(domain, _)| domain)
}
fn info(players: &[PlayerInfo], id: PlayerId) -> Option<&PlayerInfo> {
    players.iter().find(|p| p.id == id)
}
fn spawn() -> BlockPos {
    BlockPos {
        x: 0,
        y: map::FLOOR + 1,
        z: 0,
    }
}
fn start_position(index: usize) -> Position {
    const POSITIONS: [(f64, f64); 8] = [
        (-7.5, -7.5),
        (7.5, 7.5),
        (-7.5, 7.5),
        (7.5, -7.5),
        (0.5, -9.5),
        (0.5, 9.5),
        (-9.5, 0.5),
        (9.5, 0.5),
    ];
    let (x, z) = POSITIONS[index % POSITIONS.len()];
    Position {
        x,
        y: f64::from(map::FLOOR + 1),
        z,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_spawn_slots_are_on_the_authored_floor_with_room_for_walls() {
        for i in 0..8 {
            let p = start_position(i);
            assert!(
                p.x.abs() < f64::from(map::RADIUS - 1) && p.z.abs() < f64::from(map::RADIUS - 1)
            );
        }
    }
}
