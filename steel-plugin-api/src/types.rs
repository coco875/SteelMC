#[stabby::stabby]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionResult {
    /// The interaction succeeded and consumed the action.
    Success,
    /// The interaction succeeded and the server should broadcast the swing.
    SuccessServer,
    /// The interaction consumed the action without swinging.
    Consume,
    /// The interaction failed and consumed the action.
    Fail,
    /// The interaction did not apply; try the next handler.
    Pass,
    /// Try the empty-hand interaction on the block.
    TryEmptyHandInteraction,
}

#[stabby::stabby]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandSenderType {
    Player = 0,
    Console = 1,
    Rcon = 2,
}

#[stabby::stabby]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PluginUuid {
    pub bytes: [u8; 16],
}

impl From<[u8; 16]> for PluginUuid {
    fn from(bytes: [u8; 16]) -> Self {
        Self { bytes }
    }
}

impl From<PluginUuid> for [u8; 16] {
    fn from(uuid: PluginUuid) -> Self {
        uuid.bytes
    }
}
