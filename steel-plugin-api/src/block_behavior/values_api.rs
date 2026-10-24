#[stabby::host_api]
pub mod api {
    #[cfg(feature = "host")]
    use crate::block_behavior::core_types::BonemealAction as NativeBonemealAction;
    use crate::block_behavior::types::{BonemealAction, Inventory, ItemStack, Player};
    use crate::{ApiResult, BlockState};
    use stabby::str::Str;
    #[cfg(feature = "host")]
    use stabby::string::String as ApiString;
    #[cfg(feature = "host")]
    use steel_registry::{REGISTRY, RegistryExt, item_stack::ItemStack as NativeItemStack};
    #[cfg(feature = "host")]
    use steel_utils::{BlockStateId, Identifier};

    impl ItemStack {
        /// Creates one item using the host's block-to-item association.
        /// Blocks without an associated item produce an empty stack.
        #[must_use]
        pub fn from_block(state: BlockState) -> ApiResult<Self> {
            let Some(block) = REGISTRY.blocks.by_state_id(BlockStateId(state.id)) else {
                return Err(ApiString::from("invalid block state")).into();
            };
            Ok(Self::from_host(NativeItemStack::new(
                REGISTRY.items.by_block(block),
            )))
            .into()
        }
    }

    impl<'a> Inventory<'a> {
        /// Tests the interacting hand's item against a host registry key.
        #[must_use]
        pub fn held_item_is(&self, key: Str<'_>) -> bool {
            let Ok(key) = key.as_str().parse::<Identifier>() else {
                return false;
            };
            let Some(item) = REGISTRY.items.by_key(&key) else {
                return false;
            };
            self.host().with_item(|stack| stack.is(item))
        }

        /// Applies durability damage, honoring the caller's creative-mode status.
        pub fn damage_item(&mut self, amount: i32, infinite_materials: bool) -> ApiResult<()> {
            if amount < 0 {
                return Err(ApiString::from("item damage must be nonnegative")).into();
            }
            self.host_mut()
                .with_item(|item| item.hurt_and_break(amount, infinite_materials));
            Ok(()).into()
        }
    }

    impl<'a> Player<'a> {
        /// Returns whether creative-mode material and durability rules apply.
        #[must_use]
        pub fn has_infinite_materials(&self) -> bool {
            self.host().has_infinite_materials()
        }
    }

    impl BonemealAction {
        /// Uses bonemeal to grow this block directly.
        #[must_use]
        pub fn grower() -> Self {
            Self::from_host(NativeBonemealAction::Grower)
        }

        /// Uses bonemeal to spread growth to neighboring blocks.
        #[must_use]
        pub fn neighbor_spreader() -> Self {
            Self::from_host(NativeBonemealAction::NeighborSpreader)
        }
    }
}
