//! Permission checks for commands that operate on player targets.

use crate::{
    command::execution::{CommandPermissionSource, CommandSource},
    permission::{PermissionExpr, PermissionKey, PermissionKeyError},
    player::Player,
};

pub(crate) fn player_target_access(root: &PermissionKey) -> PermissionExpr {
    PermissionExpr::key(root.clone()) | PermissionExpr::any_descendant(root.clone())
}

/// Requires every effective target group, so additional groups protect a player.
pub(crate) fn can_target_player(
    source: &CommandSource,
    root: &PermissionKey,
    target: &Player,
) -> Result<bool, PermissionKeyError> {
    if source
        .player()
        .is_some_and(|player| player.gameprofile.id == target.gameprofile.id)
        && source.has_permission(&self_permission(root)?)
    {
        return Ok(true);
    }

    let assigned = source
        .server()
        .player_permission_state(target.gameprofile.id)
        .unwrap_or_default();
    let groups = source
        .server()
        .permission_groups
        .effective_group_names(assigned.groups());
    if groups.is_empty() {
        return Ok(false);
    }
    for group in groups {
        if !source.has_permission(&group_permission(root, &group)?) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) fn self_permission(root: &PermissionKey) -> Result<PermissionExpr, PermissionKeyError> {
    let key = root.child("self")?;
    Ok(PermissionExpr::scoped_key(root.clone(), key))
}

pub(crate) fn group_permission(
    root: &PermissionKey,
    group: &str,
) -> Result<PermissionExpr, PermissionKeyError> {
    let group_root = root.child("group")?;
    let key = group_root.child(group)?;
    Ok(PermissionExpr::scoped_key(root.clone(), key))
}
