//! 核心算法模块（通用，不随规则变化）
//! 
//! 包含听牌数计算、胡牌判定、预计算表等核心算法
//! 注意：番型计算已迁移至 `rules/fan.rs`（易变规则层）

pub mod shanten;
pub mod winning;
pub mod counter;

pub use shanten::*;
pub use winning::*;
pub use counter::*;