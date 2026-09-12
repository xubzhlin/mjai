//! 特征编码模块
//!
//! 将游戏状态编码为神经网络可处理的特征向量。
//! 通过 trait 抽象支持未来替换编码器实现。

pub mod encoder;

pub use encoder::*;

use crate::arena::Board;

/// 特征编码器抽象接口（可替换实现）
pub trait ObservationEncoderTrait: Send + Sync {
    /// 将指定玩家视角的完整牌桌状态编码为特征向量
    fn encode(&self, board: &Board, player_id: usize) -> Vec<f32>;

    /// 输出特征维度（展平后长度，= 通道数 × 27）
    fn feature_count(&self) -> usize;

    /// 通道数
    fn num_channels(&self) -> usize;
}
