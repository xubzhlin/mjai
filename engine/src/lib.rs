//! 四川麻将引擎 Rust 核心库
//!
//! 提供高性能的麻将规则引擎和游戏状态管理
//! PyO3 绑定已启用。

// ===== 核心模块 =====
pub mod tile;
pub mod algo;
pub mod state;
pub mod arena;
pub mod phases;
pub mod observation;
pub mod rules;

// ===== 重导出 =====
pub use tile::{Hand, Tile, Suit};
pub use algo::{ShantenResult, WinChecker, WinType, WinMethod, WinResult};
pub use state::*;
pub use arena::{Board, Game, Wall};
pub use phases::*;
pub use observation::ObservationEncoder;
pub use rules::*;

// ===== PyO3 绑定 =====
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn register_rules_submodule<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    let sub = PyModule::new_bound(py, "rules")?;
    sub.add_class::<crate::rules::fan::FanCalculator>()?;
    sub.add_class::<crate::rules::scoring::ScoringCalculator>()?;
    sub.add_class::<crate::rules::swap::SwapRuleCalculator>()?;
    sub.add_class::<crate::rules::missing::MissingRuleCalculator>()?;
    sub.add_class::<crate::rules::presets::xue_zhan::XueZhanRuleSet>()?;
    Ok(sub)
}

fn register_placeholder_submodule<'py>(
    py: Python<'py>,
    name: &'static str,
) -> PyResult<Bound<'py, PyModule>> {
    PyModule::new_bound(py, name)
}

/// 主入口：#[pymodule] mjai_engine
#[pymodule]
fn mjai_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // 注册子模块
    let sub_rules = register_rules_submodule(py)?;
    let sub_tile = register_placeholder_submodule(py, "tile")?;
    let sub_algo = register_placeholder_submodule(py, "algo")?;
    let sub_state = register_placeholder_submodule(py, "state")?;
    let sub_arena = register_placeholder_submodule(py, "arena")?;
    let sub_observation = register_placeholder_submodule(py, "observation")?;

    m.add_submodule(&sub_rules)?;
    m.add_submodule(&sub_tile)?;
    m.add_submodule(&sub_algo)?;
    m.add_submodule(&sub_state)?;
    m.add_submodule(&sub_arena)?;
    m.add_submodule(&sub_observation)?;

    // 顶层也暴露常用类
    m.add_class::<crate::rules::fan::FanCalculator>()?;
    m.add_class::<crate::rules::scoring::ScoringCalculator>()?;
    m.add_class::<crate::rules::presets::xue_zhan::XueZhanRuleSet>()?;

    // 用 sys.modules 注册，支持 `from mjai_engine.rules import XueZhanRuleSet`
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("mjai_engine", m)?;
    modules.set_item("mjai_engine.rules", &sub_rules)?;
    modules.set_item("mjai_engine.tile", &sub_tile)?;
    modules.set_item("mjai_engine.algo", &sub_algo)?;
    modules.set_item("mjai_engine.state", &sub_state)?;
    modules.set_item("mjai_engine.arena", &sub_arena)?;
    modules.set_item("mjai_engine.observation", &sub_observation)?;

    Ok(())
}
