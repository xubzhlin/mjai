use crate::rules::{ScoringRule, FanResult, KongType};
use crate::algo::winning::WinType;
use pyo3::prelude::*;

/// 计分规则实现
#[pyclass(name = "ScoringCalculator")]
#[derive(Clone)]
pub struct ScoringCalculator {
    pub max_fan: u32,
    pub self_draw_bonus: u32,
}

#[pymethods]
impl ScoringCalculator {
    #[new]
    pub fn new() -> Self {
        Self {
            max_fan: 16,
            self_draw_bonus: 1,
        }
    }

    /// Python 暴露：胡牌得分计算
    #[pyo3(name = "calculate_hu")]
    pub fn calculate_hu_py(&self, fan: u32, extra_bases: u32, win_type: &str, base: u32, num_payers: usize) -> i32 {
        let wt = parse_win_type(win_type);
        let result = FanResult { fan, extra_bases };
        <Self as ScoringRule>::calculate_hu(self, result, wt, base, num_payers)
    }

    /// Python 暴露：杠结算计算
    #[pyo3(name = "calculate_kong")]
    pub fn calculate_kong_py(&self, kong_type: &str, base: u32, num_payers: usize) -> i32 {
        let kt = match kong_type {
            "ankan" => KongType::AnKan,
            "minkan" => KongType::MinKan,
            "bukang" => KongType::BuKan,
            _ => KongType::MinKan,
        };
        <Self as ScoringRule>::calculate_kong(self, kt, base, num_payers)
    }

    #[getter]
    pub fn max_fan(&self) -> u32 { self.max_fan }

    #[getter]
    pub fn self_draw_bonus(&self) -> u32 { self.self_draw_bonus }
}

impl ScoringCalculator {
    pub fn new_with_params(max_fan: u32, self_draw_bonus: u32) -> Self {
        Self {
            max_fan,
            self_draw_bonus,
        }
    }
}

impl ScoringRule for ScoringCalculator {
    fn calculate_hu(&self, result: FanResult, _win_type: WinType, base: u32, num_payers: usize) -> i32 {
        let per_payer = (base as i32) * (result.fan as i32) + (result.extra_bases as i32) * (base as i32);
        per_payer * (num_payers as i32)
    }

    fn calculate_kong(&self, kong_type: KongType, base: u32, num_payers: usize) -> i32 {
        match kong_type {
            KongType::AnKan => (base as i32) * 2 * (num_payers as i32),
            KongType::MinKan => (base as i32) * 2,
            KongType::BuKan => (base as i32) * (num_payers as i32),
        }
    }
}

fn parse_win_type(s: &str) -> WinType {
    match s {
        "tsumo" => WinType::Tsumo,
        "ron" => WinType::Ron,
        "kanshang" => WinType::KanShang,
        "qianggang" => WinType::QiangGang,
        _ => WinType::Tsumo,
    }
}
