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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{FanRule, FanResult, KongType};
    use crate::algo::winning::WinType;

    fn sc() -> ScoringCalculator { ScoringCalculator::new() }

    // ===== Hu 计分 =====
    #[test]
    fn test_hu_formula_base() {
        // base=1, fan=2, extra_bases=1, num_payers=3 → (1*2 + 1*1)*3 = 9
        let result = FanResult { fan: 2, extra_bases: 1 };
        let total = sc().calculate_hu(result, WinType::Tsumo, 1, 3);
        assert_eq!(total, 9, "自摸 2 番 + 加底，3 家支付 = 9");
    }

    #[test]
    fn test_hu_ron_single_payer() {
        // base=1, fan=4, extra_bases=0, num_payers=1 → (1*4 + 0)*1 = 4
        let result = FanResult { fan: 4, extra_bases: 0 };
        let total = sc().calculate_hu(result, WinType::Ron, 1, 1);
        assert_eq!(total, 4, "点炮 4 番，独付 = 4");
    }

    #[test]
    fn test_hu_tsumo_extra_bases() {
        // base=2, fan=16 (封顶), extra_bases=1, num_payers=2 → (2*16 + 2*1)*2 = 68
        let result = FanResult { fan: 16, extra_bases: 1 };
        let total = sc().calculate_hu(result, WinType::Tsumo, 2, 2);
        assert_eq!(total, 68);
    }

    // ===== Kong 计分 =====
    #[test]
    fn test_kong_ankan() {
        // 暗杠：2 × base × num_payers → 2*1*3 = 6
        let total = sc().calculate_kong(KongType::AnKan, 1, 3);
        assert_eq!(total, 6);
    }

    #[test]
    fn test_kong_minkan() {
        // 明杠：2 × base → 2*1 = 2（点杠者独付，不乘 num_payers）
        let total = sc().calculate_kong(KongType::MinKan, 1, 3);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_kong_bukan() {
        // 补杠：1 × base × num_payers → 1*1*3 = 3
        let total = sc().calculate_kong(KongType::BuKan, 1, 3);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_kong_zero_sum_payers() {
        // AnKan 零和验证：winner 收 = 所有 payer 付
        let base = 1u32;
        let n = 3usize;
        let amount = sc().calculate_kong(KongType::AnKan, base, n);
        let per_payer = 2 * (base as i32);
        assert_eq!(amount, per_payer * (n as i32), "AnKan 总额 = 每家付 × 人数");
    }
}
