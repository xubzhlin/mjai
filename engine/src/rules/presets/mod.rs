pub mod xue_zhan;

pub use xue_zhan::XueZhanRuleSet;

/// 规则集预设工厂
pub struct RuleSetFactory;

impl RuleSetFactory {
    /// 创建血战到底规则集
    pub fn create_xue_zhan() -> XueZhanRuleSet {
        XueZhanRuleSet::new()
    }
    
    /// 创建自定义规则集
    pub fn create_custom(
        max_fan: u32,
        self_draw_bonus: u32,
        swap_enabled: bool,
        missing_enabled: bool,
        tian_que_threshold: f64,
    ) -> Box<dyn crate::rules::RuleSet> {
        Box::new(XueZhanRuleSet::new_with_params(
            max_fan,
            self_draw_bonus,
            swap_enabled,
            missing_enabled,
            tian_que_threshold,
        ))
    }
}